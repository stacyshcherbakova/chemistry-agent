"""Orchestrator tests with stub agents — no SDK, no network."""

import pytest

from chemistry_agent.agents.orchestrator import Orchestrator
from chemistry_agent.config import AgentConfig
from chemistry_agent.models import (
    FinalAnswer,
    Plan,
    PlanStep,
    UsageTotals,
    Verdict,
)


def make_plan(difficulty="gcse"):
    return Plan(
        steps=[PlanStep(description="look up molar mass", suggested_tool="molar_mass")],
        target_quantity="molar mass of H2O",
        expected_units="g/mol",
        difficulty=difficulty,
        topic="stoichiometry",
    )


def make_answer(text="18.02 g/mol"):
    return FinalAnswer(
        value=18.02, units="g/mol", formatted_answer=text, reasoning_summary="looked up"
    )


class StubPlanner:
    def __init__(self, plan=None, error=None):
        self.usage = UsageTotals(input_tokens=100, output_tokens=50)
        self._plan = plan or make_plan()
        self._error = error
        self.calls = []

    def plan(self, prompt, level_hint=None):
        self.calls.append((prompt, level_hint))
        if self._error:
            raise self._error
        return self._plan


class StubSolver:
    def __init__(self, results=None):
        """results: list of FinalAnswer | Exception, one per attempt."""
        self.usage = UsageTotals(input_tokens=1000, output_tokens=500)
        self.results = list(results or [make_answer()])
        self.feedback_seen = []
        self.efforts = []

    def solve(self, prompt, plan, effort, feedback_history=None, on_tool_call=None):
        self.efforts.append(effort)
        self.feedback_seen.append(list(feedback_history or []))
        if on_tool_call:
            on_tool_call("molar_mass")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result, ["molar_mass"]


class StubVerifier:
    def __init__(self, verdicts=None, error=None):
        self.usage = UsageTotals(input_tokens=200, output_tokens=100)
        self.verdicts = list(
            verdicts
            or [Verdict(approved=True, unit_check=True, magnitude_plausible=True)]
        )
        self._error = error

    def verify(self, prompt, plan, answer, tool_calls):
        if self._error:
            raise self._error
        return self.verdicts.pop(0)


def rejected(feedback="units are wrong"):
    return Verdict(
        approved=False,
        unit_check=False,
        magnitude_plausible=True,
        issues=["bad units"],
        feedback_for_solver=feedback,
    )


def approved():
    return Verdict(approved=True, unit_check=True, magnitude_plausible=True)


class TestHappyPath:
    def test_first_attempt_success(self):
        orch = Orchestrator(StubPlanner(), StubSolver(), StubVerifier())
        outcome = orch.solve("molar mass of water?")
        assert outcome.success is True
        assert outcome.attempts == 1
        assert outcome.answer.formatted_answer == "18.02 g/mol"
        assert outcome.tool_calls == ["molar_mass"]
        assert outcome.error is None

    def test_effort_follows_plan_difficulty(self):
        solver = StubSolver()
        orch = Orchestrator(
            StubPlanner(plan=make_plan("university")), solver, StubVerifier()
        )
        orch.solve("hard problem")
        assert solver.efforts == [AgentConfig().effort_for("university")]

    def test_level_hint_forwarded_to_planner(self):
        planner = StubPlanner()
        orch = Orchestrator(planner, StubSolver(), StubVerifier())
        orch.solve("p", level_hint="alevel")
        assert planner.calls == [("p", "alevel")]


class TestRetryLoop:
    def test_rejected_then_approved(self):
        solver = StubSolver(results=[make_answer("bad"), make_answer("good")])
        verifier = StubVerifier(verdicts=[rejected("fix units"), approved()])
        orch = Orchestrator(StubPlanner(), solver, verifier)
        outcome = orch.solve("p")
        assert outcome.success is True
        assert outcome.attempts == 2
        # second attempt saw the first verdict's feedback
        assert solver.feedback_seen == [[], ["fix units"]]
        assert outcome.answer.formatted_answer == "good"

    def test_all_attempts_rejected(self):
        cfg = AgentConfig()
        n = cfg.max_retries + 1
        solver = StubSolver(results=[make_answer(f"a{i}") for i in range(n)])
        verifier = StubVerifier(verdicts=[rejected() for _ in range(n)])
        outcome = Orchestrator(StubPlanner(), solver, verifier, config=cfg).solve("p")
        assert outcome.success is False
        assert outcome.attempts == n
        assert outcome.verdict.approved is False
        # answer still surfaced even though never approved
        assert outcome.answer is not None

    def test_solver_error_retried_then_success(self):
        solver = StubSolver(results=[RuntimeError("boom"), make_answer()])
        verifier = StubVerifier(verdicts=[approved()])
        outcome = Orchestrator(StubPlanner(), solver, verifier).solve("p")
        assert outcome.success is True
        assert outcome.attempts == 2
        assert "boom" in solver.feedback_seen[1][0]

    def test_solver_fails_every_attempt(self):
        cfg = AgentConfig()
        n = cfg.max_retries + 1
        solver = StubSolver(results=[RuntimeError("boom") for _ in range(n)])
        outcome = Orchestrator(StubPlanner(), solver, StubVerifier(), config=cfg).solve(
            "p"
        )
        assert outcome.success is False
        assert "boom" in outcome.error
        assert outcome.answer is None

    def test_planner_error_short_circuits(self):
        planner = StubPlanner(error=RuntimeError("refused"))
        outcome = Orchestrator(planner, StubSolver(), StubVerifier()).solve("p")
        assert outcome.success is False
        assert outcome.plan is None
        assert "planner" in outcome.error

    def test_verifier_error_surfaces_answer_unverified(self):
        verifier = StubVerifier(error=RuntimeError("verifier down"))
        outcome = Orchestrator(StubPlanner(), StubSolver(), verifier).solve("p")
        assert outcome.success is False
        assert outcome.answer is not None
        assert "verifier" in outcome.error


class TestAccounting:
    def test_usage_and_cost(self):
        cfg = AgentConfig()
        outcome = Orchestrator(
            StubPlanner(), StubSolver(), StubVerifier(), config=cfg
        ).solve("p")
        assert set(outcome.usage) == {"planner", "solver", "verifier"}
        assert outcome.usage["solver"].input_tokens == 1000
        inp, out = cfg.price_for(cfg.solver_model)
        expected = (
            UsageTotals(input_tokens=100, output_tokens=50).cost_usd(inp, out)
            + UsageTotals(input_tokens=1000, output_tokens=500).cost_usd(inp, out)
            + UsageTotals(input_tokens=200, output_tokens=100).cost_usd(inp, out)
        )
        assert outcome.cost_usd == pytest.approx(expected, abs=1e-6)


class TestObserver:
    def test_event_sequence(self):
        events = []
        orch = Orchestrator(
            StubPlanner(),
            StubSolver(results=[make_answer("bad"), make_answer("good")]),
            StubVerifier(verdicts=[rejected(), approved()]),
            observer=lambda e: events.append(e.kind),
        )
        outcome = orch.solve("p")
        assert outcome.success
        assert events == [
            "plan_ready",
            "attempt_start",
            "tool_call",
            "answer_submitted",
            "verdict",
            "attempt_start",
            "tool_call",
            "answer_submitted",
            "verdict",
            "done",
        ]

    def test_broken_observer_does_not_crash(self):
        def bad_observer(event):
            raise ValueError("ui exploded")

        outcome = Orchestrator(
            StubPlanner(), StubSolver(), StubVerifier(), observer=bad_observer
        ).solve("p")
        assert outcome.success is True
