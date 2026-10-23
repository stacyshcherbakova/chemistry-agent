"""Orchestrator: plain-Python control flow over the agent team.

plan -> [solve -> verify] x up to (max_retries + 1) attempts. Verifier
feedback is fed back to the solver as a new user message on retry, keeping
the solver's system prompt and tool list byte-stable for prompt caching.
"""

from typing import Optional

from ..config import AgentConfig
from ..models import Observer, SolveEvent, SolveOutcome, UsageTotals
from .planner import Planner
from .solver import Solver
from .verifier import Verifier


class Orchestrator:
    """Coordinates Planner, Solver, and Verifier into one solve loop.

    Agents are injected via the constructor so tests can pass stubs and the
    eval harness can swap solver models without touching this class.
    """

    def __init__(
        self,
        planner: Optional[Planner] = None,
        solver: Optional[Solver] = None,
        verifier: Optional[Verifier] = None,
        config: Optional[AgentConfig] = None,
        observer: Optional[Observer] = None,
    ):
        self.config = config or AgentConfig()
        self.planner = planner if planner is not None else Planner(config=self.config)
        self.solver = solver if solver is not None else Solver(config=self.config)
        self.verifier = (
            verifier if verifier is not None else Verifier(config=self.config)
        )
        self.observer = observer

    def _emit(self, kind: str, **payload) -> None:
        if self.observer is None:
            return
        try:
            self.observer(SolveEvent(kind=kind, payload=payload))
        except Exception:
            pass  # a broken UI callback must never kill a solve

    def solve(self, prompt: str, level_hint: Optional[str] = None) -> SolveOutcome:
        outcome = SolveOutcome(prompt=prompt)

        try:
            plan = self.planner.plan(prompt, level_hint=level_hint)
        except Exception as exc:
            outcome.error = f"planner: {exc}"
            self._emit("error", stage="planner", message=str(exc))
            self._finalize(outcome)
            return outcome

        outcome.plan = plan
        effort = self.config.effort_for(plan.difficulty)
        self._emit(
            "plan_ready",
            steps=[s.description for s in plan.steps],
            target_quantity=plan.target_quantity,
            expected_units=plan.expected_units,
            difficulty=plan.difficulty,
            effort=effort,
        )

        feedback_history: list[str] = []
        max_attempts = self.config.max_retries + 1
        last_error: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            outcome.attempts = attempt
            self._emit("attempt_start", attempt=attempt, max_attempts=max_attempts)

            try:
                answer, tool_calls = self.solver.solve(
                    prompt,
                    plan,
                    effort,
                    feedback_history=feedback_history or None,
                    on_tool_call=lambda name: self._emit("tool_call", name=name),
                )
            except Exception as exc:
                last_error = f"solver: {exc}"
                self._emit("error", stage="solver", attempt=attempt, message=str(exc))
                feedback_history.append(
                    f"The previous attempt failed with an error: {exc}. "
                    "Please solve the problem again from the beginning."
                )
                continue

            outcome.tool_calls.extend(tool_calls)
            outcome.answer = answer
            self._emit(
                "answer_submitted",
                attempt=attempt,
                formatted_answer=answer.formatted_answer,
            )

            if not self.config.use_verifier:
                # Ablation mode: accept the first answer without an audit.
                outcome.success = True
                last_error = None
                break

            try:
                verdict = self.verifier.verify(prompt, plan, answer, tool_calls)
            except Exception as exc:
                last_error = f"verifier: {exc}"
                self._emit("error", stage="verifier", attempt=attempt, message=str(exc))
                # Answer stands unverified; do not retry blindly on a verifier
                # failure — surface the answer with the error recorded.
                break

            outcome.verdict = verdict
            self._emit(
                "verdict",
                attempt=attempt,
                approved=verdict.approved,
                issues=verdict.issues,
            )

            if verdict.approved:
                outcome.success = True
                last_error = None
                break

            feedback_history.append(
                verdict.feedback_for_solver
                or "; ".join(verdict.issues)
                or "Answer rejected; re-derive carefully."
            )

        outcome.error = last_error
        self._finalize(outcome)
        return outcome

    def _finalize(self, outcome: SolveOutcome) -> None:
        roles = {
            "planner": (self.planner, self.config.planner_model),
            "solver": (self.solver, self.config.solver_model),
            "verifier": (self.verifier, self.config.verifier_model),
        }
        total = 0.0
        for role, (agent, model) in roles.items():
            usage = getattr(agent, "usage", None) or UsageTotals()
            outcome.usage[role] = usage
            inp, out = self.config.price_for(model)
            total += usage.cost_usd(inp, out)
        outcome.cost_usd = round(total, 6)
        self._emit(
            "done",
            success=outcome.success,
            attempts=outcome.attempts,
            cost_usd=outcome.cost_usd,
        )
