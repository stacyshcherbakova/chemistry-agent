"""CLI tests — stubbed Orchestrator and evals modules, no SDK or network."""

import json
import os

import pytest
from typer.testing import CliRunner

from chemistry_agent import cli
from chemistry_agent.models import (
    FinalAnswer,
    Plan,
    PlanStep,
    SolveEvent,
    SolveOutcome,
    UsageTotals,
    Verdict,
)

runner = CliRunner()
ENV = {"ANTHROPIC_API_KEY": "test-key"}


def make_outcome(success=True, answer_text="18.02 g/mol", error=None):
    return SolveOutcome(
        prompt="p",
        plan=Plan(
            steps=[PlanStep(description="look up molar mass")],
            target_quantity="molar mass",
            expected_units="g/mol",
            difficulty="gcse",
            topic="stoichiometry",
        ),
        answer=FinalAnswer(
            value=18.02,
            units="g/mol",
            formatted_answer=answer_text,
            reasoning_summary="summed atomic masses",
        ),
        verdict=Verdict(approved=success, unit_check=True, magnitude_plausible=True),
        attempts=1,
        tool_calls=["lookup_molar_mass"],
        usage={"solver": UsageTotals(input_tokens=100, output_tokens=50)},
        cost_usd=0.0125,
        success=success,
        error=error,
    )


class StubOrchestrator:
    """Records constructor kwargs; returns a scripted outcome."""

    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.solve_calls = []
        StubOrchestrator.instances.append(self)

    def solve(self, prompt, level_hint=None):
        self.solve_calls.append((prompt, level_hint))
        return make_outcome(**getattr(StubOrchestrator, "outcome_kwargs", {}))


@pytest.fixture(autouse=True)
def stub_orchestrator(monkeypatch):
    StubOrchestrator.instances = []
    StubOrchestrator.outcome_kwargs = {}
    monkeypatch.setattr(cli, "Orchestrator", StubOrchestrator)
    yield StubOrchestrator


class TestSolve:
    def test_happy_path(self):
        result = runner.invoke(cli.app, ["solve", "molar mass of H2O?"], env=ENV)
        assert result.exit_code == 0
        assert "18.02 g/mol" in result.output
        assert "0.0125" in result.output

    def test_failure_exits_nonzero(self):
        StubOrchestrator.outcome_kwargs = {"success": False, "error": "solver: boom"}
        result = runner.invoke(cli.app, ["solve", "p"], env=ENV)
        assert result.exit_code == 1
        assert "boom" in result.output

    def test_missing_api_key_blocks_before_constructing_orchestrator(
        self, monkeypatch, tmp_path
    ):
        # run outside the repo so a developer's local .env cannot supply a key
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        result = runner.invoke(cli.app, ["solve", "p"])
        assert result.exit_code == 1
        assert "ANTHROPIC_API_KEY" in result.output
        assert StubOrchestrator.instances == []

    def test_auth_token_also_accepted(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = runner.invoke(
            cli.app, ["solve", "p"], env={"ANTHROPIC_AUTH_TOKEN": "oauth"}
        )
        assert result.exit_code == 0

    def test_dotenv_supplies_key_when_env_is_empty(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-dotenv\n")
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(cli.app, ["solve", "p"])
        assert result.exit_code == 0
        assert os.environ["ANTHROPIC_API_KEY"] == "from-dotenv"

    def test_real_env_var_beats_dotenv(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")
        (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-dotenv\n")
        monkeypatch.chdir(tmp_path)
        runner.invoke(cli.app, ["solve", "p"])
        assert os.environ["ANTHROPIC_API_KEY"] == "from-shell"

    def test_show_work_includes_reasoning_and_tools(self):
        result = runner.invoke(
            cli.app, ["solve", "p", "--show-work"], env=ENV
        )
        assert "Worked solution" in result.output
        assert "summed atomic masses" in result.output
        assert "lookup_molar_mass" in result.output

    def test_level_hint_forwarded(self):
        runner.invoke(cli.app, ["solve", "p", "--level", "university"], env=ENV)
        assert StubOrchestrator.instances[0].solve_calls == [("p", "university")]

    def test_solver_model_override_reaches_config(self):
        runner.invoke(
            cli.app, ["solve", "p", "--solver-model", "claude-sonnet-5"], env=ENV
        )
        cfg = StubOrchestrator.instances[0].kwargs["config"]
        assert cfg.solver_model == "claude-sonnet-5"
        assert cfg.planner_model == "claude-opus-5"

    def test_json_out_written(self, tmp_path):
        out = tmp_path / "outcome.json"
        result = runner.invoke(
            cli.app, ["solve", "p", "--json-out", str(out)], env=ENV
        )
        assert result.exit_code == 0
        data = json.loads(out.read_text())
        assert data["answer"]["formatted_answer"] == "18.02 g/mol"
        assert data["success"] is True

    def test_observer_renders_every_event_kind(self):
        runner.invoke(cli.app, ["solve", "p"], env=ENV)
        observer = StubOrchestrator.instances[0].kwargs["observer"]
        events = [
            SolveEvent(
                kind="plan_ready",
                payload={
                    "steps": ["a", "b"],
                    "target_quantity": "mass",
                    "expected_units": "g",
                    "difficulty": "alevel",
                    "effort": "high",
                },
            ),
            SolveEvent(kind="attempt_start", payload={"attempt": 1, "max_attempts": 3}),
            SolveEvent(kind="tool_call", payload={"name": "calc"}),
            SolveEvent(kind="answer_submitted", payload={"formatted_answer": "1 g"}),
            SolveEvent(kind="verdict", payload={"approved": True, "issues": []}),
            SolveEvent(kind="verdict", payload={"approved": False, "issues": ["bad"]}),
            SolveEvent(kind="error", payload={"stage": "solver", "message": "boom"}),
            SolveEvent(kind="done", payload={"success": True}),
        ]
        for event in events:
            observer(event)  # must not raise

    def test_observer_tolerates_empty_payloads(self):
        runner.invoke(cli.app, ["solve", "p"], env=ENV)
        observer = StubOrchestrator.instances[0].kwargs["observer"]
        for kind in ("plan_ready", "attempt_start", "tool_call", "verdict", "error"):
            observer(SolveEvent(kind=kind, payload={}))


class TestEval:
    def test_forwards_options_and_saves(self, monkeypatch, tmp_path):
        captured = {}

        def fake_run_benchmark(factory, **kwargs):
            captured["kwargs"] = kwargs
            captured["factory_result"] = factory()
            report = _fake_report()
            # exercise the progress callback with the harness's real signature
            kwargs["on_progress"](0, 1, report.results[0])
            return report

        saved = {}
        monkeypatch.setattr(
            "chemistry_agent.evals.harness.run_benchmark", fake_run_benchmark
        )
        monkeypatch.setattr(
            "chemistry_agent.evals.report.render", lambda r, console=None: None
        )
        monkeypatch.setattr(
            "chemistry_agent.evals.report.save_json",
            lambda r, p: saved.update(path=str(p), label=r.label),
        )

        out = tmp_path / "r.json"
        result = runner.invoke(
            cli.app,
            [
                "eval",
                "--limit",
                "5",
                "--level",
                "gcse",
                "--level",
                "alevel",
                "--label",
                "smoke",
                "--json-out",
                str(out),
            ],
            env=ENV,
        )
        assert result.exit_code == 0, result.output
        assert captured["kwargs"]["limit"] == 5
        assert captured["kwargs"]["levels"] == ["gcse", "alevel"]
        assert captured["kwargs"]["label"] == "smoke"
        assert hasattr(captured["factory_result"], "solve")
        assert saved["path"] == str(out)
        assert "1/1" in result.output  # progress callback rendered

    def test_bank_and_no_verifier_forwarded(self, monkeypatch, tmp_path):
        captured = {}

        def fake_run_benchmark(factory, **kwargs):
            captured.update(kwargs)
            return _fake_report()

        monkeypatch.setattr(
            "chemistry_agent.evals.harness.run_benchmark", fake_run_benchmark
        )
        monkeypatch.setattr(
            "chemistry_agent.evals.report.render", lambda r, console=None: None
        )
        monkeypatch.setattr(
            "chemistry_agent.evals.report.save_json", lambda r, p: None
        )
        result = runner.invoke(
            cli.app,
            ["eval", "--bank", "hard", "--no-verifier", "--json-out", str(tmp_path / "r.json")],
            env=ENV,
        )
        assert result.exit_code == 0, result.output
        assert captured["bank"] == "hard"
        assert captured["config"].use_verifier is False

    def test_verifier_on_and_standard_bank_by_default(self, monkeypatch, tmp_path):
        captured = {}
        monkeypatch.setattr(
            "chemistry_agent.evals.harness.run_benchmark",
            lambda factory, **kw: captured.update(kw) or _fake_report(),
        )
        monkeypatch.setattr(
            "chemistry_agent.evals.report.render", lambda r, console=None: None
        )
        monkeypatch.setattr(
            "chemistry_agent.evals.report.save_json", lambda r, p: None
        )
        result = runner.invoke(
            cli.app, ["eval", "--json-out", str(tmp_path / "r.json")], env=ENV
        )
        assert result.exit_code == 0, result.output
        assert captured["bank"] == "standard"
        assert captured["config"].use_verifier is True

    def test_default_json_path_uses_label(self, monkeypatch, tmp_path):
        saved = {}
        monkeypatch.setattr(
            "chemistry_agent.evals.harness.run_benchmark",
            lambda factory, **kw: _fake_report(),
        )
        monkeypatch.setattr(
            "chemistry_agent.evals.report.render", lambda r, console=None: None
        )
        monkeypatch.setattr(
            "chemistry_agent.evals.report.save_json",
            lambda r, p: saved.update(path=str(p)),
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(cli.app, ["eval", "--label", "baseline"], env=ENV)
        assert result.exit_code == 0
        assert saved["path"] == "eval_baseline.json"

    def test_requires_api_key(self, monkeypatch, tmp_path):
        # run outside the repo so a developer's local .env cannot supply a key
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        result = runner.invoke(cli.app, ["eval"])
        assert result.exit_code == 1
        assert "ANTHROPIC_API_KEY" in result.output


class TestCompare:
    def test_compare_loads_both_reports(self, monkeypatch, tmp_path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text("{}")
        b.write_text("{}")
        seen = {}
        monkeypatch.setattr(
            "chemistry_agent.evals.report.load_json", lambda p: f"report:{p.name}"
        )
        monkeypatch.setattr(
            "chemistry_agent.evals.report.compare",
            lambda x, y, console=None: seen.update(a=x, b=y),
        )
        result = runner.invoke(cli.app, ["compare", str(a), str(b)])
        assert result.exit_code == 0
        assert seen == {"a": "report:a.json", "b": "report:b.json"}

    def test_compare_needs_no_api_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text("{}")
        b.write_text("{}")
        monkeypatch.setattr("chemistry_agent.evals.report.load_json", lambda p: None)
        monkeypatch.setattr(
            "chemistry_agent.evals.report.compare", lambda x, y, console=None: None
        )
        result = runner.invoke(cli.app, ["compare", str(a), str(b)])
        assert result.exit_code == 0

    def test_missing_file_rejected(self, tmp_path):
        result = runner.invoke(
            cli.app, ["compare", str(tmp_path / "nope.json"), str(tmp_path / "x.json")]
        )
        assert result.exit_code != 0


class TestMeta:
    def test_version(self):
        result = runner.invoke(cli.app, ["--version"])
        assert result.exit_code == 0
        assert "chem" in result.output

    def test_help_lists_commands(self):
        result = runner.invoke(cli.app, ["--help"])
        assert result.exit_code == 0
        for command in ("solve", "eval", "compare"):
            assert command in result.output


def _fake_report():
    from chemistry_agent.evals.harness import EvalReport, ProblemResult

    return EvalReport(
        label="smoke",
        solver_model="claude-opus-5",
        n_problems=1,
        n_correct=1,
        accuracy=1.0,
        per_level={"gcse": {"correct": 1, "total": 1}},
        per_topic={"stoichiometry": {"correct": 1, "total": 1}},
        total_cost_usd=0.01,
        total_input_tokens=100,
        total_output_tokens=50,
        tool_histogram={"lookup_molar_mass": 1},
        mean_attempts=1.0,
        results=[
            ProblemResult(
                prompt="molar mass of H2O?",
                expected="18.02 g/mol",
                got="18.02 g/mol",
                correct=True,
                level="gcse",
                topic="stoichiometry",
                attempts=1,
                success=True,
                cost_usd=0.01,
                tool_calls=["lookup_molar_mass"],
            )
        ],
    )
