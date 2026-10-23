"""Eval harness and report tests — stub orchestrators only, no SDK or network."""

import pytest
from rich.console import Console

from chemistry_agent.config import AgentConfig
from chemistry_agent.evals.harness import (
    EvalReport,
    ProblemResult,
    INSUFFICIENT_DATA,
    load_bank,
    rescore_report,
    run_benchmark,
)
from chemistry_agent.evals.report import compare, load_json, render, save_json
from chemistry_agent.models import FinalAnswer, Plan, PlanStep, SolveOutcome, UsageTotals
from chemistry_agent.problem_state.problem_state import ProblemState


def outcome_for(
    answer_text, *, attempts=1, success=True, cost=0.01, tools=None, value=None, units=None
):
    return SolveOutcome(
        prompt="p",
        plan=Plan(
            steps=[PlanStep(description="s")],
            target_quantity="q",
            expected_units="g",
            difficulty="gcse",
            topic="general",
        ),
        answer=(
            FinalAnswer(
                value=value,
                units=units,
                formatted_answer=answer_text,
                reasoning_summary="r",
            )
            if answer_text is not None
            else None
        ),
        attempts=attempts,
        success=success,
        cost_usd=cost,
        tool_calls=list(tools or ["molar_mass"]),
        usage={
            "planner": UsageTotals(input_tokens=10, output_tokens=5),
            "solver": UsageTotals(input_tokens=100, output_tokens=50),
        },
    )


class StubOrchestrator:
    """Returns a scripted answer per prompt; optionally raises."""

    def __init__(self, answers: dict, raise_on: str | None = None):
        self.answers = answers
        self.raise_on = raise_on
        self.seen = []

    def solve(self, prompt, level_hint=None):
        self.seen.append((prompt, level_hint))
        if self.raise_on is not None and self.raise_on in prompt:
            raise RuntimeError("orchestrator exploded")
        return outcome_for(self.answers.get(prompt, "0 g"))


def bank():
    return [
        ProblemState(
            prompt="molar mass of H2O?",
            params={},
            expected="18.02 g/mol",
            level="gcse",
            topic="stoichiometry",
        ),
        ProblemState(
            prompt="buffer pH?",
            params={},
            expected="pH = 4.74",
            level="alevel",
            topic="acids_bases",
        ),
        ProblemState(
            prompt="cell potential?",
            params={},
            expected="E = 1.130 V",
            level="university",
            topic="electrochemistry",
        ),
    ]


def factory_for(stub):
    return lambda: stub


class TestScoring:
    def test_accuracy_and_tolerance(self):
        # "18.016 g/mol" rounds to the key "18.02 g/mol" -> correct via comparator
        stub = StubOrchestrator(
            {
                "molar mass of H2O?": "18.016 g/mol",
                "buffer pH?": "pH = 4.74",
                "cell potential?": "E = 9.999 V",
            }
        )
        report = run_benchmark(factory_for(stub), problems=bank(), label="t")
        assert report.n_problems == 3
        assert report.n_correct == 2
        assert report.accuracy == pytest.approx(2 / 3)
        assert [r.correct for r in report.results] == [True, True, False]

    def test_missing_answer_is_incorrect(self):
        class NoAnswer:
            def solve(self, prompt, level_hint=None):
                return outcome_for(None, success=False)

        report = run_benchmark(lambda: NoAnswer(), problems=bank())
        assert report.n_correct == 0
        assert all(r.got is None for r in report.results)

    def test_level_hint_passed_through(self):
        stub = StubOrchestrator({})
        run_benchmark(factory_for(stub), problems=bank())
        assert [level for _, level in stub.seen] == ["gcse", "alevel", "university"]


class TestBreakdowns:
    def test_per_level_and_per_topic(self):
        stub = StubOrchestrator(
            {"molar mass of H2O?": "18.02 g/mol", "buffer pH?": "pH = 4.74"}
        )
        report = run_benchmark(factory_for(stub), problems=bank())
        assert report.per_level["gcse"] == {"correct": 1, "total": 1}
        assert report.per_level["university"] == {"correct": 0, "total": 1}
        assert report.per_topic["acids_bases"] == {"correct": 1, "total": 1}

    def test_tool_histogram_and_mean_attempts(self):
        class Multi:
            def solve(self, prompt, level_hint=None):
                return outcome_for("18.02 g/mol", attempts=2, tools=["calc", "calc"])

        report = run_benchmark(lambda: Multi(), problems=bank())
        assert report.tool_histogram == {"calc": 6}
        assert report.mean_attempts == pytest.approx(2.0)

    def test_token_and_cost_totals(self):
        report = run_benchmark(lambda: StubOrchestrator({}), problems=bank())
        assert report.total_input_tokens == 3 * 110
        assert report.total_output_tokens == 3 * 55
        assert report.total_cost_usd == pytest.approx(0.03)


class TestFiltering:
    def test_limit(self):
        report = run_benchmark(
            lambda: StubOrchestrator({}), problems=bank(), limit=2
        )
        assert report.n_problems == 2

    def test_levels_filter(self):
        report = run_benchmark(
            lambda: StubOrchestrator({}), problems=bank(), levels=["university"]
        )
        assert report.n_problems == 1
        assert report.results[0].level == "university"

    def test_levels_then_limit(self):
        report = run_benchmark(
            lambda: StubOrchestrator({}),
            problems=bank(),
            levels=["gcse", "alevel"],
            limit=1,
        )
        assert report.n_problems == 1
        assert report.results[0].level == "gcse"

    def test_default_bank_is_the_real_problem_set(self):
        report = run_benchmark(lambda: StubOrchestrator({}), limit=4)
        assert report.n_problems == 4


class TestRobustness:
    def test_exception_isolated_and_run_continues(self):
        stub = StubOrchestrator(
            {"buffer pH?": "pH = 4.74"}, raise_on="molar mass"
        )
        report = run_benchmark(factory_for(stub), problems=bank())
        assert report.n_problems == 3
        assert report.results[0].correct is False
        assert "orchestrator exploded" in report.results[0].error
        assert report.results[1].correct is True

    def test_broken_progress_callback_ignored(self):
        def bad(index, total, result):
            raise ValueError("ui exploded")

        report = run_benchmark(
            lambda: StubOrchestrator({}), problems=bank(), on_progress=bad
        )
        assert report.n_problems == 3

    def test_progress_callback_receives_results(self):
        seen = []
        run_benchmark(
            lambda: StubOrchestrator({}),
            problems=bank(),
            on_progress=lambda i, t, r: seen.append((i, t, r.level)),
        )
        assert seen == [(0, 3, "gcse"), (1, 3, "alevel"), (2, 3, "university")]

    def test_empty_bank(self):
        report = run_benchmark(lambda: StubOrchestrator({}), problems=[])
        assert report.n_problems == 0
        assert report.accuracy == 0.0
        assert report.mean_attempts == 0.0

    def test_solver_model_recorded(self):
        cfg = AgentConfig(solver_model="claude-sonnet-5")
        report = run_benchmark(
            lambda: StubOrchestrator({}), problems=bank(), config=cfg
        )
        assert report.solver_model == "claude-sonnet-5"


class TestReportIO:
    def test_json_round_trip(self, tmp_path):
        report = run_benchmark(lambda: StubOrchestrator({}), problems=bank(), label="x")
        path = tmp_path / "report.json"
        save_json(report, path)
        loaded = load_json(path)
        assert isinstance(loaded, EvalReport)
        assert loaded.label == "x"
        assert loaded.n_problems == report.n_problems
        assert [r.prompt for r in loaded.results] == [r.prompt for r in report.results]

    def test_render_smoke(self):
        report = run_benchmark(
            lambda: StubOrchestrator({"buffer pH?": "pH = 4.74"}), problems=bank()
        )
        console = Console(record=True, width=100)
        render(report, console=console)
        text = console.export_text()
        assert "alevel" in text

    def test_compare_smoke(self):
        a = run_benchmark(lambda: StubOrchestrator({}), problems=bank(), label="a")
        b = run_benchmark(
            lambda: StubOrchestrator({"buffer pH?": "pH = 4.74"}),
            problems=bank(),
            label="b",
        )
        console = Console(record=True, width=120)
        compare(a, b, console=console)
        text = console.export_text()
        assert "a" in text and "b" in text

    def test_problem_result_shape(self):
        report = run_benchmark(lambda: StubOrchestrator({}), problems=bank(), limit=1)
        r = report.results[0]
        assert isinstance(r, ProblemResult)
        assert r.expected == "18.02 g/mol"
        assert r.level == "gcse"


class StructuredStub:
    """Returns one fixed answer with structured value/units for every prompt."""

    def __init__(self, text, value, units):
        self.text, self.value, self.units = text, value, units

    def solve(self, prompt, level_hint=None):
        return outcome_for(self.text, value=self.value, units=self.units)


def one_problem(expected="0.17 M"):
    return [
        ProblemState(
            prompt="mix two solutions?",
            params={},
            expected=expected,
            level="gcse",
            topic="solutions",
        )
    ]


class TestStructuredScoring:
    def test_structured_value_rescues_unparseable_text(self):
        stub = StructuredStub("about point one seven molar", 0.17, "M")
        report = run_benchmark(lambda: stub, problems=one_problem())
        assert report.n_correct == 1

    def test_wrong_structured_value_still_fails(self):
        stub = StructuredStub("about point two molar", 0.20, "M")
        report = run_benchmark(lambda: stub, problems=one_problem())
        assert report.n_correct == 0

    def test_unitless_structured_value(self):
        stub = StructuredStub("the pH is roughly 4.74", 4.74, "")
        report = run_benchmark(lambda: stub, problems=one_problem("pH = 4.74"))
        assert report.n_correct == 1

    def test_report_keeps_the_formatted_text(self):
        stub = StructuredStub("about point one seven molar", 0.17, "M")
        report = run_benchmark(lambda: stub, problems=one_problem())
        assert report.results[0].got == "about point one seven molar"


class TestRescore:
    def test_corrected_key_flips_the_grade(self):
        stub = StructuredStub("0.1667 M", 0.1667, "M")
        report = run_benchmark(lambda: stub, problems=one_problem("0.1 M"))
        assert report.n_correct == 0

        rescored, unmatched = rescore_report(report, problems=one_problem("0.17 M"))
        assert unmatched == []
        assert rescored.n_correct == 1
        assert rescored.accuracy == 1.0
        assert rescored.results[0].expected == "0.17 M"
        assert rescored.per_topic == {"solutions": {"correct": 1, "total": 1}}

    def test_reworded_question_keeps_original_grade(self):
        stub = StructuredStub("0.1667 M", 0.1667, "M")
        report = run_benchmark(lambda: stub, problems=one_problem("0.1 M"))
        reworded = one_problem("0.17 M")
        reworded[0].prompt = "a differently worded question?"

        rescored, unmatched = rescore_report(report, problems=reworded)
        assert unmatched == ["mix two solutions?"]
        assert rescored.n_correct == 0

    def test_cost_and_tokens_carried_over(self):
        stub = StructuredStub("0.17 M", 0.17, "M")
        report = run_benchmark(lambda: stub, problems=one_problem())
        rescored, _ = rescore_report(report, problems=one_problem())
        assert rescored.total_cost_usd == report.total_cost_usd
        assert rescored.total_input_tokens == report.total_input_tokens


class FlakyStub:
    """Fails like a dropped connection on the second problem."""

    def __init__(self):
        self.calls = 0

    def solve(self, prompt, level_hint=None):
        self.calls += 1
        if self.calls == 2:
            outcome = outcome_for(None, success=False, attempts=0, cost=0.0)
            return outcome.model_copy(update={"error": "planner: Connection error."})
        return outcome_for("18.02 g/mol")


class TestErrorsAreNotWrongAnswers:
    def test_connection_failure_counted_as_error(self):
        stub = FlakyStub()
        report = run_benchmark(lambda: stub, problems=bank()[:1] * 3)
        assert report.n_problems == 3
        assert report.n_errors == 1
        assert report.n_correct == 2
        assert [r.errored for r in report.results] == [False, True, False]

    def test_wrong_answer_is_not_an_error(self):
        stub = StubOrchestrator({"molar mass of H2O?": "99 g/mol"})
        report = run_benchmark(lambda: stub, problems=bank()[:1])
        assert report.n_errors == 0
        assert report.n_correct == 0

    def test_old_reports_without_the_field_still_load(self, tmp_path):
        stub = StubOrchestrator({})
        report = run_benchmark(lambda: stub, problems=bank()[:1])
        data = report.model_dump()
        data.pop("n_errors")
        assert EvalReport.model_validate(data).n_errors == 0


class FlaggedStub:
    """Submits an answer flagged insufficient_data for every prompt."""

    def solve(self, prompt, level_hint=None):
        outcome = outcome_for("Cannot be answered: the volume is not given.", value=0, units="")
        outcome.answer.insufficient_data = True
        return outcome


class TestInsufficientData:
    def test_flag_is_correct_for_unanswerable_problem(self):
        report = run_benchmark(lambda: FlaggedStub(), problems=one_problem(INSUFFICIENT_DATA))
        assert report.n_correct == 1
        assert report.results[0].got_insufficient_data is True

    def test_flag_is_wrong_for_answerable_problem(self):
        report = run_benchmark(lambda: FlaggedStub(), problems=one_problem("0.17 M"))
        assert report.n_correct == 0

    def test_number_is_wrong_for_unanswerable_problem(self):
        stub = StructuredStub("0.17 M", 0.17, "M")
        report = run_benchmark(lambda: stub, problems=one_problem(INSUFFICIENT_DATA))
        assert report.n_correct == 0

    def test_rescore_keeps_the_flag(self):
        report = run_benchmark(lambda: FlaggedStub(), problems=one_problem(INSUFFICIENT_DATA))
        rescored, _ = rescore_report(report, problems=one_problem(INSUFFICIENT_DATA))
        assert rescored.n_correct == 1


class TestLatencyAndMetadata:
    def test_seconds_and_percentiles_recorded(self):
        stub = StructuredStub("0.17 M", 0.17, "M")
        report = run_benchmark(lambda: stub, problems=one_problem() * 4)
        assert all(r.seconds is not None and r.seconds >= 0 for r in report.results)
        assert report.latency_p50_s is not None
        assert report.latency_p95_s is not None
        assert report.latency_p95_s >= report.latency_p50_s

    def test_errored_problems_excluded_from_latency(self):
        stub = FlakyStub()
        report = run_benchmark(lambda: stub, problems=bank()[:1] * 2)
        assert report.n_errors == 1
        assert report.latency_p50_s is not None

    def test_report_records_verifier_and_bank(self):
        from chemistry_agent.config import AgentConfig

        stub = StructuredStub("0.17 M", 0.17, "M")
        report = run_benchmark(
            lambda: stub, problems=one_problem(), config=AgentConfig(use_verifier=False)
        )
        assert report.use_verifier is False
        assert report.bank == "custom"


class TestBanks:
    def test_standard_bank(self):
        assert len(load_bank("standard")) == 30

    def test_unknown_bank_rejected(self):
        with pytest.raises(ValueError):
            load_bank("nope")

