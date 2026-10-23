"""Eval harness: run an orchestrator over the problem bank and score it.

Scoring uses the deterministic comparator (``verification/comparator.py``),
never the verifier agent — the verifier judges without the expected answer,
while this harness holds the ground truth.

The harness takes an ``orchestrator_factory`` (a zero-arg callable returning
a fresh orchestrator-like object) rather than a single orchestrator instance:
agents accumulate ``usage`` across calls, so a fresh team per problem keeps
per-problem cost/token accounting honest.
"""

import math
import statistics
import time
from typing import Callable, Optional

from pydantic import BaseModel, Field

from ..config import AgentConfig
from ..helper.problems import examples
from ..problem_state.problem_state import ProblemState
from ..verification.comparator import answers_match


class ProblemResult(BaseModel):
    """Scored outcome of one problem in a benchmark run."""

    prompt: str
    expected: str
    got: Optional[str]
    got_value: Optional[float] = None
    got_units: Optional[str] = None
    got_insufficient_data: bool = False
    correct: bool
    level: str
    topic: str
    attempts: int
    success: bool
    cost_usd: float
    tool_calls: list[str]
    error: Optional[str] = None
    seconds: Optional[float] = Field(
        default=None, description="Wall-clock time for the whole solve"
    )

    @property
    def errored(self) -> bool:
        """No answer was produced because something failed, not a wrong answer."""
        return self.got is None and self.error is not None


class EvalReport(BaseModel):
    """Aggregate results of one benchmark run."""

    label: str
    solver_model: str
    use_verifier: bool = True
    bank: str = "standard"
    n_problems: int
    n_correct: int
    accuracy: float
    n_errors: int = Field(
        default=0,
        description="Problems with no answer because a call failed (network, API). "
        "Counted in n_problems but they say nothing about the agent: re-run them.",
    )
    per_level: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description='{"gcse": {"correct": 3, "total": 5}, ...}',
    )
    per_topic: dict[str, dict[str, int]] = Field(default_factory=dict)
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    tool_histogram: dict[str, int] = Field(default_factory=dict)
    mean_attempts: float
    latency_p50_s: Optional[float] = None
    latency_p95_s: Optional[float] = None
    results: list[ProblemResult] = Field(default_factory=list)


def _bump(breakdown: dict[str, dict[str, int]], key: str, correct: bool) -> None:
    bucket = breakdown.setdefault(key, {"correct": 0, "total": 0})
    bucket["total"] += 1
    if correct:
        bucket["correct"] += 1


# Expected value for problems that cannot be answered as written. Only an
# answer flagged ``insufficient_data`` is correct for them; a flagged answer
# on any other problem is wrong.
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

BANKS = ("standard", "hard", "all")


def load_bank(name: str = "standard") -> list[ProblemState]:
    """Return a problem bank by name: ``standard``, ``hard`` or ``all``."""
    if name not in BANKS:
        raise ValueError(f"unknown bank {name!r}; choose from {', '.join(BANKS)}")
    if name == "standard":
        return list(examples)
    from ..helper.problems_hard import hard_examples

    return list(hard_examples) if name == "hard" else [*examples, *hard_examples]


def _percentile(values: list[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return round(ordered[rank - 1], 2)


def _latency(results: list["ProblemResult"]) -> dict[str, Optional[float]]:
    secs = [r.seconds for r in results if r.seconds is not None and not r.errored]
    return {
        "latency_p50_s": round(statistics.median(secs), 2) if secs else None,
        "latency_p95_s": _percentile(secs, 95),
    }


def _score(answer, expected: str, tolerance: Optional[float]) -> bool:
    """Score a submitted answer against the expected one.

    Correct if either the human-readable ``formatted_answer`` or the solver's
    structured ``value`` + ``units`` matches. The structured fields come from
    the ``submit_answer`` tool schema, so they are immune to the prose the
    model sometimes adds to the formatted string ("≈0.17 M (0.025 mol / 0.15 L)").
    """
    if answer is None:
        return False
    flagged = bool(getattr(answer, "insufficient_data", False))
    if expected == INSUFFICIENT_DATA:
        return flagged
    if flagged:
        return False
    if answers_match(answer.formatted_answer, expected, rel_tol=tolerance):
        return True
    if answer.value is None:
        return False
    structured = f"{float(answer.value)!r} {answer.units or ''}".strip()
    return answers_match(structured, expected, rel_tol=tolerance)


def run_benchmark(
    orchestrator_factory: Callable[[], object],
    problems: Optional[list[ProblemState]] = None,
    label: str = "",
    limit: Optional[int] = None,
    levels: Optional[list[str]] = None,
    config: Optional[AgentConfig] = None,
    on_progress: Optional[Callable[[int, int, ProblemResult], None]] = None,
    bank: str = "standard",
) -> EvalReport:
    """Run every selected problem through a fresh orchestrator and score it.

    Args:
        orchestrator_factory: Zero-arg callable returning a fresh
            orchestrator-like object (``.solve(prompt, level_hint=None) ->
            SolveOutcome``) per problem.
        problems: Problem bank to draw from; defaults to
            ``chemistry_agent.helper.problems.examples``.
        label: Human-readable tag stored on the report (e.g. "opus-baseline").
        limit: Keep at most this many problems (applied after level filtering).
        levels: Keep only problems whose ``level`` is in this list.
        config: ``AgentConfig`` the report's ``solver_model`` is read from.
        on_progress: Optional ``(index, total, result)`` callback invoked after
            each problem; exceptions it raises are ignored.

    Returns:
        An ``EvalReport`` with overall/per-level/per-topic accuracy, cost and
        token totals, a tool-call histogram, and every ``ProblemResult``.

    One problem's exception never kills the run: it is recorded as an
    incorrect ``ProblemResult`` with ``error`` set, and the run continues.
    """
    # deferred import: helper.tools pulls in the full GCSE toolkit
    from ..helper.tools import build_prompt

    cfg = config or AgentConfig()

    pool = list(problems) if problems is not None else load_bank(bank)
    if levels is not None:
        wanted = set(levels)
        pool = [p for p in pool if p.level in wanted]
    if limit is not None:
        pool = pool[:limit]

    results: list[ProblemResult] = []
    per_level: dict[str, dict[str, int]] = {}
    per_topic: dict[str, dict[str, int]] = {}
    tool_histogram: dict[str, int] = {}
    total_cost = 0.0
    total_in = 0
    total_out = 0

    total = len(pool)
    for index, problem in enumerate(pool):
        prompt = problem.prompt or (problem.task or "")
        try:
            if not problem.prompt:
                prompt = build_prompt(problem)
            started = time.perf_counter()
            outcome = orchestrator_factory().solve(prompt, level_hint=problem.level)
            seconds = round(time.perf_counter() - started, 2)

            got = outcome.answer.formatted_answer if outcome.answer else None
            correct = _score(outcome.answer, problem.expected, problem.tolerance)
            result = ProblemResult(
                prompt=prompt,
                expected=problem.expected,
                got=got,
                got_value=outcome.answer.value if outcome.answer else None,
                got_units=outcome.answer.units if outcome.answer else None,
                got_insufficient_data=bool(
                    outcome.answer and outcome.answer.insufficient_data
                ),
                correct=correct,
                level=problem.level,
                topic=problem.topic,
                attempts=outcome.attempts,
                success=outcome.success,
                cost_usd=outcome.cost_usd,
                tool_calls=list(outcome.tool_calls),
                error=outcome.error,
                seconds=seconds,
            )
            total_cost += outcome.cost_usd
            for usage in outcome.usage.values():
                total_in += (
                    usage.input_tokens
                    + usage.cache_read_input_tokens
                    + usage.cache_creation_input_tokens
                )
                total_out += usage.output_tokens
        except Exception as exc:  # one bad problem must not kill the run
            result = ProblemResult(
                prompt=prompt,
                expected=problem.expected,
                got=None,
                correct=False,
                level=problem.level,
                topic=problem.topic,
                attempts=0,
                success=False,
                cost_usd=0.0,
                tool_calls=[],
                error=f"{type(exc).__name__}: {exc}",
            )

        results.append(result)
        _bump(per_level, result.level, result.correct)
        _bump(per_topic, result.topic, result.correct)
        for name in result.tool_calls:
            tool_histogram[name] = tool_histogram.get(name, 0) + 1

        if on_progress is not None:
            try:
                on_progress(index, total, result)
            except Exception:
                pass  # a broken progress callback must never kill the run

    n_correct = sum(1 for r in results if r.correct)
    return EvalReport(
        label=label,
        solver_model=cfg.solver_model,
        use_verifier=cfg.use_verifier,
        bank=bank if problems is None else "custom",
        n_problems=total,
        n_correct=n_correct,
        accuracy=(n_correct / total) if total else 0.0,
        n_errors=sum(1 for r in results if r.errored),
        **_latency(results),
        per_level=per_level,
        per_topic=per_topic,
        total_cost_usd=round(total_cost, 6),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        tool_histogram=tool_histogram,
        mean_attempts=(sum(r.attempts for r in results) / total) if total else 0.0,
        results=results,
    )


def rescore_report(
    report: EvalReport, problems: Optional[list[ProblemState]] = None
) -> tuple[EvalReport, list[str]]:
    """Re-grade a saved report against the current answer key, without API calls.

    Results are matched to problems by prompt text. Matched results take the
    problem's current ``expected`` and are scored with the current comparator,
    using the structured ``got_value``/``got_units`` when the report has them.
    Unmatched results (the question was reworded since the run) keep their
    original grade. Cost, token and tool fields are carried over unchanged.

    Returns:
        The re-scored report and the prompts that could not be matched.
    """
    from types import SimpleNamespace

    from ..helper.tools import build_prompt

    if problems is not None:
        bank = list(problems)
    else:
        try:
            bank = load_bank("all")
        except ImportError:  # hard bank not written yet
            bank = load_bank("standard")
    by_prompt = {(p.prompt or build_prompt(p)): p for p in bank}

    results: list[ProblemResult] = []
    unmatched: list[str] = []
    per_level: dict[str, dict[str, int]] = {}
    per_topic: dict[str, dict[str, int]] = {}
    for old in report.results:
        problem = by_prompt.get(old.prompt)
        if problem is None:
            unmatched.append(old.prompt)
            new = old
        else:
            answer = (
                SimpleNamespace(
                    formatted_answer=old.got,
                    value=old.got_value,
                    units=old.got_units,
                    insufficient_data=old.got_insufficient_data,
                )
                if old.got is not None
                else None
            )
            new = old.model_copy(
                update={
                    "expected": problem.expected,
                    "correct": _score(answer, problem.expected, problem.tolerance),
                }
            )
        results.append(new)
        _bump(per_level, new.level, new.correct)
        _bump(per_topic, new.topic, new.correct)

    n_correct = sum(1 for r in results if r.correct)
    total = len(results)
    rescored = report.model_copy(
        update={
            "results": results,
            "n_correct": n_correct,
            "accuracy": (n_correct / total) if total else 0.0,
            "n_errors": sum(1 for r in results if r.errored),
            **_latency(results),
            "per_level": per_level,
            "per_topic": per_topic,
        }
    )
    return rescored, unmatched
