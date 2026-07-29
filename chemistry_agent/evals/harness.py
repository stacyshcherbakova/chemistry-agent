"""Eval harness: run an orchestrator over the problem bank and score it.

Scoring uses the deterministic comparator (``verification/comparator.py``),
never the verifier agent — the verifier judges without the expected answer,
while this harness holds the ground truth.

The harness takes an ``orchestrator_factory`` (a zero-arg callable returning
a fresh orchestrator-like object) rather than a single orchestrator instance:
agents accumulate ``usage`` across calls, so a fresh team per problem keeps
per-problem cost/token accounting honest.
"""

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
    correct: bool
    level: str
    topic: str
    attempts: int
    success: bool
    cost_usd: float
    tool_calls: list[str]
    error: Optional[str] = None


class EvalReport(BaseModel):
    """Aggregate results of one benchmark run."""

    label: str
    solver_model: str
    n_problems: int
    n_correct: int
    accuracy: float
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
    results: list[ProblemResult] = Field(default_factory=list)


def _bump(breakdown: dict[str, dict[str, int]], key: str, correct: bool) -> None:
    bucket = breakdown.setdefault(key, {"correct": 0, "total": 0})
    bucket["total"] += 1
    if correct:
        bucket["correct"] += 1


def run_benchmark(
    orchestrator_factory: Callable[[], object],
    problems: Optional[list[ProblemState]] = None,
    label: str = "",
    limit: Optional[int] = None,
    levels: Optional[list[str]] = None,
    config: Optional[AgentConfig] = None,
    on_progress: Optional[Callable[[int, int, ProblemResult], None]] = None,
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

    pool = list(problems) if problems is not None else list(examples)
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
            outcome = orchestrator_factory().solve(prompt, level_hint=problem.level)

            got = outcome.answer.formatted_answer if outcome.answer else None
            correct = (
                answers_match(got, problem.expected, rel_tol=problem.tolerance)
                if got is not None
                else False
            )
            result = ProblemResult(
                prompt=prompt,
                expected=problem.expected,
                got=got,
                correct=correct,
                level=problem.level,
                topic=problem.topic,
                attempts=outcome.attempts,
                success=outcome.success,
                cost_usd=outcome.cost_usd,
                tool_calls=list(outcome.tool_calls),
                error=outcome.error,
            )
            total_cost += outcome.cost_usd
            for usage in outcome.usage.values():
                total_in += usage.input_tokens
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
        n_problems=total,
        n_correct=n_correct,
        accuracy=(n_correct / total) if total else 0.0,
        per_level=per_level,
        per_topic=per_topic,
        total_cost_usd=round(total_cost, 6),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        tool_histogram=tool_histogram,
        mean_attempts=(sum(r.attempts for r in results) / total) if total else 0.0,
        results=results,
    )
