"""Command-line interface for the chemistry agent team.

Commands:
    chem solve PROMPT   -- live-rendered multi-agent solve of one problem
    chem eval           -- run the benchmark problem bank and save a report
    chem compare A B    -- diff two saved eval reports

The eval/report modules are imported lazily inside their commands so that
``chem solve`` keeps working even while ``chemistry_agent.evals`` is under
construction, and so tests can monkeypatch them via ``sys.modules``.
"""

import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agents.orchestrator import Orchestrator
from .config import AgentConfig
from .models import SolveEvent, SolveOutcome

app = typer.Typer(
    name="chem",
    help="Multi-agent chemistry problem solver (GCSE through intro-university).",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


class LevelChoice(str, Enum):
    gcse = "gcse"
    alevel = "alevel"
    university = "university"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _require_api_key() -> None:
    """Exit early with a clear message if no Anthropic credentials are set."""
    if "ANTHROPIC_API_KEY" not in os.environ and "ANTHROPIC_AUTH_TOKEN" not in os.environ:
        console.print(
            "[bold red]Set ANTHROPIC_API_KEY to run the solver.[/bold red]"
        )
        raise typer.Exit(1)


def _build_config(solver_model: Optional[str]) -> AgentConfig:
    cfg = AgentConfig()
    if solver_model:
        cfg = AgentConfig(solver_model=solver_model)
    return cfg


def _make_observer(out: Console):
    """Build an Observer that renders SolveEvents live with rich."""

    def observer(event: SolveEvent) -> None:
        p = event.payload
        if event.kind == "plan_ready":
            lines = [
                f"{i}. {desc}" for i, desc in enumerate(p.get("steps", []), start=1)
            ]
            target = p.get("target_quantity") or "?"
            units = p.get("expected_units")
            lines.append("")
            lines.append(
                f"target: {target}" + (f"  [dim]({units})[/dim]" if units else "")
            )
            badge = f"{p.get('difficulty', '?')} · effort: {p.get('effort', '?')}"
            out.print(
                Panel(
                    "\n".join(lines),
                    title="Plan",
                    subtitle=badge,
                    border_style="cyan",
                )
            )
        elif event.kind == "attempt_start":
            out.rule(f"Attempt {p.get('attempt', '?')}/{p.get('max_attempts', '?')}")
        elif event.kind == "tool_call":
            out.print(f"  ⚙ {p.get('name', '?')}", style="dim", markup=False)
        elif event.kind == "answer_submitted":
            out.print(
                f"  → answer: {p.get('formatted_answer', '')}", markup=False
            )
        elif event.kind == "verdict":
            if p.get("approved"):
                out.print("✓ verifier approved", style="green", markup=False)
            else:
                issues = "; ".join(p.get("issues", [])) or "no details"
                out.print(f"✗ rejected: {issues}", style="yellow", markup=False)
        elif event.kind == "error":
            out.print(
                f"error ({p.get('stage', '?')}): {p.get('message', '')}",
                style="red",
                markup=False,
            )
        # "done": nothing — the footer is rendered after solve() returns.

    return observer


def _result_panel(outcome: SolveOutcome) -> Panel:
    if outcome.success and outcome.answer is not None:
        return Panel(
            outcome.answer.formatted_answer,
            title="Answer",
            border_style="green",
        )
    lines = []
    if outcome.answer is not None:
        lines.append(f"last answer: {outcome.answer.formatted_answer}")
    if outcome.error:
        lines.append(f"error: {outcome.error}")
    if not lines:
        lines.append("no answer produced")
    return Panel("\n".join(lines), title="Failed", border_style="red")


def _worked_solution_panel(outcome: SolveOutcome) -> Panel:
    lines: list[str] = []
    if outcome.plan is not None:
        lines.append("Plan:")
        for i, step in enumerate(outcome.plan.steps, start=1):
            lines.append(f"  {i}. {step.description}")
    if outcome.tool_calls:
        lines.append("")
        lines.append("Tool calls (in order):")
        for i, name in enumerate(outcome.tool_calls, start=1):
            lines.append(f"  {i}. {name}")
    if outcome.answer is not None:
        lines.append("")
        lines.append(f"Reasoning: {outcome.answer.reasoning_summary}")
        lines.append(f"Answer: {outcome.answer.formatted_answer}")
    return Panel("\n".join(lines) or "nothing recorded", title="Worked solution")


def _print_footer(outcome: SolveOutcome) -> None:
    table = Table(box=box.SIMPLE)
    table.add_column("role")
    table.add_column("in tokens", justify="right")
    table.add_column("out tokens", justify="right")
    for role, usage in outcome.usage.items():
        table.add_row(role, str(usage.input_tokens), str(usage.output_tokens))
    console.print(table)
    console.print(
        f"attempts: {outcome.attempts}   total cost: ${outcome.cost_usd:.4f}",
        markup=False,
    )


# --------------------------------------------------------------------------- #
# chem --version
# --------------------------------------------------------------------------- #


def _package_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("chemistry-agent")
    except Exception:
        return "0.2.0"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"chem {_package_version()}", markup=False)
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the package version and exit.",
    ),
) -> None:
    """Multi-agent chemistry problem solver."""


# --------------------------------------------------------------------------- #
# chem solve
# --------------------------------------------------------------------------- #


@app.command()
def solve(
    prompt: str = typer.Argument(..., help="The chemistry problem to solve."),
    level: Optional[LevelChoice] = typer.Option(
        None, "--level", help="Hint the difficulty tier (gcse/alevel/university)."
    ),
    show_work: bool = typer.Option(
        False, "--show-work", help="Render the full worked solution afterwards."
    ),
    solver_model: Optional[str] = typer.Option(
        None, "--solver-model", help="Override the solver model."
    ),
    json_out: Optional[Path] = typer.Option(
        None, "--json-out", help="Write the SolveOutcome as JSON to this path."
    ),
) -> None:
    """Solve one chemistry problem with live progress rendering."""
    _require_api_key()

    cfg = _build_config(solver_model)
    obs = _make_observer(console)
    orchestrator = Orchestrator(config=cfg, observer=obs)
    outcome = orchestrator.solve(
        prompt, level_hint=level.value if level is not None else None
    )

    console.print(_result_panel(outcome))
    if show_work:
        console.print(_worked_solution_panel(outcome))
    _print_footer(outcome)

    if json_out is not None:
        Path(json_out).write_text(outcome.model_dump_json(indent=2))
        console.print(f"wrote {json_out}", style="dim", markup=False)

    raise typer.Exit(0 if outcome.success else 1)


# --------------------------------------------------------------------------- #
# chem eval
# --------------------------------------------------------------------------- #


@app.command("eval")
def eval_cmd(
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Cap the number of benchmark problems."
    ),
    level: List[LevelChoice] = typer.Option(
        [], "--level", help="Restrict to these levels (repeatable)."
    ),
    label: str = typer.Option("", "--label", help="Label recorded in the report."),
    solver_model: Optional[str] = typer.Option(
        None, "--solver-model", help="Override the solver model."
    ),
    json_out: Optional[Path] = typer.Option(
        None,
        "--json-out",
        help="Report JSON path (default: eval_{label or timestamp}.json).",
    ),
) -> None:
    """Run the benchmark problem bank and save an eval report."""
    _require_api_key()

    # Lazy imports: the evals modules may still be under construction, and
    # `chem solve` must not depend on them.
    from .evals.harness import run_benchmark
    from .evals.report import render, save_json

    cfg = _build_config(solver_model)
    factory = lambda: Orchestrator(config=cfg)  # noqa: E731

    def on_progress(index: int, total: int, result) -> None:
        mark = "✓" if result.correct else "✗"
        prompt = result.prompt
        prefix = prompt if len(prompt) <= 60 else prompt[:57] + "..."
        console.print(f"  [{index + 1}/{total}] {mark} {prefix}", markup=False)

    levels = [lv.value for lv in level] or None
    report = run_benchmark(
        factory,
        label=label,
        limit=limit,
        levels=levels,
        config=cfg,
        on_progress=on_progress,
    )

    render(report, console=console)

    if json_out is None:
        stamp = label or datetime.now().strftime("%Y%m%d-%H%M%S")
        json_out = Path(f"eval_{stamp}.json")
    save_json(report, json_out)
    console.print(f"saved report to {json_out}", markup=False)


# --------------------------------------------------------------------------- #
# chem compare
# --------------------------------------------------------------------------- #


@app.command()
def compare(
    a: Path = typer.Argument(..., exists=True, help="First eval report JSON."),
    b: Path = typer.Argument(..., exists=True, help="Second eval report JSON."),
) -> None:
    """Compare two saved eval reports side by side."""
    from .evals.report import compare as compare_reports
    from .evals.report import load_json

    compare_reports(load_json(a), load_json(b), console=console)


if __name__ == "__main__":  # pragma: no cover
    app()
