"""Rendering and persistence for eval reports: rich tables, JSON, comparison."""

from pathlib import Path
from typing import Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .harness import EvalReport


def _accuracy_cell(bucket: dict[str, int]) -> str:
    total = bucket.get("total", 0)
    correct = bucket.get("correct", 0)
    pct = (correct / total * 100) if total else 0.0
    return f"{correct}/{total} ({pct:.0f}%)"


def _seconds(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f} s"


def _truncate(text: str, width: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _breakdown_table(title: str, key_header: str, breakdown: dict) -> Table:
    table = Table(title=title)
    table.add_column(key_header)
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Accuracy", justify="right")
    for key, bucket in breakdown.items():
        total = bucket.get("total", 0)
        correct = bucket.get("correct", 0)
        pct = (correct / total * 100) if total else 0.0
        table.add_row(key, str(correct), str(total), f"{pct:.0f}%")
    return table


def render(report: EvalReport, console: Optional[Console] = None) -> None:
    """Print a full eval report as rich panels/tables."""
    console = console or Console()

    summary = (
        f"[bold]{report.label or 'eval'}[/bold]\n"
        f"solver model:  {report.solver_model}\n"
        f"accuracy:      {report.n_correct}/{report.n_problems} "
        f"({report.accuracy * 100:.1f}%)\n"
        f"total cost:    ${report.total_cost_usd:.4f}\n"
        f"tokens:        {report.total_input_tokens:,} in / "
        f"{report.total_output_tokens:,} out\n"
        f"mean attempts: {report.mean_attempts:.2f}\n"
        f"verifier:      {'on' if report.use_verifier else 'off'}\n"
        f"bank:          {report.bank}\n"
        f"latency:       {_seconds(report.latency_p50_s)} median, "
        f"{_seconds(report.latency_p95_s)} p95"
    )
    console.print(Panel(summary, title="Eval summary", expand=False))
    if report.n_errors:
        console.print(
            f"[bold yellow]{report.n_errors} problem(s) produced no answer because a "
            "call failed (network or API).[/bold yellow] They count as incorrect "
            "above but say nothing about the agent: re-run them."
        )

    console.print(_breakdown_table("Accuracy by level", "Level", report.per_level))
    console.print(_breakdown_table("Accuracy by topic", "Topic", report.per_topic))

    histogram = Table(title="Tool calls (top 15)")
    histogram.add_column("Tool")
    histogram.add_column("Calls", justify="right")
    top = sorted(report.tool_histogram.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    for name, count in top:
        histogram.add_row(name, str(count))
    console.print(histogram)

    failures = [r for r in report.results if not r.correct]
    if failures:
        table = Table(title=f"Failures ({len(failures)})")
        table.add_column("Prompt", max_width=62)
        table.add_column("Expected")
        table.add_column("Got")
        table.add_column("Error")
        for r in failures:
            table.add_row(
                _truncate(r.prompt, 60),
                r.expected,
                r.got if r.got is not None else "—",
                r.error or "",
            )
        console.print(table)
    else:
        console.print("[green]No failures.[/green]")


def save_json(report: EvalReport, path: Union[str, Path]) -> None:
    """Write a report to ``path`` as pretty-printed JSON."""
    Path(path).write_text(report.model_dump_json(indent=2), encoding="utf-8")


def load_json(path: Union[str, Path]) -> EvalReport:
    """Load a report previously written by :func:`save_json`."""
    return EvalReport.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def compare(a: EvalReport, b: EvalReport, console: Optional[Console] = None) -> None:
    """Print a side-by-side comparison of two runs with a delta column."""
    console = console or Console()

    name_a = a.label or "run A"
    name_b = b.label or "run B"

    table = Table(title=f"{name_a} vs {name_b}")
    table.add_column("Metric")
    table.add_column(name_a, justify="right")
    table.add_column(name_b, justify="right")
    table.add_column("Δ (B−A)", justify="right")

    def row(metric: str, va: float, vb: float, fmt: str, delta_fmt: str) -> None:
        table.add_row(metric, fmt.format(va), fmt.format(vb), delta_fmt.format(vb - va))

    table.add_row("solver model", a.solver_model, b.solver_model, "")
    table.add_row(
        "verifier",
        "on" if a.use_verifier else "off",
        "on" if b.use_verifier else "off",
        "",
    )
    row("accuracy", a.accuracy * 100, b.accuracy * 100, "{:.1f}%", "{:+.1f}%")

    for level in dict.fromkeys([*a.per_level, *b.per_level]):
        ba = a.per_level.get(level, {"correct": 0, "total": 0})
        bb = b.per_level.get(level, {"correct": 0, "total": 0})
        pa = (ba["correct"] / ba["total"] * 100) if ba["total"] else 0.0
        pb = (bb["correct"] / bb["total"] * 100) if bb["total"] else 0.0
        table.add_row(
            f"accuracy [{level}]",
            _accuracy_cell(ba),
            _accuracy_cell(bb),
            f"{pb - pa:+.1f}%",
        )

    row("cost (USD)", a.total_cost_usd, b.total_cost_usd, "${:.4f}", "{:+.4f}")
    row(
        "input tokens",
        a.total_input_tokens,
        b.total_input_tokens,
        "{:,.0f}",
        "{:+,.0f}",
    )
    row(
        "output tokens",
        a.total_output_tokens,
        b.total_output_tokens,
        "{:,.0f}",
        "{:+,.0f}",
    )
    row("mean attempts", a.mean_attempts, b.mean_attempts, "{:.2f}", "{:+.2f}")
    if a.latency_p50_s is not None and b.latency_p50_s is not None:
        row("latency median (s)", a.latency_p50_s, b.latency_p50_s, "{:.1f}", "{:+.1f}")
    if a.latency_p95_s is not None and b.latency_p95_s is not None:
        row("latency p95 (s)", a.latency_p95_s, b.latency_p95_s, "{:.1f}", "{:+.1f}")

    console.print(table)
