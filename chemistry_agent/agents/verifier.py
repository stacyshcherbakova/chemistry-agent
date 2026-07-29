"""Verifier agent: independently audits a solver's answer without the expected answer.

The verifier receives the original problem, the planner's plan, the solver's
submitted :class:`FinalAnswer`, and the ordered list of tool calls the solver
made. It never sees the expected answer — it approves only answers that are
defensible on their own terms (units, dimensional consistency, magnitude
plausibility, plan coverage, formatting).
"""

from __future__ import annotations

import anthropic

from ..config import AgentConfig
from ..models import FinalAnswer, Plan, UsageTotals, Verdict

VERIFIER_SYSTEM = """\
You are a meticulous chemistry answer verifier. You are given a chemistry
problem, the solution plan, the submitted final answer, and the ordered list
of calculation tools the solver invoked. You do NOT know the expected answer;
you must audit the submission independently.

Work through this checklist:
1. UNITS: Do the answer's units match the plan's expected units (or a valid
   equivalent, e.g. mL vs L, kJ/mol vs J/mol with consistent magnitude)?
2. DIMENSIONAL CONSISTENCY: Is the chain of tools used dimensionally sound
   for deriving the target quantity from the given data?
3. MAGNITUDE PLAUSIBILITY: Is the numeric value physically plausible for this
   kind of quantity? (e.g. a molar mass of 0.18 g/mol or a pH of 45 is
   implausible; concentrations, volumes, enthalpies, and cell potentials all
   have characteristic ranges.)
4. PLAN COVERAGE: Are the plan's steps actually reflected in the tool calls
   made? Missing calculation steps mean the answer was not properly derived.
5. FORMAT & SIG FIGS: Is the formatted answer sane — sensible significant
   figures, units present when required, consistent with the numeric value?

Approve (approved=true) only when the answer is defensible on every check.
When rejecting, list the specific problems in `issues` and write
`feedback_for_solver` as specific, actionable guidance the solver can use to
correct its work on a retry — name the failing check, the offending quantity,
and what to do differently. Never reject without concrete feedback.
"""


class VerifierError(RuntimeError):
    """Raised when the verifier cannot produce a usable verdict."""


def _render_user_message(
    prompt: str, plan: Plan, answer: FinalAnswer, tool_calls: list[str]
) -> str:
    steps = "\n".join(
        f"  {i}. {step.description}"
        + (f" (suggested tool: {step.suggested_tool})" if step.suggested_tool else "")
        for i, step in enumerate(plan.steps, start=1)
    )
    calls = (
        "\n".join(f"  {i}. {name}" for i, name in enumerate(tool_calls, start=1))
        or "  (none)"
    )
    return (
        "Audit the following submission.\n\n"
        f"## Problem\n{prompt}\n\n"
        "## Plan\n"
        f"Target quantity: {plan.target_quantity}\n"
        f"Expected units: {plan.expected_units or '(unspecified)'}\n"
        f"Steps:\n{steps}\n\n"
        "## Submitted answer\n"
        f"Value: {answer.value}\n"
        f"Units: {answer.units}\n"
        f"Formatted answer: {answer.formatted_answer}\n"
        f"Reasoning summary: {answer.reasoning_summary}\n\n"
        "## Tool calls made (in order)\n"
        f"{calls}\n"
    )


class Verifier:
    """Audits a submitted answer and returns a :class:`Verdict`."""

    def __init__(
        self,
        client: "anthropic.Anthropic | None" = None,
        config: AgentConfig | None = None,
    ):
        self.client = client if client is not None else anthropic.Anthropic()
        self.config = config or AgentConfig()
        self.usage = UsageTotals()

    def verify(
        self, prompt: str, plan: Plan, answer: FinalAnswer, tool_calls: list[str]
    ) -> Verdict:
        """Independently audit ``answer`` against the plan and tool trace."""
        response = self.client.messages.parse(
            model=self.config.verifier_model,
            max_tokens=self.config.verifier_max_tokens,
            system=VERIFIER_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": _render_user_message(prompt, plan, answer, tool_calls),
                }
            ],
            output_format=Verdict,
            output_config={"effort": "high"},
        )

        self.usage.add(response.usage)

        if response.stop_reason == "refusal":
            raise VerifierError("Verifier request was refused (stop_reason=refusal)")

        verdict = response.parsed_output
        if verdict is None:
            raise VerifierError("Verifier returned no parsable Verdict")

        # The orchestrator depends on non-empty feedback to drive retries.
        if verdict.approved is False and not verdict.feedback_for_solver:
            verdict.feedback_for_solver = (
                "; ".join(verdict.issues) or "Answer rejected; re-derive carefully."
            )

        return verdict
