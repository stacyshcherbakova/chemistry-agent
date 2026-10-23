"""Solver agent: executes a plan with the SDK tool runner over @beta_tool tools.

The solver drives ``client.beta.messages.tool_runner`` across the full
chemistry tool registry (:data:`ALL_TOOLS`) plus a special ``submit_answer``
tool that captures the final result. The system prompt
(:data:`SOLVER_SYSTEM`) is a byte-stable module constant — do not build it
dynamically, or prompt caching breaks. Verifier feedback for retries is
appended as additional *user* messages so the system prompt and tool list
stay byte-identical across attempts.

API rules for ``claude-opus-5`` (see SPEC.md): never pass ``temperature`` /
``top_p`` / ``top_k``, never pass a ``thinking`` parameter (thinking is on by
default; depth is controlled via ``output_config.effort``), and no assistant
prefills.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Callable, Optional

from anthropic import beta_tool

from ..config import AgentConfig
from ..helper import tools as helper_tools
from ..models import FinalAnswer, Plan, UsageTotals
from ..toolkit import (
    acids_bases,
    electrochem,
    energetics,
    equilibrium,
    gases,
    kinetics,
    mathtools,
    reactions,
)

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    import anthropic


# Bound the agentic loop: a model that never calls submit_answer would
# otherwise iterate until the API stops it.
MAX_TOOL_ITERATIONS = 25


class SolverError(RuntimeError):
    """Raised when the solver cannot produce a FinalAnswer (refusal or no answer)."""


# ---------------------------------------------------------------------------
# Submission capture
# ---------------------------------------------------------------------------

# The tool runner executes tools itself, so ``submit_answer`` cannot return the
# answer to the Solver directly. Instead it records the submission in a
# context variable, which ``solve()`` resets at the start of each attempt and
# reads once the runner loop finishes. ContextVars are safe under threads and
# asyncio tasks alike.
_SUBMISSION: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "_SUBMISSION", default=None
)


@beta_tool
def submit_answer(
    value: float,
    units: str,
    formatted_answer: str,
    reasoning_summary: str,
    insufficient_data: bool = False,
) -> str:
    """Submit the final answer to the problem. Call this exactly once, at the end.

    If the answer is non-numeric (for example a balanced chemical equation or
    an empirical formula), pass ``value=0`` and put the full answer in
    ``formatted_answer``.

    Args:
        value (float): Numeric value of the final answer. Use 0 when the
            answer is non-numeric.
        units (str): Units of the answer (e.g. 'g', 'mol/L'). Use an empty
            string when the answer is unitless or non-numeric.
        formatted_answer (str): Human-readable final answer carrying value and
            units in the conventional format, e.g. '18.02 g/mol'.
        reasoning_summary (str): Brief summary of how the answer was derived.
        insufficient_data (bool): True only when the problem cannot be
            answered as written because data is missing, contradictory, or
            physically impossible. Then pass value=0, units='' and explain
            what is wrong in formatted_answer. Never use it to dodge a hard
            but answerable problem.

    Returns:
        str: Confirmation that the answer was recorded.
    """
    _SUBMISSION.set(
        {
            "value": value,
            "units": units,
            "formatted_answer": formatted_answer,
            "reasoning_summary": reasoning_summary,
            "insufficient_data": insufficient_data,
        }
    )
    return "Answer recorded. You may stop."


# ---------------------------------------------------------------------------
# Wrappers for functions whose raw signatures do not schema-generate cleanly
# ---------------------------------------------------------------------------


def empirical_formula(elements: list[str], mass_percentages: list[float]) -> str:
    """Derive the empirical formula of a compound from mass percentages.

    Assumes the percentages sum to 100 and treats them as grams of each
    element in a 100 g sample.

    Args:
        elements (list[str]): Element symbols, e.g. ['C', 'H', 'O'].
        mass_percentages (list[float]): Mass percent of each element, parallel
            to ``elements`` (e.g. [40.0, 6.7, 53.3]).

    Returns:
        str: Empirical formula, e.g. 'CH2O'.
    """
    if len(elements) != len(mass_percentages):
        raise ValueError("elements and mass_percentages must have the same length")
    return helper_tools.empirical_formula(dict(zip(elements, mass_percentages)))


def calc(expression: str) -> str:
    """Safely evaluate a plain arithmetic expression.

    Only numeric literals, + - * / ** % // , unary +/-, parentheses and the
    functions log10(x), ln(x), exp(x), sqrt(x) are allowed — no other names,
    calls, or attribute access. Use this for EVERY arithmetic step, however
    simple.

    Args:
        expression (str): Arithmetic expression, e.g. '0.1132 / 0.5' or
            '-log10(1.8e-5)'.

    Returns:
        str: The numeric result of the expression as a string.
    """
    return str(helper_tools.calc(expression))


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_REGISTRY_FUNCTIONS = [
    # helper.tools (GCSE core)
    helper_tools.lookup_molar_mass,
    helper_tools.compute_moles,
    helper_tools.mass_from_moles,
    helper_tools.mix_concentration,
    helper_tools.dilution,
    helper_tools.percent_yield,
    helper_tools.energy_change,
    helper_tools.pH_from_H,
    helper_tools.calculate_density,
    helper_tools.titration_end_point,
    helper_tools.concentration_from_moles,
    helper_tools.mass_from_density,
    empirical_formula,  # wrapper: list params instead of a dict
    calc,  # wrapper: clean single-arg signature, str return
    # toolkit.gases
    gases.ideal_gas,
    gases.gas_volume_rtp,
    gases.combined_gas_law,
    gases.partial_pressures,
    # toolkit.energetics
    energetics.hess_law,
    energetics.bond_energy_enthalpy,
    energetics.gibbs_free_energy,
    # toolkit.reactions
    reactions.balance_equation,
    reactions.limiting_reagent,
    reactions.atom_economy,
    # toolkit.equilibrium
    equilibrium.equilibrium_kc,
    equilibrium.equilibrium_kp,
    # toolkit.kinetics
    kinetics.rate_law_order,
    kinetics.arrhenius,
    kinetics.arrhenius_two_temperatures,
    kinetics.half_life_first_order,
    # toolkit.acids_bases
    acids_bases.weak_acid_pH,
    acids_bases.buffer_pH,
    acids_bases.poh_conversions,
    acids_bases.ksp_solubility,
    # toolkit.electrochem
    electrochem.cell_potential,
    electrochem.nernst_equation,
    electrochem.electrolysis_mass,
    # toolkit.mathtools (mathtools.calc skipped — already exposed above)
    mathtools.convert_units,
    mathtools.solve_equation,
]

ALL_TOOLS: list = [beta_tool(fn) for fn in _REGISTRY_FUNCTIONS]


# ---------------------------------------------------------------------------
# System prompt (byte-stable — never build dynamically)
# ---------------------------------------------------------------------------

SOLVER_SYSTEM = """\
You are an expert scientific problem-solver working through a chemistry
problem step by step.

Rules you MUST follow:
1. Use the provided tools for EVERY calculation — no mental arithmetic, not
   even simple multiplication. If no dedicated tool fits an arithmetic step,
   use the calc tool.
2. Follow the plan you are given, but adapt if a step fails or turns out to
   be wrong: choose a different tool or approach and continue.
3. You MUST finish by calling submit_answer exactly once with the final
   result. The formatted_answer should carry the value and units in the
   conventional format (e.g. '18.02 g/mol'). If the answer is non-numeric
   (e.g. a balanced equation), pass value=0 and put the full answer in
   formatted_answer.
4. Never invent data. If the problem cannot be answered as written because a
   needed quantity is missing, two given values contradict each other, or the
   setup is physically impossible, call submit_answer with
   insufficient_data=true, value=0, units='' and a formatted_answer that says
   exactly what is missing or inconsistent. Standard constants and textbook
   values a student would be expected to know (atomic masses, R, F, Kw) are
   not missing data.
"""


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _render_plan(plan: Plan) -> str:
    """Render a Plan into the text appended to the first user message."""
    lines = ["Plan:"]
    for index, step in enumerate(plan.steps, start=1):
        suffix = f" (suggested tool: {step.suggested_tool})" if step.suggested_tool else ""
        lines.append(f"{index}. {step.description}{suffix}")
    lines.append(f"Target quantity: {plan.target_quantity}")
    lines.append(f"Expected units: {plan.expected_units or '(unitless)'}")
    return "\n".join(lines)


class Solver:
    """Solves a chemistry problem with the tool runner, following a Plan."""

    def __init__(
        self,
        client: "anthropic.Anthropic | None" = None,
        config: AgentConfig | None = None,
    ) -> None:
        self._client = client
        self.config = config or AgentConfig()
        self.usage = UsageTotals()

    @property
    def client(self) -> "anthropic.Anthropic":
        """The Anthropic client, created lazily so tests never need a key."""
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def solve(
        self,
        prompt: str,
        plan: Plan,
        effort: str,
        feedback_history: Optional[list[str]] = None,
        on_tool_call: "Callable[[str], None] | None" = None,
    ) -> tuple[FinalAnswer, list[str]]:
        """Solve ``prompt`` following ``plan``.

        Args:
            prompt: The chemistry problem statement.
            plan: The planner's decomposition (steps, target, units).
            effort: ``output_config.effort`` level for this attempt.
            feedback_history: Verifier feedback from earlier rejected
                attempts; each entry becomes an additional user message so
                the system prompt and tool list stay byte-identical across
                attempts (prompt caching).
            on_tool_call: Optional callback fired with each tool name, in
                order, as tool_use blocks are observed.

        Returns:
            tuple[FinalAnswer, list[str]]: The answer and the ordered list of
            tool names the solver invoked.

        Raises:
            SolverError: On refusal, or when no answer could be produced.
        """
        config = self.config

        messages: list[dict] = [
            {"role": "user", "content": f"{prompt}\n\n{_render_plan(plan)}"}
        ]
        for feedback in feedback_history or []:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "A verifier reviewed your previous attempt and rejected "
                        f"it: {feedback}. Please redo the solution, addressing "
                        "the feedback, and submit again."
                    ),
                }
            )

        # Reset the capture for this attempt so a stale submission from an
        # earlier attempt can never leak into this one.
        _SUBMISSION.set(None)

        runner = self.client.beta.messages.tool_runner(
            model=config.solver_model,
            max_tokens=config.solver_max_tokens,
            output_config={"effort": effort},
            system=SOLVER_SYSTEM,
            tools=[*ALL_TOOLS, submit_answer],
            messages=messages,
            # The system prompt and tool list are byte-stable across attempts,
            # so the whole prefix (~39 tool schemas) is cacheable.
            cache_control={"type": "ephemeral"},
            max_iterations=MAX_TOOL_ITERATIONS,
        )

        tool_call_names: list[str] = []
        last_message = None
        for message in runner:
            last_message = message
            self.usage.add(message.usage)
            for block in message.content:
                if getattr(block, "type", None) == "tool_use":
                    tool_call_names.append(block.name)
                    if on_tool_call is not None:
                        on_tool_call(block.name)
            if message.stop_reason == "refusal":
                raise SolverError("Solver request was refused by the model")
            if message.stop_reason == "max_tokens":
                raise SolverError(
                    "Solver hit max_tokens before finishing; thinking and text "
                    f"share the {config.solver_max_tokens}-token budget. "
                    "Raise solver_max_tokens or lower the effort level."
                )

        submission = _SUBMISSION.get()
        if submission is not None:
            return FinalAnswer(**submission), tool_call_names

        # Fallback: the model never called submit_answer — extract a
        # FinalAnswer from the last text block of the final message.
        text = None
        if last_message is not None:
            for block in last_message.content:
                if getattr(block, "type", None) == "text":
                    text = block.text
        if text:
            response = self.client.messages.parse(
                model=config.solver_model,
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Extract the final answer from this solution text:"
                            f"\n{text}"
                        ),
                    }
                ],
                output_format=FinalAnswer,
            )
            self.usage.add(response.usage)
            if response.parsed_output is not None:
                return response.parsed_output, tool_call_names

        raise SolverError("Solver produced no answer (no submission, no parsable text)")
