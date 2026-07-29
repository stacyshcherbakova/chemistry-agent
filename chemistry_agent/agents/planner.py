"""Planner agent: decomposes a chemistry problem into an ordered, tool-oriented plan.

Calls ``client.messages.parse`` with ``output_format=Plan`` so the response is a
validated :class:`~chemistry_agent.models.Plan`. The system prompt
(:data:`PLANNER_SYSTEM`) is a byte-stable module constant — do not build it
dynamically, or prompt caching breaks.

API rules for ``claude-opus-5`` (see SPEC.md): never pass ``temperature`` /
``top_p`` / ``top_k``, never pass a ``thinking`` parameter (thinking is on by
default; depth is controlled via ``output_config.effort``), and no assistant
prefills.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..config import AgentConfig
from ..models import Plan, UsageTotals

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    import anthropic


class PlannerError(RuntimeError):
    """Raised when the planner cannot produce a Plan (refusal or empty parse)."""


# NOTE: keep this constant byte-stable. It is rendered verbatim as the system
# prompt on every request so the prompt-cache prefix never changes. The tool
# catalog below is a static, hand-written mirror of the registered tool names
# (helper/tools.py TOOLS keys + the public functions in chemistry_agent/toolkit/)
# — it is deliberately NOT built by introspection at import time.
PLANNER_SYSTEM = """\
You are an expert chemistry teacher planning how to solve a chemistry problem.
You do not solve the problem yourself. Instead, you decompose it into a short,
ordered list of concrete steps that a solver can execute using calculation
tools. Every calculation must map to a tool call, so phrase each step around
the tool that performs it.

Your plan must:
1. State the target quantity — exactly what the final answer must report
   (e.g. "mass of CaO", "pH of the buffer", "equilibrium constant Kc").
2. State the expected units of the final answer (e.g. "g", "mol/L", "kJ/mol"),
   or leave units empty only if the answer is genuinely unitless (e.g. pH, an
   empirical formula).
3. Assess the difficulty as exactly one of: gcse, alevel, university.
   - gcse: a single-formula plug-in calculation (one tool call, direct
     substitution — moles, dilution, percent yield, q = mcΔT, density).
   - alevel: multi-step reasoning, or logarithmic / equilibrium reasoning
     (weak acids, buffers, Kc/Kp, gas laws chained with stoichiometry,
     Hess's law, rate orders).
   - university: electrochemistry (cell potentials, Nernst, electrolysis),
     Ksp solubility, Arrhenius with two temperatures, or chains that combine
     several distinct concepts.
4. Pick the primary topic from: stoichiometry, solutions, gases, energetics,
   equilibrium, kinetics, acids_bases, electrochemistry, general.
5. Give each step a clear description and, where one fits, the name of a
   suggested tool taken verbatim from the catalog below. Do not invent tool
   names; if no tool fits a step, omit the suggestion.

Available tools, grouped by domain:

Stoichiometry & solutions (core):
  molar_mass, compute_moles, mass_from_moles, mix_concentration, dilution,
  percent_yield, energy_change, pH_from_H, calculate_density,
  empirical_formula, titration, concentration_from_moles, mass_from_density

Gases:
  ideal_gas, gas_volume_rtp, combined_gas_law, partial_pressures

Energetics:
  hess_law, bond_energy_enthalpy, gibbs_free_energy

Equilibrium:
  equilibrium_kc, equilibrium_kp

Kinetics:
  rate_law_order, arrhenius, arrhenius_two_temperatures, half_life_first_order

Acids & bases:
  weak_acid_pH, buffer_pH, poh_conversions, ksp_solubility

Electrochemistry:
  cell_potential, nernst_equation, electrolysis_mass

Reactions:
  balance_equation, limiting_reagent, atom_economy

Math utilities:
  convert_units, solve_equation, calc

Keep the plan minimal: the fewest steps that fully solve the problem, in
execution order.
"""


class Planner:
    """Produces a :class:`Plan` for a chemistry problem via structured output."""

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

    def plan(self, prompt: str, level_hint: Optional[str] = None) -> Plan:
        """Ask the model to decompose ``prompt`` into a :class:`Plan`.

        Args:
            prompt: The chemistry problem statement.
            level_hint: Optional user-declared level (``gcse`` / ``alevel`` /
                ``university``). Appended to the *user* message so the system
                prompt stays byte-stable for prompt caching.

        Raises:
            PlannerError: If the model refused or returned no parsed output.
        """
        user_message = prompt
        if level_hint is not None:
            user_message = (
                f"{prompt}\nThe user indicates this is {level_hint}-level."
            )

        response = self.client.messages.parse(
            model=self.config.planner_model,
            max_tokens=self.config.planner_max_tokens,
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
            output_format=Plan,
            output_config={"effort": "high"},
        )

        self.usage.add(response.usage)

        if response.stop_reason == "refusal":
            raise PlannerError("Planner request was refused by the model")
        if response.parsed_output is None:
            raise PlannerError("Planner returned no parsed Plan output")

        return response.parsed_output
