"""Planner tests — MagicMock client only, no network, no ANTHROPIC_API_KEY."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from chemistry_agent.agents.planner import PLANNER_SYSTEM, Planner, PlannerError
from chemistry_agent.config import AgentConfig
from chemistry_agent.models import Plan, PlanStep, UsageTotals


def make_plan() -> Plan:
    return Plan(
        steps=[
            PlanStep(
                description="Look up the molar mass of NaCl",
                suggested_tool="molar_mass",
            ),
            PlanStep(
                description="Convert mass to moles",
                suggested_tool="compute_moles",
            ),
        ],
        target_quantity="moles of NaCl",
        expected_units="mol",
        difficulty="gcse",
        topic="stoichiometry",
    )


def make_response(
    plan=None,
    stop_reason="end_turn",
    input_tokens=120,
    output_tokens=45,
):
    response = MagicMock()
    response.parsed_output = plan
    response.stop_reason = stop_reason
    response.usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=7,
        cache_creation_input_tokens=3,
    )
    return response


@pytest.fixture
def client():
    return MagicMock()


@pytest.fixture
def planner(client):
    return Planner(client=client, config=AgentConfig())


class TestPlanReturnsParsedOutput:
    def test_returns_the_parsed_plan(self, client, planner):
        plan = make_plan()
        client.messages.parse.return_value = make_response(plan=plan)

        result = planner.plan("How many moles in 58.44 g of NaCl?")

        assert result is plan

    def test_usage_is_accumulated(self, client, planner):
        client.messages.parse.return_value = make_response(
            plan=make_plan(), input_tokens=120, output_tokens=45
        )

        planner.plan("How many moles in 58.44 g of NaCl?")

        assert planner.usage.input_tokens == 120
        assert planner.usage.output_tokens == 45
        assert planner.usage.cache_read_input_tokens == 7
        assert planner.usage.cache_creation_input_tokens == 3

    def test_usage_accumulates_across_calls(self, client, planner):
        client.messages.parse.return_value = make_response(
            plan=make_plan(), input_tokens=100, output_tokens=50
        )

        planner.plan("Problem one")
        planner.plan("Problem two")

        assert planner.usage.input_tokens == 200
        assert planner.usage.output_tokens == 100


class TestParseCallShape:
    def test_parse_called_once_with_expected_kwargs(self, client, planner):
        client.messages.parse.return_value = make_response(plan=make_plan())
        config = AgentConfig()

        planner.plan("What is the pH of 0.1 M HCl?")

        client.messages.parse.assert_called_once()
        kwargs = client.messages.parse.call_args.kwargs
        assert kwargs["model"] == config.planner_model
        assert kwargs["max_tokens"] == config.planner_max_tokens
        assert kwargs["system"][0]["text"] == PLANNER_SYSTEM
        assert kwargs["output_format"] is Plan
        assert kwargs["output_config"] == {"effort": "high"}
        assert kwargs["messages"] == [
            {"role": "user", "content": "What is the pH of 0.1 M HCl?"}
        ]

    def test_forbidden_kwargs_absent(self, client, planner):
        """claude-opus-5 rejects sampling params and any thinking config."""
        client.messages.parse.return_value = make_response(plan=make_plan())

        planner.plan("What is the pH of 0.1 M HCl?")

        kwargs = client.messages.parse.call_args.kwargs
        for forbidden in ("temperature", "top_p", "top_k", "thinking"):
            assert forbidden not in kwargs

    def test_system_prompt_is_the_module_constant(self, client, planner):
        client.messages.parse.return_value = make_response(plan=make_plan())

        planner.plan("Balance the equation H2 + O2 -> H2O")

        assert (
            client.messages.parse.call_args.kwargs["system"][0]["text"]
            == PLANNER_SYSTEM
        )


class TestLevelHint:
    def test_level_hint_appended_to_user_message(self, client, planner):
        client.messages.parse.return_value = make_response(plan=make_plan())

        planner.plan("Find the cell potential.", level_hint="university")

        messages = client.messages.parse.call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"].startswith("Find the cell potential.")
        assert "The user indicates this is university-level." in messages[0]["content"]

    def test_level_hint_does_not_touch_system_prompt(self, client, planner):
        """The hint must land in the user message; the system stays byte-stable."""
        client.messages.parse.return_value = make_response(plan=make_plan())

        planner.plan("Find the cell potential.", level_hint="alevel")

        kwargs = client.messages.parse.call_args.kwargs
        # Identity check: the exact module constant is passed, not a copy or
        # a hint-augmented variant.
        assert kwargs["system"][0]["text"] is PLANNER_SYSTEM

    def test_no_level_hint_sends_prompt_verbatim(self, client, planner):
        client.messages.parse.return_value = make_response(plan=make_plan())

        planner.plan("Find the cell potential.")

        messages = client.messages.parse.call_args.kwargs["messages"]
        assert messages[0]["content"] == "Find the cell potential."


class TestErrorPaths:
    def test_refusal_raises_planner_error(self, client, planner):
        client.messages.parse.return_value = make_response(
            plan=make_plan(), stop_reason="refusal"
        )

        with pytest.raises(PlannerError):
            planner.plan("A problem the classifiers dislike")

    def test_none_parsed_output_raises_planner_error(self, client, planner):
        client.messages.parse.return_value = make_response(plan=None)

        with pytest.raises(PlannerError):
            planner.plan("A problem yielding no parse")

    def test_planner_error_is_runtime_error(self):
        assert issubclass(PlannerError, RuntimeError)

    def test_usage_still_accumulated_on_refusal(self, client, planner):
        client.messages.parse.return_value = make_response(
            plan=make_plan(), stop_reason="refusal", input_tokens=99, output_tokens=1
        )

        with pytest.raises(PlannerError):
            planner.plan("A refused problem")

        assert planner.usage.input_tokens == 99
        assert planner.usage.output_tokens == 1


class TestConstruction:
    def test_defaults_config_when_none(self, client):
        planner = Planner(client=client)
        assert isinstance(planner.config, AgentConfig)

    def test_stores_given_config(self, client):
        config = AgentConfig(planner_model="claude-sonnet-5")
        planner = Planner(client=client, config=config)
        assert planner.config is config

    def test_starts_with_zero_usage(self, client):
        planner = Planner(client=client)
        assert planner.usage == UsageTotals()

    def test_no_client_construction_at_init(self):
        """client=None must not eagerly build anthropic.Anthropic() in __init__."""
        planner = Planner()
        assert planner._client is None


class TestSystemPromptContent:
    def test_mentions_difficulty_tiers(self):
        for level in ("gcse", "alevel", "university"):
            assert level in PLANNER_SYSTEM

    def test_catalog_contains_real_tool_names(self):
        """Every name the prompt suggests must be a real registered tool."""
        from chemistry_agent.helper.tools import TOOLS
        from chemistry_agent.toolkit import (
            acids_bases,
            electrochem,
            energetics,
            equilibrium,
            gases,
            kinetics,
            mathtools,
            reactions,
        )

        real_names = set(TOOLS)
        for module in (
            gases, energetics, equilibrium, kinetics,
            acids_bases, electrochem, reactions, mathtools,
        ):
            real_names.update(
                name for name in dir(module)
                if not name.startswith("_") and callable(getattr(module, name))
            )

        catalog_names = [
            "molar_mass", "compute_moles", "mass_from_moles", "mix_concentration",
            "dilution", "percent_yield", "energy_change", "pH_from_H",
            "calculate_density", "empirical_formula", "titration",
            "concentration_from_moles", "mass_from_density",
            "ideal_gas", "gas_volume_rtp", "combined_gas_law", "partial_pressures",
            "hess_law", "bond_energy_enthalpy", "gibbs_free_energy",
            "equilibrium_kc", "equilibrium_kp",
            "rate_law_order", "arrhenius", "arrhenius_two_temperatures",
            "half_life_first_order",
            "weak_acid_pH", "buffer_pH", "poh_conversions", "ksp_solubility",
            "cell_potential", "nernst_equation", "electrolysis_mass",
            "balance_equation", "limiting_reagent", "atom_economy",
            "convert_units", "solve_equation", "calc",
        ]
        for name in catalog_names:
            assert name in PLANNER_SYSTEM, f"{name} missing from PLANNER_SYSTEM"
            assert name in real_names, f"{name} is not a real tool"
