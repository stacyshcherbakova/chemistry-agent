"""Tests for chemistry_agent.agents.verifier — MagicMock client, no network."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from chemistry_agent.agents.verifier import VERIFIER_SYSTEM, Verifier, VerifierError
from chemistry_agent.config import AgentConfig
from chemistry_agent.models import FinalAnswer, Plan, PlanStep, Verdict


def make_plan() -> Plan:
    return Plan(
        steps=[
            PlanStep(description="Compute molar mass of H2O", suggested_tool="molar_mass"),
            PlanStep(description="Report the result"),
        ],
        target_quantity="molar mass of H2O",
        expected_units="g/mol",
        difficulty="gcse",
        topic="stoichiometry",
    )


def make_answer() -> FinalAnswer:
    return FinalAnswer(
        value=18.02,
        units="g/mol",
        formatted_answer="18.02 g/mol",
        reasoning_summary="Summed atomic masses of 2 H and 1 O.",
    )


def make_usage(input_tokens=100, output_tokens=50):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=7,
        cache_creation_input_tokens=3,
    )


def make_response(verdict, stop_reason="end_turn", usage=None):
    response = MagicMock()
    response.parsed_output = verdict
    response.stop_reason = stop_reason
    response.usage = usage if usage is not None else make_usage()
    return response


def make_client(response):
    client = MagicMock()
    client.messages.parse.return_value = response
    return client


APPROVED = Verdict(
    approved=True,
    unit_check=True,
    magnitude_plausible=True,
    issues=[],
    feedback_for_solver=None,
)


def test_verify_returns_verdict_and_calls_parse_once():
    client = make_client(make_response(APPROVED))
    verifier = Verifier(client=client, config=AgentConfig())

    verdict = verifier.verify("What is the molar mass of H2O?", make_plan(), make_answer(), ["molar_mass"])

    assert verdict is APPROVED
    client.messages.parse.assert_called_once()
    kwargs = client.messages.parse.call_args.kwargs
    assert kwargs["output_format"] is Verdict
    assert kwargs["system"][0]["text"] == VERIFIER_SYSTEM
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["model"] == AgentConfig().verifier_model
    assert kwargs["max_tokens"] == AgentConfig().verifier_max_tokens
    assert kwargs["output_config"] == {"effort": "high"}


def test_parse_kwargs_contain_no_temperature_or_thinking():
    client = make_client(make_response(APPROVED))
    verifier = Verifier(client=client)

    verifier.verify("prompt", make_plan(), make_answer(), [])

    kwargs = client.messages.parse.call_args.kwargs
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
    assert "thinking" not in kwargs


def test_user_message_renders_prompt_answer_and_tool_calls():
    client = make_client(make_response(APPROVED))
    verifier = Verifier(client=client)
    prompt = "What is the molar mass of H2O?"
    tool_calls = ["molar_mass", "calc"]

    verifier.verify(prompt, make_plan(), make_answer(), tool_calls)

    kwargs = client.messages.parse.call_args.kwargs
    messages = kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert prompt in content
    assert "18.02 g/mol" in content
    for name in tool_calls:
        assert name in content


def test_usage_accumulates_across_two_calls():
    client = make_client(
        make_response(APPROVED, usage=make_usage(input_tokens=100, output_tokens=50))
    )
    verifier = Verifier(client=client)

    verifier.verify("p1", make_plan(), make_answer(), ["molar_mass"])
    verifier.verify("p2", make_plan(), make_answer(), ["molar_mass"])

    assert verifier.usage.input_tokens == 200
    assert verifier.usage.output_tokens == 100
    assert verifier.usage.cache_read_input_tokens == 14
    assert verifier.usage.cache_creation_input_tokens == 6


def test_refusal_raises_verifier_error():
    client = make_client(make_response(APPROVED, stop_reason="refusal"))
    verifier = Verifier(client=client)

    with pytest.raises(VerifierError):
        verifier.verify("prompt", make_plan(), make_answer(), [])


def test_parsed_output_none_raises_verifier_error():
    client = make_client(make_response(None))
    verifier = Verifier(client=client)

    with pytest.raises(VerifierError):
        verifier.verify("prompt", make_plan(), make_answer(), [])


def test_rejected_with_empty_feedback_backfilled_from_issues():
    rejected = Verdict(
        approved=False,
        unit_check=False,
        magnitude_plausible=True,
        issues=["units are mL but plan expects L", "step 2 not covered by tool calls"],
        feedback_for_solver=None,
    )
    client = make_client(make_response(rejected))
    verifier = Verifier(client=client)

    verdict = verifier.verify("prompt", make_plan(), make_answer(), ["calc"])

    assert verdict.approved is False
    assert verdict.feedback_for_solver == (
        "units are mL but plan expects L; step 2 not covered by tool calls"
    )


def test_rejected_with_no_issues_gets_default_feedback():
    rejected = Verdict(
        approved=False,
        unit_check=True,
        magnitude_plausible=False,
        issues=[],
        feedback_for_solver=None,
    )
    client = make_client(make_response(rejected))
    verifier = Verifier(client=client)

    verdict = verifier.verify("prompt", make_plan(), make_answer(), [])

    assert verdict.feedback_for_solver == "Answer rejected; re-derive carefully."
