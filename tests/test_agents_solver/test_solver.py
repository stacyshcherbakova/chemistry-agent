"""Solver tests — MagicMock client only, no network, no ANTHROPIC_API_KEY."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from chemistry_agent.agents.solver import (
    _SUBMISSION,
    ALL_TOOLS,
    SOLVER_SYSTEM,
    Solver,
    SolverError,
    submit_answer,
)
from chemistry_agent.config import AgentConfig
from chemistry_agent.models import FinalAnswer, Plan, PlanStep, UsageTotals


def make_plan() -> Plan:
    return Plan(
        steps=[
            PlanStep(
                description="Look up the molar mass of NaCl",
                suggested_tool="lookup_molar_mass",
            ),
            PlanStep(description="Divide mass by molar mass", suggested_tool="calc"),
        ],
        target_quantity="moles of NaCl",
        expected_units="mol",
        difficulty="gcse",
        topic="stoichiometry",
    )


def make_usage(input_tokens=100, output_tokens=40):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=7,
        cache_creation_input_tokens=3,
    )


def tool_use_block(name):
    return SimpleNamespace(type="tool_use", name=name, id="toolu_x", input={})


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def make_message(blocks, stop_reason="tool_use", input_tokens=100, output_tokens=40):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        usage=make_usage(input_tokens, output_tokens),
    )


SUBMISSION_INPUT = {
    "value": 1.0,
    "units": "mol",
    "formatted_answer": "1.00 mol",
    "reasoning_summary": "58.44 g / 58.44 g/mol = 1.00 mol",
}


def submitting_runner(messages):
    """Iterator over fake runner messages that executes the real submit_answer
    tool mid-iteration, exactly as the SDK tool runner would."""
    for i, message in enumerate(messages):
        if i == len(messages) - 1:
            submit_answer.call(SUBMISSION_INPUT)
        yield message


@pytest.fixture(autouse=True)
def clear_submission():
    _SUBMISSION.set(None)
    yield
    _SUBMISSION.set(None)


@pytest.fixture
def client():
    return MagicMock()


@pytest.fixture
def solver(client):
    return Solver(client=client, config=AgentConfig())


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_at_least_30_tools(self):
        assert len(ALL_TOOLS) >= 30

    def test_all_tool_names_distinct(self):
        names = [tool.name for tool in ALL_TOOLS]
        assert len(names) == len(set(names))

    def test_submit_answer_not_in_all_tools(self):
        assert "submit_answer" not in {tool.name for tool in ALL_TOOLS}

    def test_every_tool_has_schema_and_description(self):
        for tool in ALL_TOOLS:
            assert tool.name
            assert tool.description
            assert tool.input_schema["type"] == "object"

    def _tool(self, name):
        return next(tool for tool in ALL_TOOLS if tool.name == name)

    def test_empirical_formula_wrapper(self):
        tool = self._tool("empirical_formula")
        result = tool.call(
            {"elements": ["C", "H", "O"], "mass_percentages": [40, 6.7, 53.3]}
        )
        assert result == "CH2O"

    def test_empirical_formula_wrapper_rejects_length_mismatch(self):
        tool = self._tool("empirical_formula")
        with pytest.raises(ValueError):
            tool.call({"elements": ["C", "H"], "mass_percentages": [40.0]})

    def test_calc_wrapper_returns_string(self):
        tool = self._tool("calc")
        result = tool.call({"expression": "0.5 * 4"})
        assert isinstance(result, str)
        assert float(result) == 2.0

    def test_lookup_molar_mass_tool(self):
        assert self._tool("lookup_molar_mass").call({"formula": "H2O"}) == "18.02 g/mol"

    def test_balance_equation_tool(self):
        result = self._tool("balance_equation").call(
            {"equation": "Fe + O2 -> Fe2O3"}
        )
        assert result == "4Fe + 3O2 -> 2Fe2O3"

    def test_solve_equation_tool(self):
        result = self._tool("solve_equation").call(
            {"equation": "2*x = 8", "unknown": "x"}
        )
        assert result == "x = 4.000"


# ---------------------------------------------------------------------------
# submit_answer capture
# ---------------------------------------------------------------------------


class TestSubmitAnswerCapture:
    def test_call_records_submission_in_contextvar(self):
        result = submit_answer.call(SUBMISSION_INPUT)
        assert result == "Answer recorded. You may stop."
        assert _SUBMISSION.get() == {**SUBMISSION_INPUT, "insufficient_data": False}

    def test_insufficient_data_flag_recorded(self):
        submit_answer.func(
            value=0,
            units="",
            formatted_answer="Cannot be answered: the volume of solution is not given.",
            reasoning_summary="Concentration needs a volume.",
            insufficient_data=True,
        )
        submission = _SUBMISSION.get()
        assert submission["insufficient_data"] is True

    def test_system_prompt_forbids_inventing_data(self):
        assert "insufficient_data=true" in SOLVER_SYSTEM
        assert "Never invent data" in SOLVER_SYSTEM

    def test_underlying_function_records_submission(self):
        submit_answer.func(
            value=0,
            units="",
            formatted_answer="4Fe + 3O2 -> 2Fe2O3",
            reasoning_summary="Balanced by inspection tool",
        )
        submission = _SUBMISSION.get()
        assert submission is not None
        assert submission["value"] == 0
        assert submission["formatted_answer"] == "4Fe + 3O2 -> 2Fe2O3"


# ---------------------------------------------------------------------------
# solve() — happy path
# ---------------------------------------------------------------------------


class TestSolveHappyPath:
    def make_runner_messages(self):
        return [
            make_message(
                [
                    text_block("Looking up molar mass"),
                    tool_use_block("lookup_molar_mass"),
                    tool_use_block("calc"),
                ],
                stop_reason="tool_use",
                input_tokens=100,
                output_tokens=40,
            ),
            make_message(
                [tool_use_block("submit_answer")],
                stop_reason="tool_use",
                input_tokens=50,
                output_tokens=20,
            ),
        ]

    def test_returns_final_answer_and_ordered_tool_calls(self, client, solver):
        client.beta.messages.tool_runner.return_value = submitting_runner(
            self.make_runner_messages()
        )

        answer, tool_calls = solver.solve(
            "How many moles in 58.44 g of NaCl?", make_plan(), effort="medium"
        )

        assert isinstance(answer, FinalAnswer)
        assert answer.value == 1.0
        assert answer.units == "mol"
        assert answer.formatted_answer == "1.00 mol"
        assert tool_calls == ["lookup_molar_mass", "calc", "submit_answer"]
        client.messages.parse.assert_not_called()

    def test_on_tool_call_fires_in_order(self, client, solver):
        client.beta.messages.tool_runner.return_value = submitting_runner(
            self.make_runner_messages()
        )
        seen = []

        solver.solve(
            "How many moles?", make_plan(), effort="medium", on_tool_call=seen.append
        )

        assert seen == ["lookup_molar_mass", "calc", "submit_answer"]

    def test_usage_accumulated_across_runner_messages(self, client, solver):
        client.beta.messages.tool_runner.return_value = submitting_runner(
            self.make_runner_messages()
        )

        solver.solve("How many moles?", make_plan(), effort="medium")

        assert solver.usage.input_tokens == 150
        assert solver.usage.output_tokens == 60
        assert solver.usage.cache_read_input_tokens == 14
        assert solver.usage.cache_creation_input_tokens == 6

    def test_stale_submission_from_earlier_attempt_is_cleared(self, client, solver):
        """A leftover submission must not satisfy an attempt that never submits."""
        _SUBMISSION.set(SUBMISSION_INPUT)
        client.beta.messages.tool_runner.return_value = iter(
            [make_message([], stop_reason="end_turn")]
        )
        client.messages.parse.return_value = SimpleNamespace(
            parsed_output=None, usage=make_usage()
        )

        with pytest.raises(SolverError):
            solver.solve("A problem", make_plan(), effort="medium")


# ---------------------------------------------------------------------------
# solve() — request shape
# ---------------------------------------------------------------------------


class TestToolRunnerCallShape:
    def run_solve(self, client, solver, feedback_history=None):
        client.beta.messages.tool_runner.return_value = submitting_runner(
            [make_message([tool_use_block("submit_answer")])]
        )
        solver.solve(
            "How many moles in 58.44 g of NaCl?",
            make_plan(),
            effort="high",
            feedback_history=feedback_history,
        )
        return client.beta.messages.tool_runner.call_args.kwargs

    def test_expected_kwargs(self, client, solver):
        config = AgentConfig()
        kwargs = self.run_solve(client, solver)

        assert kwargs["model"] == config.solver_model
        assert kwargs["max_tokens"] == config.solver_max_tokens
        assert kwargs["output_config"] == {"effort": "high"}
        assert kwargs["system"] is SOLVER_SYSTEM
        assert kwargs["tools"] == [*ALL_TOOLS, submit_answer]

    def test_forbidden_kwargs_absent(self, client, solver):
        """claude-opus-5 rejects sampling params and any thinking config."""
        kwargs = self.run_solve(client, solver)
        for forbidden in ("temperature", "top_p", "top_k", "thinking"):
            assert forbidden not in kwargs

    def test_first_message_contains_prompt_and_plan(self, client, solver):
        kwargs = self.run_solve(client, solver)
        messages = kwargs["messages"]

        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert content.startswith("How many moles in 58.44 g of NaCl?")
        assert "Look up the molar mass of NaCl" in content
        assert "moles of NaCl" in content  # target quantity
        assert "mol" in content  # expected units

    def test_feedback_history_appended_as_user_messages(self, client, solver):
        kwargs = self.run_solve(
            client, solver, feedback_history=["Units were wrong", "Value off by 10x"]
        )
        messages = kwargs["messages"]

        assert len(messages) == 3
        for message, feedback in zip(messages[1:], ["Units were wrong", "Value off by 10x"]):
            assert message["role"] == "user"
            assert message["content"] == (
                "A verifier reviewed your previous attempt and rejected it: "
                f"{feedback}. Please redo the solution, addressing the "
                "feedback, and submit again."
            )

    def test_system_and_tools_byte_identical_across_attempts(self, client, solver):
        """Prompt caching: retries change only the user messages."""
        client.beta.messages.tool_runner.side_effect = [
            submitting_runner([make_message([tool_use_block("submit_answer")])]),
            submitting_runner([make_message([tool_use_block("submit_answer")])]),
        ]

        solver.solve("A problem", make_plan(), effort="medium")
        solver.solve(
            "A problem", make_plan(), effort="medium",
            feedback_history=["Wrong units"],
        )

        first = client.beta.messages.tool_runner.call_args_list[0].kwargs
        second = client.beta.messages.tool_runner.call_args_list[1].kwargs
        assert first["system"] == second["system"]
        assert first["system"] is SOLVER_SYSTEM
        assert second["system"] is SOLVER_SYSTEM
        assert first["tools"] == second["tools"]
        assert len(first["messages"]) == 1
        assert len(second["messages"]) == 2


# ---------------------------------------------------------------------------
# solve() — fallback extraction path
# ---------------------------------------------------------------------------


class TestFallbackExtraction:
    def test_parse_called_when_no_submission(self, client, solver):
        config = AgentConfig()
        client.beta.messages.tool_runner.return_value = iter(
            [
                make_message(
                    [text_block("The answer is 1.00 mol of NaCl.")],
                    stop_reason="end_turn",
                    input_tokens=80,
                    output_tokens=30,
                )
            ]
        )
        extracted = FinalAnswer(
            value=1.0,
            units="mol",
            formatted_answer="1.00 mol",
            reasoning_summary="Extracted from solution text",
        )
        client.messages.parse.return_value = SimpleNamespace(
            parsed_output=extracted, usage=make_usage(input_tokens=60, output_tokens=25)
        )

        answer, tool_calls = solver.solve("A problem", make_plan(), effort="medium")

        assert answer is extracted
        assert tool_calls == []
        client.messages.parse.assert_called_once()
        kwargs = client.messages.parse.call_args.kwargs
        assert kwargs["model"] == config.solver_model
        assert kwargs["max_tokens"] == 2048
        assert kwargs["output_format"] is FinalAnswer
        assert kwargs["messages"] == [
            {
                "role": "user",
                "content": (
                    "Extract the final answer from this solution text:\n"
                    "The answer is 1.00 mol of NaCl."
                ),
            }
        ]
        # Usage counts runner message + parse call.
        assert solver.usage.input_tokens == 140
        assert solver.usage.output_tokens == 55

    def test_no_submission_and_no_text_raises(self, client, solver):
        client.beta.messages.tool_runner.return_value = iter(
            [make_message([], stop_reason="end_turn")]
        )

        with pytest.raises(SolverError):
            solver.solve("A problem", make_plan(), effort="medium")

        client.messages.parse.assert_not_called()

    def test_parse_returning_none_raises(self, client, solver):
        client.beta.messages.tool_runner.return_value = iter(
            [make_message([text_block("some text")], stop_reason="end_turn")]
        )
        client.messages.parse.return_value = SimpleNamespace(
            parsed_output=None, usage=make_usage()
        )

        with pytest.raises(SolverError):
            solver.solve("A problem", make_plan(), effort="medium")


# ---------------------------------------------------------------------------
# solve() — refusal and construction
# ---------------------------------------------------------------------------


class TestRefusal:
    def test_refusal_raises_solver_error(self, client, solver):
        client.beta.messages.tool_runner.return_value = iter(
            [make_message([text_block("")], stop_reason="refusal")]
        )

        with pytest.raises(SolverError):
            solver.solve("A refused problem", make_plan(), effort="medium")

    def test_usage_still_accumulated_on_refusal(self, client, solver):
        client.beta.messages.tool_runner.return_value = iter(
            [
                make_message(
                    [], stop_reason="refusal", input_tokens=99, output_tokens=1
                )
            ]
        )

        with pytest.raises(SolverError):
            solver.solve("A refused problem", make_plan(), effort="medium")

        assert solver.usage.input_tokens == 99
        assert solver.usage.output_tokens == 1


class TestConstruction:
    def test_solver_error_is_runtime_error(self):
        assert issubclass(SolverError, RuntimeError)

    def test_defaults_config_when_none(self, client):
        solver = Solver(client=client)
        assert isinstance(solver.config, AgentConfig)

    def test_stores_given_config(self, client):
        config = AgentConfig(solver_model="claude-sonnet-5")
        solver = Solver(client=client, config=config)
        assert solver.config is config

    def test_starts_with_zero_usage(self, client):
        solver = Solver(client=client)
        assert solver.usage == UsageTotals()

    def test_no_client_construction_at_init(self):
        """client=None must not eagerly build anthropic.Anthropic() in __init__."""
        solver = Solver()
        assert solver._client is None
