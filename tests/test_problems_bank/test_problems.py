"""Structural tests for the problem bank (helper/problems.py)."""

import inspect
from typing import get_args

from chemistry_agent.helper.problems import examples
from chemistry_agent.helper.tools import TOOLS
from chemistry_agent.models import Level, Topic

VALID_LEVELS = set(get_args(Level))
VALID_TOPICS = set(get_args(Topic))


class TestProblemBankStructure:

    def test_count_at_least_30(self):
        assert len(examples) >= 30

    def test_every_problem_has_nonempty_expected(self):
        for i, problem in enumerate(examples):
            assert isinstance(problem.expected, str), f"problem {i}"
            assert problem.expected.strip(), f"problem {i} has empty expected"

    def test_every_problem_has_valid_level(self):
        for i, problem in enumerate(examples):
            assert problem.level in VALID_LEVELS, (
                f"problem {i} has invalid level {problem.level!r}"
            )

    def test_every_problem_has_valid_topic(self):
        for i, problem in enumerate(examples):
            assert problem.topic in VALID_TOPICS, (
                f"problem {i} has invalid topic {problem.topic!r}"
            )

    def test_every_problem_has_prompt_or_task(self):
        for i, problem in enumerate(examples):
            assert problem.prompt or problem.task, (
                f"problem {i} has neither prompt nor task"
            )

    def test_all_levels_represented(self):
        levels = {p.level for p in examples}
        assert levels == VALID_LEVELS

    def test_alevel_and_university_counts(self):
        assert sum(1 for p in examples if p.level == "alevel") >= 7
        assert sum(1 for p in examples if p.level == "university") >= 8


class TestParamsMatchToolSignatures:

    def test_task_names_are_known_tools(self):
        for i, problem in enumerate(examples):
            if problem.task is not None:
                assert problem.task in TOOLS, (
                    f"problem {i} references unknown tool {problem.task!r}"
                )

    def test_params_subset_of_tool_signature(self):
        """For problems that set a task, every param key must be a real
        parameter of the corresponding tool function (prompt-only problems
        are skipped)."""
        for i, problem in enumerate(examples):
            if problem.task is None:
                continue
            fn = TOOLS[problem.task]
            sig_params = set(inspect.signature(fn).parameters)
            extra = set(problem.params) - sig_params
            assert not extra, (
                f"problem {i} (task={problem.task!r}) has params {extra} "
                f"not accepted by {fn.__name__}{inspect.signature(fn)}"
            )
