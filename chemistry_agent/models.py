"""Shared pydantic types used across the agent team.

These types form the contract between the planner, solver, verifier,
orchestrator, eval harness, and CLI. Keep them dependency-free (pydantic
only) so every layer can import them without pulling in the SDK.
"""

from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

Level = Literal["gcse", "alevel", "university"]

Topic = Literal[
    "stoichiometry",
    "solutions",
    "gases",
    "energetics",
    "equilibrium",
    "kinetics",
    "acids_bases",
    "electrochemistry",
    "general",
]


class PlanStep(BaseModel):
    """One step of the planner's proposed solution path."""

    description: str = Field(description="What to do in this step")
    suggested_tool: Optional[str] = Field(
        default=None, description="Name of a tool likely useful for this step"
    )


class Plan(BaseModel):
    """Structured output of the planner."""

    steps: list[PlanStep] = Field(description="Ordered solution steps")
    target_quantity: str = Field(
        description="The quantity the final answer must report, e.g. 'mass of CaO'"
    )
    expected_units: Optional[str] = Field(
        default=None, description="Units the final answer should carry, e.g. 'g'"
    )
    difficulty: Level = Field(
        description="Assessed difficulty tier of the problem"
    )
    topic: Topic = Field(default="general", description="Primary chemistry topic")


class FinalAnswer(BaseModel):
    """The solver's submitted answer."""

    value: Optional[float] = Field(
        default=None, description="Numeric value of the answer, if numeric"
    )
    units: Optional[str] = Field(default=None, description="Units of the answer")
    formatted_answer: str = Field(
        description="Human-readable final answer, e.g. '18.02 g/mol'"
    )
    reasoning_summary: str = Field(
        description="Brief summary of how the answer was derived"
    )


class Verdict(BaseModel):
    """Structured output of the verifier (produced without seeing the expected answer)."""

    approved: bool = Field(description="Whether the answer passes verification")
    unit_check: bool = Field(
        description="Whether the answer's units match the plan's expected units"
    )
    magnitude_plausible: bool = Field(
        description="Whether the numeric magnitude is physically plausible"
    )
    issues: list[str] = Field(
        default_factory=list, description="Specific problems found, if any"
    )
    feedback_for_solver: Optional[str] = Field(
        default=None,
        description="Actionable feedback for a retry, present when not approved",
    )


class UsageTotals(BaseModel):
    """Accumulated token usage for one role (planner/solver/verifier)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, usage: object) -> None:
        """Accumulate an SDK ``response.usage`` object (duck-typed)."""
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_input_tokens += (
            getattr(usage, "cache_read_input_tokens", 0) or 0
        )
        self.cache_creation_input_tokens += (
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )

    def cost_usd(self, input_per_mtok: float, output_per_mtok: float) -> float:
        """Cost in USD given per-MTok input/output prices.

        Cache reads bill at ~0.1x input price; cache writes at ~1.25x.
        """
        return (
            self.input_tokens * input_per_mtok
            + self.cache_read_input_tokens * input_per_mtok * 0.1
            + self.cache_creation_input_tokens * input_per_mtok * 1.25
            + self.output_tokens * output_per_mtok
        ) / 1_000_000


class SolveOutcome(BaseModel):
    """Everything the orchestrator learned while solving one problem."""

    prompt: str
    plan: Optional[Plan] = None
    answer: Optional[FinalAnswer] = None
    verdict: Optional[Verdict] = None
    attempts: int = 0
    tool_calls: list[str] = Field(
        default_factory=list, description="Names of tools the solver invoked, in order"
    )
    usage: dict[str, UsageTotals] = Field(default_factory=dict)
    cost_usd: float = 0.0
    success: bool = False
    error: Optional[str] = None


class SolveEvent(BaseModel):
    """Progress event emitted by the orchestrator for UI rendering."""

    kind: Literal[
        "plan_ready",
        "attempt_start",
        "tool_call",
        "answer_submitted",
        "verdict",
        "done",
        "error",
    ]
    payload: dict = Field(default_factory=dict)


Observer = Callable[[SolveEvent], None]
