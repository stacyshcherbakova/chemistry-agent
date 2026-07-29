"""Configuration for the agent team: models per role, effort, pricing, tolerances."""

from pydantic import BaseModel, Field

from .models import Level

DEFAULT_MODEL = "claude-opus-5"

# USD per million tokens: (input, output)
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Harder problems get more reasoning depth.
LEVEL_EFFORT: dict[str, str] = {
    "gcse": "medium",
    "alevel": "high",
    # "xhigh" wants a much larger max_tokens than the SDK allows on a
    # non-streaming request (it refuses above ~32k), and thinking shares
    # that budget — so university runs at "high" until the solver streams.
    "university": "high",
}


class AgentConfig(BaseModel):
    """Runtime configuration shared by the orchestrator and its agents."""

    planner_model: str = DEFAULT_MODEL
    solver_model: str = DEFAULT_MODEL
    verifier_model: str = DEFAULT_MODEL

    # max_retries is the number of *re*-attempts after the first solve,
    # so total solver attempts = max_retries + 1.
    max_retries: int = 2

    planner_max_tokens: int = 4096
    verifier_max_tokens: int = 4096
    solver_max_tokens: int = 16000

    # Relative tolerance used by the deterministic comparator in evals/tests.
    rel_tolerance: float = 0.01

    default_level: Level = "gcse"

    def effort_for(self, level: str) -> str:
        return LEVEL_EFFORT.get(level, "high")

    def price_for(self, model: str) -> tuple[float, float]:
        try:
            return PRICING[model]
        except KeyError:
            # Silently falling back to Opus pricing would corrupt exactly
            # the cross-model cost comparison `chem compare` exists for.
            raise KeyError(
                f"No pricing for {model!r}; add it to PRICING in config.py"
            ) from None


def cost_usd(model: str, usage_totals, config: AgentConfig | None = None) -> float:
    """USD cost for one role's accumulated usage on a given model."""
    inp, out = (config or AgentConfig()).price_for(model)
    return usage_totals.cost_usd(inp, out)
