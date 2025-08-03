from ..env.env import ChemistryEnv
from ..agent.agent import GuidedAgent
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field
import logging
logger = logging.getLogger(__name__)

class ProblemState(BaseModel):
    prompt: Optional[str] = None
    task:     Optional[Literal[
                  "molar_mass",
                  "compute_moles",
                  "mass_from_moles",
                  "mix_concentration",
                  "dilution",
                  "percent_yield",
                  "energy_change",
                  "pH_from_H",
                  "calculate_density",
                  "empirical_formula",
                  "titration"
              ]] = Field(
                  default=None,
                  description="(optional) which tool to start with"
              )
    params: Dict[str, Any]
    expected: str 

async def solve_problem(problem: ProblemState, env: ChemistryEnv, agent: GuidedAgent):
    MAX_CALLS = 5
    obs, tools = await env.reset()
    state = await agent.init_state(tools)
    logger.info("Problem prompt: %s", problem.prompt or "<built>")

    print("\n--- Chain‐of‐Thought ---")
    for step in range(1, MAX_CALLS + 1):
        print("--- Step", step)
        # 1) Ask the agent for its next move
        op_result, state, _ = await agent.get_asv(state, obs)
        action = op_result.value
        logger.info("LLM ToolRequest: %r", action)
        print("LLM ToolRequest:", action)

        # 2) Execute in the env
        obs, _, _, _ = await env.step(action)
        tool_output = obs[0].content.strip()
        logger.info("Tool output: %s", tool_output)
        logger.info("Expected: %s", problem.expected.strip())
        print("[Tool output]", tool_output)
        print(f"[Expected]: {problem.expected.strip()}")


        # 3) If the answer is here, stop
        if tool_output == problem.expected.strip():
            logger.info("Final answer reached: %s", tool_output)
            logger.info("Reward: %s", float(True))
            print("\nFinal agent answer:", tool_output)
            print("Expected:", problem.expected)
            print("Reward:", float(tool_output == problem.expected.strip()))

            break
    else:
        logger.warning("Max calls (%d) reached without matching expected output", MAX_CALLS)