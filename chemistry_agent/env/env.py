from aviary.core import Environment, Message, Tool, ToolRequestMessage,ToolResponseMessage
from chemistry_agent.helper.tools import *
from typing import Tuple, List
import logging
logger = logging.getLogger(__name__)

# --- Step 1: Science (more like chemistry) Environment -> ---
class ChemistryEnv(Environment[ProblemState]):
    def __init__(self, problem: ProblemState):
          super().__init__()
          self.problem = problem
        
    async def reset(self):
        self.tools = [Tool.from_function(fun) 
                      for _, fun in TOOLS.items()]
        logger.info("Resetting environment with problem state: %r", self.problem)
        prompt = self.problem.prompt or build_prompt(self.problem)
        logger.info("Generated prompt: %s", prompt)
        return [Message(content=prompt)], self.tools

    async def step(self, action):
        """
        Execute one turn in the chemistry environment.

        - If `action` is a ToolRequestMessage with calls, run them and return the raw ToolResponseMessage(s).
        - Otherwise, treat `action` as the final answer, score it, and terminate the episode.

        Args:
            action: The agent’s proposed tool call or final answer.

        Returns:
            - observations: List of new Messages,
            - reward: 1.0 if correct, else 0.0,
            - done: True if this was the final answer,
            - truncated: False (unused).
        """
        # 1) If it’s a tool request, run it and return *native* ToolResponseMessage(s)
        if isinstance(action, ToolRequestMessage) and action.tool_calls:
            logger.info("Executing tool call: %s", action.tool_calls)
            raw = await self.exec_tool_calls(action)
            # raw may be one ToolResponseMessage or a list of them
            if isinstance(raw, ToolResponseMessage):
                obs = [raw]
            elif isinstance(raw, list) and all(isinstance(r, ToolResponseMessage) for r in raw):
                obs = raw
            else:
                # (shouldn’t happen—just in case)
                obs = [Message(content=str(raw))]
            logger.info("Tool response: %r", obs)
            return obs, 0.0, False, False
        
        # 2) Otherwise `action` is the final assistant Message—score and finish
        assert isinstance(action, Message) and action.content is not None
        reward = float(action.content.strip() == self.problem.expected.strip())
        logger.info(
            "Agent answer: %s | Expected: %s | Reward: %s",
            action.content, self.problem.expected, reward
        )
        return [action], reward, True, False
