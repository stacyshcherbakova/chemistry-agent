from aviary.core import Environment, Message, Tool, ToolRequestMessage,ToolResponseMessage
from helper.tools import *


# --- Step 1: Science (more like chemistry) Environment -> ---
class ChemistryEnv(Environment[ProblemState]):
    def __init__(self, problem: ProblemState):
          super().__init__()
          self.problem = problem
          self._oracle = None
        
    async def reset(self):
        self.tools = [Tool.from_function(fun) 
                      for _, fun in TOOLS.items()]
        print(f"[self.problem]: {self.problem}")
        prompt = self.problem.prompt or build_prompt(self.problem)
        print(f"[RESET] prompt: {prompt}")
        return [Message(content=prompt)], self.tools

    async def step(self, action):
        # 1) If it’s a tool request, run it and return *native* ToolResponseMessage(s)
        if isinstance(action, ToolRequestMessage) and action.tool_calls:
            raw = await self.exec_tool_calls(action)
            # raw may be one ToolResponseMessage or a list of them
            if isinstance(raw, ToolResponseMessage):
                obs = [raw]
            elif isinstance(raw, list) and all(isinstance(r, ToolResponseMessage) for r in raw):
                obs = raw
            else:
                # (shouldn’t happen—just in case)
                obs = [Message(content=str(raw))]
            print(f"[DEBUG]: {obs}")
            return obs, 0.0, False, False
        
        # 2) Otherwise `action` is the final assistant Message—score and finish
        assert isinstance(action, Message) and action.content is not None
        reward = float(action.content.strip() == self.problem.expected.strip())
        return [action], reward, True, False
