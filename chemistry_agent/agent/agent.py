from aviary.core import  Message, Tool, ToolRequestMessage
from ldp.agent import Agent
from ldp.graph import LLMCallOp
from helper.tools import *
from typing import Any, Tuple, List
from pydantic import BaseModel, Field
from ldp.graph import LLMCallOp, OpResult, compute_graph

# --- Step 2: Guided Agent ---
# can decide what tools to use at each step without using tool selector
# uses compute graph

class GuidedAgentState(BaseModel):
    messages: List[Message]
    tools:    List[Tool]

    def get_next_state(self, obs: List[Message]) -> "GuidedAgentState":
        new_msgs = list(self.messages)

        for item in obs:
            if isinstance(item, Message):
                new_msgs.append(item)

            elif hasattr(item, "content"):
                new_msgs.append(Message(content=item.content))

            elif isinstance(item, dict):
                new_msgs.append(Message(**item))

            else:
                continue
        print(f"[DEBUG]: {new_msgs}")
        return GuidedAgentState(messages=new_msgs, tools=self.tools)

class GuidedAgent(BaseModel, Agent[GuidedAgentState]):
    llm_model: dict[str, Any] = Field(
        default={
            "name": "gpt-4o-2024-08-06",
            "temperature": 0.1,
        }
    )
    guidelines_msg: Message = Field(
        default=Message(
            role="system",
            content=(
                "You are a scientific assistant.  You **must** use the provided tools for *every* calculation.\n"
                "Step by step:\n"
                "  1) To compute a mixture’s concentration, call **mix_concentration** with JSON args.\n"
                "  2) Observe the tool’s output (e.g. “0.175 M”).\n"
                "  3) To compute dilution volume, call **dilution** with JSON args using that exact output.\n"
                "  4) Observe and return that final tool output as your answer."
            )
        ),
        description="System‐level instructions for the LLM.",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._llm_call_op = LLMCallOp()

    async def init_state(self, tools: List[Tool]) -> GuidedAgentState:
        return GuidedAgentState(messages=[self.guidelines_msg], tools=tools)

    @compute_graph()
    async def get_asv(self, agent_state: GuidedAgentState, obs: List[Message]) -> Tuple[
          OpResult[ToolRequestMessage], 
          GuidedAgentState,
          float                          
        ]:
        # 1) incorporate the new environment observation
        next_state = agent_state.get_next_state(obs)

        # 2) call the LLM to pick exactly one tool + args
        llm_result: OpResult[ToolRequestMessage] = await self._llm_call_op(
            self.llm_model,
            [self.guidelines_msg, *next_state.messages],
            next_state.tools
        )

        # 3) append the tool call into the agent’s message history
        next_state.messages.append(llm_result.value)

        # 4) return the op result
        return llm_result, next_state, 0.0