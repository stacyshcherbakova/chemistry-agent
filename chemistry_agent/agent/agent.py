from aviary.core import  Message, Tool, ToolRequestMessage
from ldp.agent import Agent
from ldp.graph import LLMCallOp
from chemistry_agent.helper.tools import *
from typing import Any, Tuple, List
from pydantic import BaseModel, Field
from ldp.graph import LLMCallOp, OpResult, compute_graph
import logging
logger = logging.getLogger(__name__)

# --- Step 2: Guided Agent ---
# can decide what tools to use at each step without using tool selector
# uses compute graph

class GuidedAgentState(BaseModel):
    """
    Store the agent’s message history and available tools.

    This state object accumulates every Message the agent has seen or produced,
    and holds the list of Tool instances the agent may call.

    Args:
        messages (List[Message]): The sequence of messages so far.
        tools (List[Tool]): The tools the agent can choose from.

    Returns:
        GuidedAgentState: A new state with appended messages.
    """
    
    messages: List[Message]
    tools:    List[Tool]

    def get_next_state(self, obs: List[Message]) -> "GuidedAgentState":
        """
        Append incoming observations to the message history.

        Args:
            obs (List[Message]): New messages or tool outputs to record.

        Returns:
            GuidedAgentState: Updated state with these observations added.
        """
        logger.info("Observations received: %r", obs)
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
        logger.debug("New message history: %r", new_msgs)
        return GuidedAgentState(messages=new_msgs, tools=self.tools)

class GuidedAgent(BaseModel, Agent[GuidedAgentState]):
    """
    An LDP‐style agent that uses a compute graph to select and invoke tools.

    Combines a system‐level guideline message, a low‐temperature LLM config,
    and the compute_graph decorator to orchestrate stepwise reasoning.

    Attributes:
        llm_model (dict): Model name and sampling parameters.
        guidelines_msg (Message): System prompt guiding tool usage.
    """
    
    llm_model: dict[str, Any] = Field(
        default={
            "name": "gpt-4o",
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
        logger.info("Initializing agent state with %d tools", len(tools))
        return GuidedAgentState(messages=[self.guidelines_msg], tools=tools)

    @compute_graph()
    async def get_asv(self, agent_state: GuidedAgentState, obs: List[Message]) -> Tuple[
          OpResult[ToolRequestMessage], 
          GuidedAgentState,
          float                          
        ]:
        logger.debug("get_asv called with state: %r and obs: %r", agent_state, obs)
        
        # 1) incorporate the new environment observation
        next_state = agent_state.get_next_state(obs)
        logger.debug("Calling LLM to select tool")

        # 2) call the LLM to pick exactly one tool + args
        llm_result: OpResult[ToolRequestMessage] = await self._llm_call_op(
            self.llm_model,
            [self.guidelines_msg, *next_state.messages],
            next_state.tools
        )

        logger.info("LLM chose tool: %s", llm_result)
        # 3) append the tool call into the agent’s message history
        next_state.messages.append(llm_result.value)

        # 4) return the op result
        return llm_result, next_state, 0.0