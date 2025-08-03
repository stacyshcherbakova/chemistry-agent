# Simple Chemisry Agent 

## About
A simple chemistry agent that can answer simple GCSE level questions (not perfectly, perphaps...)
Developed using FutureHouse [Aviary](https://futurehouse.gitbook.io/futurehouse-cookbook/aviary) and [LDP](https://futurehouse.gitbook.io/futurehouse-cookbook/ldp-language-decision-processes) frameworks.
## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/stacyshcherbakova/chemistry-agent.git
   cd chemistry-agent
   ```

2. **Create a virtual environment**

   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment**

   **macOS/Linux (bash/zsh):**
     ```bash
     source .venv/bin/activate
     ```

4. **Upgrade pip and install dependencies**

   ```bash
   pip install --upgrade pip
   pip install .
   ```

## Detailed Implementation 

### Chemistry Environemnt 
The `ChemistryEnv` class orchestrates the problem-solving flow:

- **Initialization**: Accepts a `ProblemState`, sets up the environment state.
- **reset()**: Converts entries in `helper.tools` into `Tool` instances, constructs the prompt (using `build_prompt()` if none is provided), and returns the initial list of `[Message]` along with available tools.
- **step()**: Executes tool request via `exec_tool_calls()`, wraps results into `ToolResponseMessage`(s), and emits observations for further processing. After a responce is recieved in a form of `Message`, calculates a reward by comparing `action.content` to `problem.expected` and terminates the episode when complete.

### Guided Agent
The `GuidedAgent` uses a declarative computation graph (`@compute_graph`) to manage LLM-driven tool selection:

- **State Model**: `GuidedAgentState` (a `pydantic.BaseModel`) tracks `messages` and `tools`, offering `get_next_state()` to append new observations (so called "unsophisticated memory"). A `system`-role `Message` enforces a strict sequence of tool calls (e.g., call `mix_concentration`, inspect output, then call `dilution`).
- **get_asv()** - incorporate the new environment observation with `get_next_state()`, calls the LLM to pick one tool, adds the tool call into the agent’s message history and returns the results as a OpResult object.
- **LLM Integration**: An `LLMCallOp` instance generates `ToolRequestMessage` proposals, which are appended to the agent’s history and executed in subsequent steps.


## How to Run a Single Problem

Make sure to activate your virtual environment before running the script, or select it as your kernel in the notebook!

### In a Python script (can be found in `test_run.py`)
```python
import os
from chemistry_agent.problem_state.problem_state import ProblemState
from chemistry_agent.run_agent.run_agent import solve_problem
from chemistry_agent.agent.agent import GuidedAgent
from chemistry_agent.env.env import ChemistryEnv
import asyncio
os.environ["OPENAI_API_KEY"] = ""

agent = GuidedAgent()

problem = ProblemState(
   task="molar_mass",
   params={"formula": "H2O"},
   expected="18.02 g/mol",
)
env = ChemistryEnv(problem)
asyncio.run(solve_problem(problem=problem, env=env, agent=agent))
```

### In a Jupyter notebook
```python
import os
from chemistry_agent.problem_state.problem_state import ProblemState
from chemistry_agent.run_agent.run_agent import solve_problem
from chemistry_agent.agent.agent import GuidedAgent
from chemistry_agent.env.env import ChemistryEnv
os.environ["OPENAI_API_KEY"] = ""

agent = GuidedAgent()

problem = ProblemState(
    task="molar_mass",
    params={"formula": "H2O"},
    expected="18.02 g/mol",
)

env = ChemistryEnv(problem)
await solve_problem(problem=problem, env=env, agent=agent)
```

### Example problems
Example problems can be found under helper/problems. The agent does not fully answer more complicated questions, and there is some inconsistency in comparing agent's and expected answer (UPD: I have run created a `run_debug.py` to debug the expected output vs tools output). Please cut me some slack as this is work in progress...

## Personal reflection on this Mini-Project

*Note: please do not take this very seriosly. I try to haave fun when I work and this personal reflection was mostly to document the rollecoster of emotions I had in the one week doing this while full time working and moving houses... I would like to do this for evey personal project I have in the future.*

## Personal Reflection on This Mini-Project

> *Note: Please don't take this too seriously—this is mostly to document the rollercoaster of emotions I experienced during one week of this project, while working full-time and moving houses. I’d like to do this for every personal project in the future.*

---

### Opening Thoughts

I’ve worked with simple machine learning models and more sophisticated neural nets like VAEs and GANs (using transfer learning and more), but I’d never created an agent or used the OpenAI API for development before. I built my project by referring to:

- Aviary [cookbook](https://futurehouse.gitbook.io/futurehouse-cookbook/aviary) and [tutorial](https://github.com/Future-House/aviary/tree/main/tutorials)
- LDP (Language Decision Processes) [cookbook](https://futurehouse.gitbook.io/futurehouse-cookbook/ldp-language-decision-processes) and [tutorial](https://github.com/Future-House/ldp/blob/main/tutorials/creating_a_language_agent.ipynb)

The frameworks had limited documentation and few real-world examples, which made it a challenge to understand all the available tools. That was my starting point.

---

### Game Plan

1. **Define a deterministic problem.** After playing around with generating arbitrary text (names for people, fruit, etc.), I realised I needed a clear validation mechanism. Deterministic outputs would make testing and debugging simpler.
2. **Build a unit-conversion tool.** This worked, but general unit conversion proved too broad, there are endless units and edge cases...
3. **Focus on chemistry.** I remembered that Future House offers a chemistry model called **Phoenix**, and GCSE-level chemistry questions (molarity, density, etc.) map neatly to well-defined calculations.
4. **Implement tool selection.** I created a chemistry agent that chose the right tool for simple questions. It struggled with multi-step problems because it lacked context memory.
5. **Add memory with Agent State.** Following the LDP tutorial, I used `computer_graph` and an **Agent State** store to accumulate previous messages and reasoning steps. This was the hardest part to debug.. running the agent repeatedly ate into my budget, but it was essential for chaining tool calls.
6. **Expand calculation capabilities.** The agent still stumbled on basic arithmetic when solving multi-step questions. To fix this, I added a generic expression-evaluation tool to the chemistry environment.

By the end, the agent could:
- Correctly select chemistry-specific tools
- Maintain a reasoning chain across calls
- Evaluate arbitrary calculations as needed

Which is pretty cool!!!!

---

### Cost

I set a budget of **£10** (≈$13 USD), since I’d never used OpenAI API calls before. In total, I spent about **≈£2.14** ($2.84 USD).

![Cost breakdown chart](assets/costs.png)

---

### What Could Be Improved

- **Structured reasoning steps.** Adopt a more formal plan-and-execute loop to guide the agent’s logic. Not sure how yet..
- **Richer toolset.** Provide domain-specific tools for thermodynamics, equilibrium calculations, etc. to broaden the agent’s capabilities!
- **Change logging.** 
   - **Concern:**  
      The current logs are “ugly”. Standard print statements provide more useful information.
   - **Next steps:**    
      Create a more robust logging experience!
- **Enhanced rewards (for intermediate steps).** Coudl provide useful in guiding the model if the step taken was correct.
- **“Exceeded Callbacks” bug.**  
  - **Error:**  
    ```  
    LiteLLM WARNING: Cannot add callback – would exceed MAX_CALLBACKS limit of 30 (current: 30)  
    ```  
  - **Next steps:**  
    Pool or deregister callbacks after use, or raise the `MAX_CALLBACKS` threshold in config.

