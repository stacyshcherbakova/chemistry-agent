# Simple Chemisry Agent 

## About
A simple chemistry agent that can answer simple GCSE level questions

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

### In a Python script
```python
from chemistry_agent.run_agent.run_agent import ProblemState, solve_problem
import asyncio

if __name__ == "__main__":
    # Example problem
    problem = ProblemState(
        task="molar_mass",
        params={"formula": "H2O"},
        expected="18.02 g/mol",
    )
    asyncio.run(solve_problem(problem))
```

### In a Jupyter notebook
```python
from chemistry_agent.run_agent.run_agent import ProblemState, solve_problem

problem = ProblemState(
    task="molar_mass",
    params={"formula": "H2O"},
    expected="18.02 g/mol",
)

await solve_problem(problem)
```

### Example problems
Example problems can be found under helper/problems. The agent does not fully answer more complicated questions, and there is some inconsistency in comparing agent's and expected answer. Please cut me some slack as this is work in progress...