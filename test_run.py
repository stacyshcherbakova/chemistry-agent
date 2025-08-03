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
