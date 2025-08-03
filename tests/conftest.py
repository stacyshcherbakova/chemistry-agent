# tests/conftest.py
import pytest
from chemistry_agent.problem_state.problem_state import ProblemState

@pytest.fixture
def sample_problem():
    """Basic sample problem for general testing"""
    return ProblemState(
        task="molar_mass",
        params={"formula": "H2O"},
        expected="18.02 g/mol"
    )

@pytest.fixture
def problem_with_prompt():
    """Problem with all fields filled"""
    return ProblemState(
        prompt="Calculate the molar mass of glucose",
        task="molar_mass",
        params={"formula": "C6H12O6"},
        expected="180.16 g/mol"
    )

@pytest.fixture
def minimal_problem():
    """Problem with only required fields"""
    return ProblemState(
        params={"value": "test"},
        expected="result"
    )

@pytest.fixture
def various_task_problems():
    """Collection of problems with different task types"""
    return {
        "molar_mass": ProblemState(
            task="molar_mass",
            params={"formula": "NaCl"},
            expected="58.44 g/mol"
        ),
        "dilution": ProblemState(
            task="dilution", 
            params={"initial_conc": "2.0 M", "final_conc": "0.5 M", "final_vol": "100 mL"},
            expected="25 mL"
        ),
        "pH": ProblemState(
            task="pH_from_H",
            params={"H_concentration": "1e-3"},
            expected="3.0"
        )
    }

@pytest.fixture
def invalid_problem_data():
    """Data that should fail validation"""
    return {
        "invalid_task": {
            "task": "nonexistent_task",
            "params": {"test": "value"},
            "expected": "result"
        },
        "missing_params": {
            "task": "molar_mass",
            "expected": "result"
            # params missing
        },
        "missing_expected": {
            "task": "molar_mass", 
            "params": {"test": "value"}
            # expected missing
        }
    }