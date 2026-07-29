import pytest
from pydantic import ValidationError
from chemistry_agent.problem_state.problem_state import ProblemState

class TestProblemStateCreation:
    
    def test_minimal_valid_creation(self):
        """Test creating ProblemState with minimal required fields"""
        problem = ProblemState(
            params={"formula": "H2O"},
            expected="18.02 g/mol"
        )
        assert problem.prompt is None
        assert problem.task is None
        assert problem.params == {"formula": "H2O"}
        assert problem.expected == "18.02 g/mol"
    
    def test_full_valid_creation(self):
        """Test creating ProblemState with all fields"""
        problem = ProblemState(
            prompt="Calculate the molar mass of water",
            task="molar_mass",
            params={"formula": "H2O"},
            expected="18.02 g/mol"
        )
        assert problem.prompt == "Calculate the molar mass of water"
        assert problem.task == "molar_mass"
        assert problem.params == {"formula": "H2O"}
        assert problem.expected == "18.02 g/mol"
    
    @pytest.mark.parametrize("task", [
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
    ])
    def test_valid_task_types(self, task):
        """Test all valid task types"""
        problem = ProblemState(
            task=task,
            params={"test": "value"},
            expected="result"
        )
        assert problem.task == task
    
    def test_arbitrary_task_string_accepted(self):
        """task is now Optional[str]: arbitrary tool names are accepted"""
        problem = ProblemState(
            task="any_future_tool",
            params={"formula": "H2O"},
            expected="18.02 g/mol"
        )
        assert problem.task == "any_future_tool"

    def test_level_topic_tolerance_defaults(self):
        """New metadata fields default sensibly"""
        problem = ProblemState(
            params={"formula": "H2O"},
            expected="18.02 g/mol"
        )
        assert problem.level == "gcse"
        assert problem.topic == "general"
        assert problem.tolerance == 0.01

    @pytest.mark.parametrize("level", ["gcse", "alevel", "university"])
    def test_valid_levels(self, level):
        """All difficulty tiers are accepted"""
        problem = ProblemState(
            level=level,
            params={},
            expected="x"
        )
        assert problem.level == level

    def test_invalid_level_rejected(self):
        """level remains a strict Literal"""
        with pytest.raises(ValidationError) as exc_info:
            ProblemState(
                level="phd",
                params={},
                expected="x"
            )
        assert "level" in str(exc_info.value)
    
    def test_missing_required_params(self):
        """Test that missing required 'params' field raises error"""
        with pytest.raises(ValidationError) as exc_info:
            ProblemState(
                task="molar_mass",
                expected="18.02 g/mol"
                # params is missing
            )
        
        assert "params" in str(exc_info.value)
    
    def test_missing_required_expected(self):
        """Test that missing required 'expected' field raises error"""
        with pytest.raises(ValidationError) as exc_info:
            ProblemState(
                task="molar_mass",
                params={"formula": "H2O"}
                # expected is missing
            )
        
        assert "expected" in str(exc_info.value)
    
    def test_empty_params_dict(self):
        """Test that empty params dict is allowed"""
        problem = ProblemState(
            params={},  # Empty but present
            expected="result"
        )
        assert problem.params == {}
    
    def test_empty_expected_string(self):
        """Test that empty expected string is allowed"""
        problem = ProblemState(
            params={"test": "value"},
            expected=""  # Empty but present
        )
        assert problem.expected == ""
    
    def test_complex_params_structure(self):
        """Test params with nested structures"""
        complex_params = {
            "formula": "C6H12O6",
            "conditions": {
                "temperature": 25,
                "pressure": 1.0
            },
            "steps": ["step1", "step2", "step3"],
            "coefficients": [2, 1, 2]
        }
        
        problem = ProblemState(
            task="molar_mass",
            params=complex_params,
            expected="180.16 g/mol"
        )
        
        assert problem.params == complex_params
        assert problem.params["conditions"]["temperature"] == 25
        assert problem.params["steps"] == ["step1", "step2", "step3"]
    
    def test_none_values_in_optional_fields(self):
        """Test explicitly setting optional fields to None"""
        problem = ProblemState(
            prompt=None,
            task=None,
            params={"formula": "H2O"},
            expected="18.02 g/mol"
        )
        assert problem.prompt is None
        assert problem.task is None
    
    def test_pydantic_model_behavior(self):
        """Test that ProblemState behaves like a proper Pydantic model"""
        problem = ProblemState(
            task="molar_mass",
            params={"formula": "H2O"},
            expected="18.02 g/mol"
        )
        
        # Test dict conversion
        problem_dict = problem.model_dump()
        assert isinstance(problem_dict, dict)
        assert problem_dict["task"] == "molar_mass"
        
        # Test JSON serialization
        problem_json = problem.model_dump_json()
        assert isinstance(problem_json, str)
        assert "molar_mass" in problem_json
    
    def test_model_reconstruction(self):
        """Test creating ProblemState from dict/JSON"""
        original = ProblemState(
            prompt="Test prompt",
            task="molar_mass", 
            params={"formula": "H2O"},
            expected="18.02 g/mol"
        )
        
        # Convert to dict and back
        problem_dict = original.model_dump()
        reconstructed = ProblemState(**problem_dict)
        
        assert reconstructed.prompt == original.prompt
        assert reconstructed.task == original.task
        assert reconstructed.params == original.params
        assert reconstructed.expected == original.expected
    
    def test_field_types(self):
        """Test that fields have correct types"""
        problem = ProblemState(
            prompt="string prompt",
            task="molar_mass",
            params={"key": "value"},
            expected="string result"
        )
        
        assert isinstance(problem.prompt, str)
        assert isinstance(problem.task, str)
        assert isinstance(problem.params, dict)
        assert isinstance(problem.expected, str)


class TestProblemStateEdgeCases:
    
    def test_very_long_strings(self):
        """Test with very long string values"""
        long_string = "x" * 10000
        
        problem = ProblemState(
            prompt=long_string,
            params={"formula": long_string},
            expected=long_string
        )
        
        assert len(problem.prompt) == 10000
        assert len(problem.expected) == 10000
    
    def test_special_characters_in_strings(self):
        """Test with special characters"""
        special_chars = "!@#$%^&*()[]{}|;':\",./<>?`~"
        
        problem = ProblemState(
            prompt=f"Calculate {special_chars}",
            params={"formula": special_chars},
            expected=special_chars
        )
        
        assert special_chars in problem.prompt
        assert problem.params["formula"] == special_chars
    
    def test_unicode_characters(self):
        """Test with unicode characters"""
        unicode_text = "Δ, Σ, π, α, β, γ, 中文, 🧪"
        
        problem = ProblemState(
            prompt=unicode_text,
            params={"formula": unicode_text},
            expected=unicode_text
        )
        
        assert problem.prompt == unicode_text
        assert problem.expected == unicode_text
