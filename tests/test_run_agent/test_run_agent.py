import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from chemistry_agent.run_agent.run_agent import solve_problem, ProblemState

class TestSolveProblemSmokeTests:
    """Basic tests to ensure the function doesn't crash"""
    
    @pytest.mark.asyncio
    @patch('chemistry_agent.run_agent.run_agent.GuidedAgent')
    @patch('chemistry_agent.run_agent.run_agent.ChemistryEnv')
    async def test_solve_problem_runs_without_crashing(self, mock_env_class, mock_agent_class):
        """Test that solve_problem executes without throwing exceptions"""
        # Setup mocks
        mock_env = AsyncMock()
        mock_agent = AsyncMock()
        mock_env_class.return_value = mock_env
        mock_agent_class.return_value = mock_agent
        
        # Mock the flow
        mock_env.reset.return_value = ("obs", ["tool1", "tool2"])
        mock_agent.init_state.return_value = "agent_state"
        
        # Mock agent response
        mock_op_result = MagicMock()
        mock_op_result.value = "calculate_molar_mass H2O"
        mock_agent.get_asv.return_value = (mock_op_result, "new_state", None)
        
        # Mock environment step
        mock_obs = MagicMock()
        mock_obs.content = "18.02 g/mol"
        mock_env.step.return_value = ([mock_obs], None, False, {})
        
        # Test problem
        problem = ProblemState(
            task="molar_mass",
            params={"formula": "H2O"},
            expected="18.02 g/mol"
        )
        
        # Should not raise an exception
        await solve_problem(problem)
        
        # Verify the basic flow was called
        mock_env_class.assert_called_once_with(problem)
        mock_env.reset.assert_called_once()
        mock_agent.init_state.assert_called_once()

    @pytest.mark.asyncio
    @patch('chemistry_agent.run_agent.run_agent.GuidedAgent')
    @patch('chemistry_agent.run_agent.run_agent.ChemistryEnv')
    async def test_solve_problem_early_success(self, mock_env_class, mock_agent_class):
        """Test when the problem is solved on first try"""
        # Setup mocks for immediate success
        mock_env = AsyncMock()
        mock_agent = AsyncMock()
        mock_env_class.return_value = mock_env
        mock_agent_class.return_value = mock_agent
        
        mock_env.reset.return_value = ("obs", ["tools"])
        mock_agent.init_state.return_value = "state"
        
        mock_op_result = MagicMock()
        mock_op_result.value = "action"
        mock_agent.get_asv.return_value = (mock_op_result, "state", None)
        
        # Return the expected answer immediately
        mock_obs = MagicMock()
        mock_obs.content = "42.5 g/mol"
        mock_env.step.return_value = ([mock_obs], None, False, {})
        
        problem = ProblemState(
            params={"formula": "test"},
            expected="42.5 g/mol"
        )
        
        await solve_problem(problem)
        
        # Should only call step once since it succeeds immediately
        assert mock_env.step.call_count == 1

    @pytest.mark.asyncio
    @patch('chemistry_agent.run_agent.run_agent.GuidedAgent')
    @patch('chemistry_agent.run_agent.run_agent.ChemistryEnv')
    async def test_solve_problem_max_calls_reached(self, mock_env_class, mock_agent_class):
        """Test when max calls are reached without success"""
        # Setup mocks that never succeed
        mock_env = AsyncMock()
        mock_agent = AsyncMock()
        mock_env_class.return_value = mock_env
        mock_agent_class.return_value = mock_agent
        
        mock_env.reset.return_value = ("obs", ["tools"])
        mock_agent.init_state.return_value = "state"
        
        mock_op_result = MagicMock()
        mock_op_result.value = "action"
        mock_agent.get_asv.return_value = (mock_op_result, "state", None)
        
        # Always return wrong answer
        mock_obs = MagicMock()
        mock_obs.content = "wrong answer"
        mock_env.step.return_value = ([mock_obs], None, False, {})
        
        problem = ProblemState(
            params={"formula": "test"},
            expected="correct answer"
        )
        
        await solve_problem(problem)
        
        # Should call step exactly MAX_CALLS times
        assert mock_env.step.call_count == 5