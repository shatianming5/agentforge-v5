# tests/test_orchestrator_pipeline.py
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentforge.agent import AgentSession, StrategySpec


def test_agent_session_develop_specs():
    """AgentSession.develop_specs() should return StrategySpec list."""
    mock_config = MagicMock()
    mock_config.challenge_name = "Test"
    mock_config.challenge_description = "Test desc"
    mock_config.target_metric = "accuracy"
    mock_config.target_direction = "maximize"
    mock_config.target_value = 0.95
    mock_config.read_only = ["tests/"]

    mock_state = MagicMock()
    mock_state.N = 2
    mock_state.hardware = MagicMock()
    mock_state.hardware.device = "cpu"
    mock_state.hardware.num_gpus = 0
    mock_state.hardware.gpu_model = ""
    mock_state.hardware.cpu_cores = 4
    mock_state.hardware.ram_gb = 16
    mock_state.hardware.disk_free_gb = 100
    mock_state.best = MagicMock(score=0.5, round=1, experiment="exp-0")
    mock_state.score_trajectory = [0.5]
    mock_state.current_round = 1
    mock_state.rounds = []
    mock_state.strategies_tried = []
    mock_state.hints_pending = []

    session = AgentSession(mock_config, mock_state, Path("/tmp/test"))

    raw_output = (
        'AGENTFORGE_SUMMARY_BEGIN\n'
        '[{"name": "cosine_lr", "description": "d", "approach": "a", '
        '"category": "optim", "risk": "low", "estimated_train_command": "echo"}]\n'
        'AGENTFORGE_SUMMARY_END'
    )

    with patch("agentforge.agent.CodexCLI.run", return_value=raw_output):
        specs = session.develop_specs()

    assert len(specs) == 1
    assert isinstance(specs[0], StrategySpec)
    assert specs[0].name == "cosine_lr"
