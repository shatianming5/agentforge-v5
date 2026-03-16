# tests/test_implement_strategy.py
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from agentforge.agent import CodexCLI, StrategySpec, Strategy


def test_implement_strategy_with_preseeded_output(tmp_path):
    """Pre-seeded agent_output.json should be consumed and return a Strategy."""
    af_dir = tmp_path / ".agentforge"
    af_dir.mkdir()
    strategy_data = [{
        "name": "cosine_lr",
        "branch": "agentforge/iter-1/exp-0",
        "confidence": 0.8,
        "measured_vram_gb": 2.5,
        "measured_epoch_seconds": 30.0,
        "batch_size": 32,
        "resume_checkpoint": False,
        "category": "optim",
        "risk": "low",
        "train_command": "python train.py",
    }]
    (af_dir / "agent_output.json").write_text(json.dumps(strategy_data))

    spec = StrategySpec(
        name="cosine_lr",
        description="Use cosine annealing",
        approach="Replace StepLR with CosineAnnealingLR",
        category="optim",
        risk="low",
        estimated_train_command="python train.py",
    )
    result = CodexCLI.implement_strategy(
        spec=spec, index=0, round_num=1,
        cwd=tmp_path, config_context="test challenge", timeout=60,
    )
    assert isinstance(result, Strategy)
    assert result.name == "cosine_lr"
    assert result.branch == "agentforge/iter-1/exp-0"


def test_implement_strategy_builds_correct_prompt():
    """Prompt should contain strategy spec details."""
    spec = StrategySpec(
        name="wider_backbone",
        description="Increase model width",
        approach="Double hidden_dim in model.py",
        category="arch",
        risk="high",
        estimated_train_command="python train.py --width=512",
    )
    prompt = CodexCLI._build_implement_prompt(
        spec, index=2, round_num=3,
        config_context="Maximize accuracy on CIFAR-10",
    )
    assert "wider_backbone" in prompt
    assert "Double hidden_dim" in prompt
    assert "agentforge/iter-3/exp-2" in prompt
    assert "CIFAR-10" in prompt
