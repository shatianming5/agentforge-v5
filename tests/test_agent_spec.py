# tests/test_agent_spec.py
from agentforge.agent import StrategySpec, PromptBuilder, OutputParser


def test_strategy_spec_fields():
    spec = StrategySpec(
        name="cosine_lr",
        description="Use cosine annealing LR schedule",
        approach="Replace StepLR with CosineAnnealingLR, warmup 5 epochs",
        category="optim",
        risk="low",
        estimated_train_command="python train.py --lr-schedule=cosine",
    )
    assert spec.name == "cosine_lr"
    assert spec.category == "optim"


def test_parse_specs_valid():
    raw = (
        'AGENTFORGE_SUMMARY_BEGIN\n'
        '[{"name": "cosine_lr", "description": "cosine schedule", '
        '"approach": "replace step with cosine", "category": "optim", '
        '"risk": "low", "estimated_train_command": "python train.py"}]\n'
        'AGENTFORGE_SUMMARY_END'
    )
    specs = OutputParser.parse_specs(raw)
    assert len(specs) == 1
    assert isinstance(specs[0], StrategySpec)
    assert specs[0].name == "cosine_lr"


def test_parse_specs_missing_name_skipped():
    raw = (
        'AGENTFORGE_SUMMARY_BEGIN\n'
        '[{"description": "no name field", "approach": "x", '
        '"category": "optim", "risk": "low", "estimated_train_command": ""},'
        '{"name": "valid", "description": "ok", "approach": "y", '
        '"category": "arch", "risk": "high", "estimated_train_command": ""}]\n'
        'AGENTFORGE_SUMMARY_END'
    )
    specs = OutputParser.parse_specs(raw)
    assert len(specs) == 1
    assert specs[0].name == "valid"


def test_parse_specs_empty_raises():
    raw = 'AGENTFORGE_SUMMARY_BEGIN\n[]\nAGENTFORGE_SUMMARY_END'
    try:
        OutputParser.parse_specs(raw)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
