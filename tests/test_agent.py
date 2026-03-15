from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentforge.agent import PromptBuilder, OutputParser, CodexCLI, Strategy
from agentforge.config import ChallengeConfig
from agentforge.state import (
    HardwareInfo, BestResult, Budget, SessionState, StrategyRecord,
)
import pytest


def _make_state(**overrides):
    defaults = dict(
        version="5.0", session_id="af-test", repo_url="/tmp/repo",
        status="running",
        hardware=HardwareInfo("cuda", "A100 80GB", 8, 64, 256, 500),
        N=8, gpus_per_experiment=1,
        best=BestResult(82.1, 5, "B1a", "abc123", "best.pt"),
        current_round=7,
        score_trajectory=[52, 67, 74, 77, 82, 81, 82.1],
        rounds=[], strategies_tried=[], hints_pending=[],
        budget=Budget(6, 25, 12.7, 200, 8.50),
        env_lockfile_hash="",
    )
    defaults.update(overrides)
    return SessionState(**defaults)


def _make_config():
    return ChallengeConfig(
        challenge_name="CIFAR-10", challenge_description="Maximize accuracy",
        target_metric="accuracy", target_value=0.95, target_direction="maximize",
        test_smoke="pytest tests/smoke/", test_full="pytest tests/",
        test_benchmark="python benchmark.py",
        writable=["src/"], read_only=["tests/"],
    )


class TestPromptBuilder:
    def test_contains_challenge(self):
        prompt = PromptBuilder.build(_make_config(), _make_state())
        assert "CIFAR-10" in prompt and "accuracy" in prompt

    def test_contains_hardware(self):
        prompt = PromptBuilder.build(_make_config(), _make_state())
        assert "A100 80GB" in prompt

    def test_contains_best_score(self):
        prompt = PromptBuilder.build(_make_config(), _make_state())
        assert "82.1" in prompt

    def test_contains_trajectory(self):
        prompt = PromptBuilder.build(_make_config(), _make_state())
        assert "52" in prompt

    def test_includes_hints(self):
        state = _make_state(hints_pending=["try cosine annealing"])
        prompt = PromptBuilder.build(_make_config(), state)
        assert "cosine annealing" in prompt

    def test_includes_taboo(self):
        state = _make_state(strategies_tried=[StrategyRecord("sgd", 1, 52.0, "surpassed")])
        prompt = PromptBuilder.build(_make_config(), state)
        assert "sgd" in prompt

    def test_includes_read_only(self):
        prompt = PromptBuilder.build(_make_config(), _make_state())
        assert "tests/" in prompt


class TestOutputParser:
    def test_parse_valid(self):
        raw = (
            "Some text\nAGENTFORGE_SUMMARY_BEGIN\n"
            + json.dumps([{
                "name": "lr_warmup", "branch": "agentforge/iter-7/exp-0",
                "confidence": 0.8, "measured_vram_gb": 41.2,
                "measured_epoch_seconds": 45.0, "batch_size": 64,
                "resume_checkpoint": False, "category": "optimization", "risk": "low",
            }])
            + "\nAGENTFORGE_SUMMARY_END\nMore text"
        )
        strategies = OutputParser.parse(raw)
        assert len(strategies) == 1
        assert strategies[0].name == "lr_warmup"
        assert strategies[0].measured_vram_gb == 41.2

    def test_parse_no_marker(self):
        with pytest.raises(ValueError, match="No summary found"):
            OutputParser.parse("random output")

    def test_parse_invalid_json(self):
        raw = "AGENTFORGE_SUMMARY_BEGIN\n{bad json\nAGENTFORGE_SUMMARY_END\n"
        with pytest.raises(ValueError, match="Invalid JSON"):
            OutputParser.parse(raw)


class TestStrategy:
    def test_create(self):
        s = Strategy(
            name="test", branch="b", confidence=0.9,
            measured_vram_gb=40.0, measured_epoch_seconds=30.0,
            batch_size=32, resume_checkpoint=False,
            category="optimization", risk="low",
        )
        assert s.name == "test"
        assert s.confidence == 0.9


class TestCodexCLI:
    def test_run_success(self):
        with patch("agentforge.agent.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="output", returncode=0)
            result = CodexCLI.run("prompt", Path("/tmp"), 60, {})
        assert result == "output"

    def test_run_timeout(self):
        with patch("agentforge.agent.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("codex", 60)
            with pytest.raises(subprocess.TimeoutExpired):
                CodexCLI.run("prompt", Path("/tmp"), 60, {})
