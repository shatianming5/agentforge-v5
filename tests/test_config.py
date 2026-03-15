from __future__ import annotations
from pathlib import Path
import pytest
from agentforge.config import ChallengeConfig, load_config


class TestChallengeConfig:
    def test_load_valid(self, sample_config_path: Path):
        config = load_config(sample_config_path)
        assert config.challenge_name == "Test Challenge"
        assert config.target_metric == "accuracy"
        assert config.target_value == 0.95
        assert config.target_direction == "maximize"
        assert "tests/" in config.read_only

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_writable_and_readonly(self, sample_config_path: Path):
        config = load_config(sample_config_path)
        assert "src/" in config.writable
        assert "tests/" in config.read_only
