from __future__ import annotations
from pathlib import Path
import pytest
from agentforge.config import ChallengeConfig, load_config, is_better, best_initial_score


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


class TestIsBetter:
    def test_maximize(self):
        assert is_better(0.9, 0.8, "maximize") is True
        assert is_better(0.8, 0.9, "maximize") is False
        assert is_better(0.9, 0.9, "maximize") is False

    def test_minimize(self):
        assert is_better(1.5, 2.0, "minimize") is True
        assert is_better(2.0, 1.5, "minimize") is False
        assert is_better(1.5, 1.5, "minimize") is False


class TestBestInitialScore:
    def test_maximize(self):
        assert best_initial_score("maximize") == 0.0

    def test_minimize(self):
        assert best_initial_score("minimize") == float("inf")
