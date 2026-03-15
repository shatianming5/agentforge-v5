from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentforge.scorer import Scorer
from agentforge.experiment import Experiment
from agentforge.agent import Strategy
from agentforge.config import ChallengeConfig


def _make_experiment(workdir):
    return Experiment(
        index=0,
        strategy=Strategy("test", "branch", 0.8, 40, 45, 64, False, "opt", "low"),
        workdir=workdir,
        log_path=workdir / "exp-0.log",
        env={"CUDA_VISIBLE_DEVICES": "0"},
        train_command=["python", "train.py"],
    )


def _make_config():
    return ChallengeConfig(
        "Test", "Desc", "accuracy", 0.95, "maximize",
        "pytest tests/smoke/", "pytest tests/",
        "python benchmark.py --output results/benchmark.json",
        ["src/"], ["tests/"],
    )


class TestScorer:
    def test_score_success(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "benchmark.json").write_text(json.dumps({"accuracy": 0.85}))
        exp = _make_experiment(tmp_path)
        config = _make_config()
        with patch("agentforge.scorer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            score = Scorer.score(exp, config, returncode=0)
        assert score == 0.85

    def test_score_failed_process(self, tmp_path):
        exp = _make_experiment(tmp_path)
        config = _make_config()
        score = Scorer.score(exp, config, returncode=1)
        assert score == 0.0

    def test_score_missing_metric(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "benchmark.json").write_text(json.dumps({"loss": 0.1}))
        exp = _make_experiment(tmp_path)
        config = _make_config()
        with patch("agentforge.scorer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            score = Scorer.score(exp, config, returncode=0)
        assert score == 0.0
