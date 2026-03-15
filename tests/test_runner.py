from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentforge.runner import ParallelRunner
from agentforge.experiment import Experiment
from agentforge.agent import Strategy
from agentforge.config import ChallengeConfig


def _make_experiment(tmp_path, index=0):
    workdir = tmp_path / f"exp-{index}"
    workdir.mkdir(exist_ok=True)
    log_path = tmp_path / "logs" / f"exp-{index}.log"
    log_path.parent.mkdir(exist_ok=True)
    return Experiment(
        index=index,
        strategy=Strategy(f"s{index}", f"b{index}", 0.5, 10, 30, 32, False, "opt", "low"),
        workdir=workdir,
        log_path=log_path,
        env=dict(os.environ),
        train_command=[sys.executable, "-c", "print('done')"],
    )


def _make_config():
    return ChallengeConfig(
        "Test", "Desc", "accuracy", 0.95, "maximize",
        "echo smoke", "echo full", "echo benchmark",
        ["src/"], ["tests/"],
    )


class TestParallelRunner:
    def test_run_single_success(self, tmp_path):
        exp = _make_experiment(tmp_path)
        results_dir = exp.workdir / "results"
        results_dir.mkdir()
        (results_dir / "benchmark.json").write_text(json.dumps({"accuracy": 0.85}))
        runner = ParallelRunner([exp], _make_config(), timeout=30, workdir=tmp_path)
        with patch("agentforge.scorer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            results = runner.run()
        assert len(results) == 1
        assert results[0].status == "ok"

    def test_run_failed_experiment(self, tmp_path):
        exp = _make_experiment(tmp_path)
        exp.train_command = [sys.executable, "-c", "import sys; sys.exit(1)"]
        runner = ParallelRunner([exp], _make_config(), timeout=30, workdir=tmp_path)
        results = runner.run()
        assert len(results) == 1
        assert results[0].status == "error"
        assert results[0].score == 0.0
