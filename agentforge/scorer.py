from __future__ import annotations
import json
import subprocess
from pathlib import Path

from agentforge.config import ChallengeConfig
from agentforge.experiment import Experiment


class Scorer:
    @staticmethod
    def score(exp: Experiment, config: ChallengeConfig, returncode: int) -> float:
        if returncode != 0:
            return 0.0
        try:
            subprocess.run(
                config.test_benchmark.split(),
                cwd=str(exp.workdir),
                env=exp.env,
                capture_output=True,
                timeout=600,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return 0.0
        return Scorer._read_score(exp.workdir, config.target_metric)

    @staticmethod
    def _read_score(workdir: Path, metric: str) -> float:
        results_path = workdir / "results" / "benchmark.json"
        if not results_path.exists():
            return 0.0
        try:
            with open(results_path) as f:
                data = json.load(f)
            return float(data.get(metric, 0.0))
        except (json.JSONDecodeError, ValueError, TypeError):
            return 0.0
