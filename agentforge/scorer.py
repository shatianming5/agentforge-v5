from __future__ import annotations
import json
import shlex
import subprocess
from pathlib import Path
from statistics import median

from agentforge.config import ChallengeConfig
from agentforge.experiment import Experiment


class Scorer:
    @staticmethod
    def score(exp: Experiment, config: ChallengeConfig, returncode: int,
              N: int = 1) -> float:
        if returncode != 0:
            return 0.0
        # Run test suite first, check for regressions
        if not Scorer._run_tests(exp, config):
            return 0.0
        # For N=1: run benchmark 3 times with different seeds, take median
        if N == 1:
            return Scorer._score_multi_seed(exp, config)
        return Scorer._run_benchmark_once(exp, config)

    @staticmethod
    def _run_tests(exp: Experiment, config: ChallengeConfig) -> bool:
        """Run full test suite. Return True if all tests pass."""
        try:
            subprocess.run(
                shlex.split(config.test_full),
                cwd=str(exp.workdir), env=exp.env,
                capture_output=True, timeout=600, check=True,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _score_multi_seed(exp: Experiment, config: ChallengeConfig) -> float:
        """Run benchmark 3 times with seeds [42, 123, 456], return median score."""
        scores = []
        for seed in [42, 123, 456]:
            env = {**exp.env, "PYTHONHASHSEED": str(seed)}
            s = Scorer._run_benchmark(exp.workdir, config, env)
            if s > 0:
                scores.append(s)
        return median(scores) if scores else 0.0

    @staticmethod
    def _run_benchmark_once(exp: Experiment, config: ChallengeConfig) -> float:
        return Scorer._run_benchmark(exp.workdir, config, exp.env)

    @staticmethod
    def _run_benchmark(workdir: Path, config: ChallengeConfig, env: dict) -> float:
        try:
            subprocess.run(
                shlex.split(config.test_benchmark),
                cwd=str(workdir), env=env,
                capture_output=True, timeout=600, check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return 0.0
        return Scorer._read_score(workdir, config.target_metric)

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
