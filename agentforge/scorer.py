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
            print(f"[Scorer] exp-{exp.index}: returncode={returncode}, score=0.0")
            return 0.0
        # Run test suite first, check for regressions
        if not Scorer._run_tests(exp, config):
            print(f"[Scorer] exp-{exp.index}: test suite failed, score=0.0")
            return 0.0
        # For N=1: run benchmark 3 times with different seeds, take median
        if N == 1:
            s = Scorer._score_multi_seed(exp, config)
        else:
            s = Scorer._run_benchmark_once(exp, config)
        print(f"[Scorer] exp-{exp.index}: score={s}")
        return s

    @staticmethod
    def _run_tests(exp: Experiment, config: ChallengeConfig) -> bool:
        """Run full test suite. Return True if all tests pass."""
        from agentforge.stream import stream_run
        try:
            stream_run(
                shlex.split(config.test_full),
                cwd=exp.workdir, env=exp.env,
                timeout=3600, prefix=f"Test exp-{exp.index}", check=True,
                quiet=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"[Scorer] test failed: {(e.output or '')[:300]}")
            return False
        except subprocess.TimeoutExpired:
            print("[Scorer] test timed out")
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
        from agentforge.stream import stream_run
        try:
            stream_run(
                shlex.split(config.test_benchmark),
                cwd=workdir, env=env,
                timeout=3600, prefix="Benchmark", check=True,
                quiet=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"[Scorer] benchmark failed: {(e.output or '')[:300]}")
            return 0.0
        except subprocess.TimeoutExpired:
            print("[Scorer] benchmark timed out")
            return 0.0
        return Scorer._read_score(workdir, config.target_metric)

    @staticmethod
    def _read_score(workdir: Path, metric: str) -> float:
        results_path = workdir / "results" / "benchmark.json"
        if not results_path.exists():
            print(f"[Scorer] results file not found: {results_path}")
            return 0.0
        try:
            with open(results_path) as f:
                data = json.load(f)
            score = float(data.get(metric, 0.0))
            if score == 0.0:
                print(f"[Scorer] metric '{metric}' not found in {data}")
            return score
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"[Scorer] error reading results: {e}")
            return 0.0
