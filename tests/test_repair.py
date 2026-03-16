from __future__ import annotations

import json
from pathlib import Path

from agentforge.repair import SelfRepair
from agentforge.state import StrategyResult


def _make_results(statuses, errors, scores=None):
    if scores is None:
        scores = [0.0] * len(statuses)
    return [
        StrategyResult(f"exp-{i}", f"s{i}", f"b{i}", sc, s, e, 0, 0, 32)
        for i, (s, e, sc) in enumerate(zip(statuses, errors, scores))
    ]


class TestSelfRepair:
    def test_diagnose_environmental(self):
        results = _make_results(["error"]*4, ["OOM"]*4)
        assert SelfRepair.diagnose_all_fail(results) == "environmental"

    def test_diagnose_code_related(self):
        results = _make_results(["error"]*4, ["OOM", "NaN", "Timeout", "OOM"])
        assert SelfRepair.diagnose_all_fail(results) == "code_related"

    def test_all_fail_true(self):
        results = _make_results(["error"]*4, ["OOM"]*4)
        assert SelfRepair.is_all_fail(results) is True

    def test_all_fail_false(self):
        results = _make_results(["ok", "error"], [None, "OOM"])
        assert SelfRepair.is_all_fail(results) is False


class TestScoringFailures:
    def test_has_scoring_failures_true(self):
        results = _make_results(["ok", "ok"], [None, None], [0.0, 0.95])
        assert SelfRepair.has_scoring_failures(results) is True

    def test_has_scoring_failures_false_all_scored(self):
        results = _make_results(["ok", "ok"], [None, None], [0.85, 0.95])
        assert SelfRepair.has_scoring_failures(results) is False

    def test_has_scoring_failures_false_training_error(self):
        results = _make_results(["error", "error"], ["OOM", "OOM"])
        assert SelfRepair.has_scoring_failures(results) is False

    def test_diagnose_benchmark_error(self, tmp_path):
        from agentforge.config import ChallengeConfig

        # Create a workdir with a failing benchmark
        workdir = tmp_path / "exp"
        workdir.mkdir()
        (workdir / "test_suite.py").write_text(
            "import sys; print('All 1 tests passed'); sys.exit(0)"
        )
        (workdir / "benchmark.py").write_text(
            "raise FileNotFoundError('no checkpoint')"
        )
        config = ChallengeConfig(
            challenge_name="test", challenge_description="",
            target_metric="val_loss", target_value=1.0,
            target_direction="minimize",
            test_smoke="python3 test_suite.py",
            test_full="python3 test_suite.py",
            test_benchmark="python3 benchmark.py",
            writable=[], read_only=[],
        )
        failure_type, detail = SelfRepair.diagnose_scoring(workdir, config)
        assert failure_type == "benchmark_error"
        assert "no checkpoint" in detail

    def test_diagnose_test_failed(self, tmp_path):
        from agentforge.config import ChallengeConfig

        workdir = tmp_path / "exp"
        workdir.mkdir()
        (workdir / "test_suite.py").write_text(
            "import sys; print('FAIL'); sys.exit(1)"
        )
        config = ChallengeConfig(
            challenge_name="test", challenge_description="",
            target_metric="val_loss", target_value=1.0,
            target_direction="minimize",
            test_smoke="python3 test_suite.py",
            test_full="python3 test_suite.py",
            test_benchmark="python3 benchmark.py",
            writable=[], read_only=[],
        )
        failure_type, detail = SelfRepair.diagnose_scoring(workdir, config)
        assert failure_type == "test_failed"

    def test_diagnose_no_metric(self, tmp_path):
        from agentforge.config import ChallengeConfig

        workdir = tmp_path / "exp"
        workdir.mkdir()
        (workdir / "test_suite.py").write_text(
            "import sys; sys.exit(0)"
        )
        results_dir = workdir / "results"
        results_dir.mkdir()
        (results_dir / "benchmark.json").write_text(json.dumps({"wrong_key": 1.0}))
        (workdir / "benchmark.py").write_text(
            "pass  # results already exist"
        )
        config = ChallengeConfig(
            challenge_name="test", challenge_description="",
            target_metric="val_loss", target_value=1.0,
            target_direction="minimize",
            test_smoke="python3 test_suite.py",
            test_full="python3 test_suite.py",
            test_benchmark="python3 benchmark.py",
            writable=[], read_only=[],
        )
        failure_type, detail = SelfRepair.diagnose_scoring(workdir, config)
        assert failure_type == "no_results"
        assert "val_loss" in detail

    def test_build_repair_prompt_benchmark(self, tmp_path):
        from agentforge.config import ChallengeConfig

        workdir = tmp_path / "proj"
        workdir.mkdir()
        (workdir / "benchmark.py").write_text("raise Error('broken')")
        config = ChallengeConfig(
            challenge_name="test", challenge_description="",
            target_metric="val_loss", target_value=1.0,
            target_direction="minimize",
            test_smoke="", test_full="", test_benchmark="python3 benchmark.py",
            writable=[], read_only=[],
        )
        prompt = SelfRepair.build_repair_prompt(
            "benchmark_error", "FileNotFoundError: no ckpt", workdir, config,
        )
        assert "benchmark.py" in prompt
        assert "val_loss" in prompt
        assert "FileNotFoundError" in prompt
