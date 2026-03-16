from __future__ import annotations
import math
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentforge.orchestrator import Orchestrator
from agentforge.state import (
    SessionState, HardwareInfo, BestResult, Budget,
    StateFile, StrategyResult,
)
from agentforge.config import ChallengeConfig


def _make_config():
    return ChallengeConfig(
        "Test", "Desc", "accuracy", 0.95, "maximize",
        "echo smoke", "echo full", "echo bench",
        ["src/"], ["tests/"],
    )


def _make_state():
    return SessionState.create_initial(
        session_id="af-test", repo_url="/tmp/repo",
        hardware=HardwareInfo("cpu", "", 0, 4, 16, 50),
        N=1, gpus_per_experiment=0, rounds_max=3, gpu_hours_max=0,
    )


class TestOrchestrator:
    def test_done_target_reached(self):
        state = _make_state()
        state.best.score = 0.96
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = _make_config()
        assert orch._done(state) is True

    def test_not_done_initially(self):
        state = _make_state()
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = _make_config()
        assert orch._done(state) is False

    def test_done_budget_exhausted(self):
        state = _make_state()
        state.budget.rounds_used = 3
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = _make_config()
        assert orch._done(state) is True

    def test_select_winners_4_maximize(self):
        results = [
            StrategyResult("e0", "s0", "b0", 0.85, "ok", None, 40, 45, 64),
            StrategyResult("e1", "s1", "b1", 0.70, "ok", None, 40, 45, 64),
            StrategyResult("e2", "s2", "b2", 0.90, "ok", None, 40, 45, 64),
            StrategyResult("e3", "s3", "b3", 0.60, "ok", None, 40, 45, 64),
        ]
        winners = Orchestrator.select_winners(results, "maximize")
        assert "e2" in winners
        assert len(winners) == 1

    def test_select_winners_4_minimize(self):
        results = [
            StrategyResult("e0", "s0", "b0", 2.5, "ok", None, 40, 45, 64),
            StrategyResult("e1", "s1", "b1", 1.8, "ok", None, 40, 45, 64),
            StrategyResult("e2", "s2", "b2", 3.0, "ok", None, 40, 45, 64),
            StrategyResult("e3", "s3", "b3", 1.5, "ok", None, 40, 45, 64),
        ]
        winners = Orchestrator.select_winners(results, "minimize")
        assert "e3" in winners  # 1.5 是最小的
        assert len(winners) == 1

    def test_select_winners_8(self):
        results = [
            StrategyResult(f"e{i}", f"s{i}", f"b{i}", 0.5+i*0.05, "ok", None, 40, 45, 64)
            for i in range(8)
        ]
        winners = Orchestrator.select_winners(results, "maximize")
        assert len(winners) == 2

    def test_select_winners_all_failed(self):
        results = [
            StrategyResult("e0", "s0", "b0", 0, "error", "OOM", 0, 0, 32),
        ]
        winners = Orchestrator.select_winners(results)
        assert winners == []

    def test_done_minimize(self):
        state = _make_state()
        state.best = BestResult(1.5, 1, "exp-1", "abc", "")
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = ChallengeConfig(
            "Test", "Desc", "val_loss", 1.8, "minimize",
            "echo smoke", "echo full", "echo bench",
            ["src/"], ["tests/"],
        )
        assert orch._done(state) is True  # 1.5 < 1.8，达标

    def test_not_done_minimize(self):
        state = _make_state()
        state.best = BestResult(2.5, 1, "exp-1", "abc", "")
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = ChallengeConfig(
            "Test", "Desc", "val_loss", 1.8, "minimize",
            "echo smoke", "echo full", "echo bench",
            ["src/"], ["tests/"],
        )
        assert orch._done(state) is False  # 2.5 > 1.8，未达标


def test_orchestrator_auto_setup_when_no_config(tmp_path):
    """config_path=None 且 workdir 无 challenge.yaml 时触发 auto-setup。"""
    from unittest.mock import patch, MagicMock
    import json
    from agentforge.orchestrator import Orchestrator

    profile_data = {
        "description": "Test auto-setup project",
        "run_command": "python main.py",
        "eval_metric": "accuracy",
        "eval_direction": "maximize",
        "eval_method": "parse stdout",
        "suggested_target": 0.9,
        "writable": ["main.py"],
        "readonly": ["data/"],
        "metric_extraction": "score = 0.9",
        "import_checks": "",
    }
    af_dir = tmp_path / ".agentforge"
    af_dir.mkdir()

    def fake_codex(*args, **kwargs):
        (af_dir / "project_profile.json").write_text(json.dumps(profile_data))
        return MagicMock(returncode=0)

    def fake_confirm(files):
        for fname, content in files.items():
            (tmp_path / fname).write_text(content)
        return {f: "accepted" for f in files}

    with patch("subprocess.run", side_effect=fake_codex), \
         patch("agentforge.orchestrator.InteractiveConfirm") as MockConfirm:
        mock_instance = MagicMock()
        mock_instance.confirm_each = fake_confirm
        MockConfirm.return_value = mock_instance

        orch = Orchestrator(config_path=None, workdir=tmp_path)
        assert (tmp_path / "challenge.yaml").exists()


def test_orchestrator_skips_setup_when_config_provided(tmp_path):
    """config_path 已提供时不触发 auto-setup。"""
    import yaml
    from agentforge.orchestrator import Orchestrator

    config_content = {
        "challenge": {"name": "test", "description": "test desc"},
        "target": {"metric": "acc", "value": 0.9, "direction": "maximize"},
        "tests": {"smoke": "echo ok", "full": "echo ok", "benchmark": "echo ok"},
        "constraints": {"writable": ["x.py"], "read_only": ["data/"]},
    }
    config_path = tmp_path / "challenge.yaml"
    config_path.write_text(yaml.dump(config_content))

    orch = Orchestrator(config_path=config_path, workdir=tmp_path)
    assert orch.config.challenge_name == "test"
