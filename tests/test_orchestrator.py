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

    def test_select_winners_4(self):
        results = [
            StrategyResult("e0", "s0", "b0", 0.85, "ok", None, 40, 45, 64),
            StrategyResult("e1", "s1", "b1", 0.70, "ok", None, 40, 45, 64),
            StrategyResult("e2", "s2", "b2", 0.90, "ok", None, 40, 45, 64),
            StrategyResult("e3", "s3", "b3", 0.60, "ok", None, 40, 45, 64),
        ]
        winners = Orchestrator.select_winners(results)
        assert "e2" in winners
        assert len(winners) == 1  # ceil(4/4) = 1

    def test_select_winners_8(self):
        results = [
            StrategyResult(f"e{i}", f"s{i}", f"b{i}", 0.5+i*0.05, "ok", None, 40, 45, 64)
            for i in range(8)
        ]
        winners = Orchestrator.select_winners(results)
        assert len(winners) == 2  # ceil(8/4) = 2

    def test_select_winners_all_failed(self):
        results = [
            StrategyResult("e0", "s0", "b0", 0, "error", "OOM", 0, 0, 32),
        ]
        winners = Orchestrator.select_winners(results)
        assert winners == []
