from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentforge.state import (
    BestResult,
    Budget,
    HardwareInfo,
    RoundResult,
    SessionState,
    StateFile,
    StrategyRecord,
    StrategyResult,
)


# ---------------------------------------------------------------------------
# HardwareInfo
# ---------------------------------------------------------------------------

class TestHardwareInfo:
    def test_create(self):
        hw = HardwareInfo(
            device="cuda",
            gpu_model="A100 80GB",
            num_gpus=4,
            cpu_cores=64,
            ram_gb=512,
            disk_free_gb=1000,
        )
        assert hw.device == "cuda"
        assert hw.gpu_model == "A100 80GB"
        assert hw.num_gpus == 4
        assert hw.cpu_cores == 64
        assert hw.ram_gb == 512
        assert hw.disk_free_gb == 1000


# ---------------------------------------------------------------------------
# StrategyResult
# ---------------------------------------------------------------------------

class TestStrategyResult:
    def test_success(self):
        sr = StrategyResult(
            id="exp-001",
            strategy="lr_sweep",
            branch="exp/lr_sweep_001",
            score=0.92,
            status="ok",
            error=None,
            actual_vram_gb=40.0,
            actual_epoch_seconds=120.5,
            actual_batch_size=32,
        )
        assert sr.status == "ok"
        assert sr.error is None
        assert sr.score == 0.92

    def test_failure(self):
        sr = StrategyResult(
            id="exp-002",
            strategy="big_batch",
            branch="exp/big_batch_002",
            score=0.0,
            status="oom",
            error="CUDA out of memory",
            actual_vram_gb=80.0,
            actual_epoch_seconds=0.0,
            actual_batch_size=256,
        )
        assert sr.status == "oom"
        assert sr.error == "CUDA out of memory"
        assert sr.score == 0.0


# ---------------------------------------------------------------------------
# SessionState.create_initial
# ---------------------------------------------------------------------------

class TestSessionStateCreateInitial:
    def test_create_initial(self):
        hw = HardwareInfo("cuda", "A100 80GB", 4, 64, 512, 1000)
        state = SessionState.create_initial(
            session_id="sess-abc",
            repo_url="https://github.com/user/repo",
            hardware=hw,
            N=4,
            gpus_per_experiment=1,
            rounds_max=10,
            gpu_hours_max=24.0,
        )
        assert state.version == "5.1"
        assert state.session_id == "sess-abc"
        assert state.repo_url == "https://github.com/user/repo"
        assert state.status == "running"
        assert state.hardware is hw
        assert state.N == 4
        assert state.gpus_per_experiment == 1
        assert state.best.score == 0.0
        assert state.best.round == 0
        assert state.best.experiment == ""
        assert state.best.commit == ""
        assert state.best.checkpoint == ""
        assert state.current_round == 0
        assert state.score_trajectory == []
        assert state.rounds == []
        assert state.strategies_tried == []
        assert state.budget.rounds_used == 0
        assert state.budget.rounds_max == 10
        assert state.budget.gpu_hours_used == 0.0
        assert state.budget.gpu_hours_max == 24.0
        assert state.budget.api_cost_usd == 0.0
        assert state.hints_pending == []
        assert state.env_lockfile_hash == ""


# ---------------------------------------------------------------------------
# SessionState.is_done
# ---------------------------------------------------------------------------

class TestIsDone:
    @pytest.fixture
    def base_state(self):
        hw = HardwareInfo("cuda", "A100 80GB", 4, 64, 512, 1000)
        return SessionState.create_initial(
            session_id="sess-test",
            repo_url="https://github.com/user/repo",
            hardware=hw,
            N=4,
            gpus_per_experiment=1,
            rounds_max=10,
            gpu_hours_max=24.0,
        )

    def test_target_reached(self, base_state: SessionState):
        """达标：best.score >= target_value"""
        base_state.best = BestResult(0.95, 3, "exp-005", "abc123", "ckpt.pt")
        assert base_state.is_done(target_value=0.95) is True
        assert base_state.is_done(target_value=0.90) is True

    def test_budget_exhausted_rounds(self, base_state: SessionState):
        """预算耗尽（rounds）"""
        base_state.budget.rounds_used = 10
        assert base_state.is_done(target_value=0.99) is True

    def test_budget_exhausted_gpu_hours(self, base_state: SessionState):
        """预算耗尽（gpu_hours）"""
        base_state.budget.gpu_hours_used = 24.0
        assert base_state.is_done(target_value=0.99) is True

    def test_not_done(self, base_state: SessionState):
        """未完成"""
        assert base_state.is_done(target_value=0.99) is False

    def test_status_completed(self, base_state: SessionState):
        """状态为 completed"""
        base_state.status = "completed"
        assert base_state.is_done(target_value=0.99) is True

    def test_status_failed(self, base_state: SessionState):
        """状态为 failed"""
        base_state.status = "failed"
        assert base_state.is_done(target_value=0.99) is True

    def test_status_paused(self, base_state: SessionState):
        """状态为 paused"""
        base_state.status = "paused"
        assert base_state.is_done(target_value=0.99) is True


class TestIsDoneMinimize:
    @pytest.fixture
    def min_state(self):
        hw = HardwareInfo("cuda", "A100 80GB", 4, 64, 512, 1000)
        return SessionState.create_initial(
            session_id="sess-min",
            repo_url="https://github.com/user/repo",
            hardware=hw,
            N=4, gpus_per_experiment=1,
            rounds_max=10, gpu_hours_max=24.0,
            direction="minimize",
        )

    def test_initial_score_is_inf(self, min_state):
        assert min_state.best.score == float("inf")

    def test_target_reached_minimize(self, min_state):
        min_state.best = BestResult(1.5, 3, "exp-005", "abc", "ckpt.pt")
        assert min_state.is_done(target_value=1.8, direction="minimize") is True

    def test_not_done_minimize(self, min_state):
        min_state.best = BestResult(2.5, 1, "exp-001", "abc", "ckpt.pt")
        assert min_state.is_done(target_value=1.8, direction="minimize") is False

    def test_exact_target_minimize(self, min_state):
        min_state.best = BestResult(1.8, 2, "exp-003", "abc", "ckpt.pt")
        assert min_state.is_done(target_value=1.8, direction="minimize") is True


# ---------------------------------------------------------------------------
# StateFile: save + load round-trip
# ---------------------------------------------------------------------------

class TestStateFile:
    def _make_state(self) -> SessionState:
        hw = HardwareInfo("cuda", "A100 80GB", 4, 64, 512, 1000)
        state = SessionState.create_initial(
            session_id="sess-rt",
            repo_url="https://github.com/user/repo",
            hardware=hw,
            N=4,
            gpus_per_experiment=1,
            rounds_max=10,
            gpu_hours_max=24.0,
        )
        # 添加一些真实数据以确保 round-trip 完整
        state.current_round = 1
        state.score_trajectory = [0.5, 0.7]
        state.best = BestResult(0.7, 1, "exp-003", "def456", "model.pt")
        state.budget.rounds_used = 1
        state.budget.gpu_hours_used = 2.5
        state.budget.api_cost_usd = 0.12
        state.hints_pending = ["try augmentation"]
        state.env_lockfile_hash = "sha256:abc123"

        exp1 = StrategyResult(
            id="exp-001", strategy="baseline", branch="exp/baseline_001",
            score=0.5, status="ok", error=None,
            actual_vram_gb=30.0, actual_epoch_seconds=60.0, actual_batch_size=16,
        )
        exp2 = StrategyResult(
            id="exp-003", strategy="lr_sweep", branch="exp/lr_sweep_003",
            score=0.7, status="ok", error=None,
            actual_vram_gb=35.0, actual_epoch_seconds=80.0, actual_batch_size=32,
        )
        exp_fail = StrategyResult(
            id="exp-002", strategy="big_batch", branch="exp/big_batch_002",
            score=0.0, status="oom", error="CUDA out of memory",
            actual_vram_gb=80.0, actual_epoch_seconds=0.0, actual_batch_size=256,
        )
        rr = RoundResult(
            round=1,
            experiments=[exp1, exp2, exp_fail],
            winners=["exp-003"],
            phase1_minutes=5.0,
            phase2_minutes=10.0,
        )
        state.rounds = [rr]
        state.strategies_tried = [
            StrategyRecord("baseline", 1, 0.5, "ok"),
            StrategyRecord("lr_sweep", 1, 0.7, "ok"),
            StrategyRecord("big_batch", 1, 0.0, "oom"),
        ]
        return state

    def test_save_load_roundtrip(self, tmp_path: Path):
        sf = StateFile(tmp_path / "state.json")
        original = self._make_state()
        sf.save(original)
        loaded = sf.load()

        # 基本字段
        assert loaded.version == original.version
        assert loaded.session_id == original.session_id
        assert loaded.repo_url == original.repo_url
        assert loaded.status == original.status
        assert loaded.N == original.N
        assert loaded.gpus_per_experiment == original.gpus_per_experiment
        assert loaded.current_round == original.current_round
        assert loaded.score_trajectory == original.score_trajectory
        assert loaded.hints_pending == original.hints_pending
        assert loaded.env_lockfile_hash == original.env_lockfile_hash

        # HardwareInfo
        assert loaded.hardware.device == original.hardware.device
        assert loaded.hardware.gpu_model == original.hardware.gpu_model
        assert loaded.hardware.num_gpus == original.hardware.num_gpus

        # BestResult
        assert loaded.best.score == original.best.score
        assert loaded.best.round == original.best.round
        assert loaded.best.experiment == original.best.experiment
        assert loaded.best.commit == original.best.commit
        assert loaded.best.checkpoint == original.best.checkpoint

        # Budget
        assert loaded.budget.rounds_used == original.budget.rounds_used
        assert loaded.budget.rounds_max == original.budget.rounds_max
        assert loaded.budget.gpu_hours_used == original.budget.gpu_hours_used
        assert loaded.budget.gpu_hours_max == original.budget.gpu_hours_max
        assert loaded.budget.api_cost_usd == original.budget.api_cost_usd

        # Rounds
        assert len(loaded.rounds) == 1
        r = loaded.rounds[0]
        assert r.round == 1
        assert len(r.experiments) == 3
        assert r.winners == ["exp-003"]
        assert r.phase1_minutes == 5.0
        assert r.phase2_minutes == 10.0

        # StrategyResult 内容
        assert r.experiments[0].id == "exp-001"
        assert r.experiments[0].status == "ok"
        assert r.experiments[2].status == "oom"
        assert r.experiments[2].error == "CUDA out of memory"

        # StrategyRecord
        assert len(loaded.strategies_tried) == 3
        assert loaded.strategies_tried[1].name == "lr_sweep"
        assert loaded.strategies_tried[1].score == 0.7

    def test_no_tmp_residue(self, tmp_path: Path):
        """原子写入后不应残留 .tmp 文件"""
        sf = StateFile(tmp_path / "state.json")
        state = self._make_state()
        sf.save(state)

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Residual .tmp files found: {tmp_files}"
        assert sf.path.exists()

    def test_exists(self, tmp_path: Path):
        sf = StateFile(tmp_path / "state.json")
        assert sf.exists() is False

        sf.save(self._make_state())
        assert sf.exists() is True

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        """save 应自动创建不存在的父目录"""
        sf = StateFile(tmp_path / "deep" / "nested" / "state.json")
        sf.save(self._make_state())
        assert sf.path.exists()
        loaded = sf.load()
        assert loaded.session_id == "sess-rt"

    def test_saved_json_valid(self, tmp_path: Path):
        """保存的文件应是合法 JSON"""
        sf = StateFile(tmp_path / "state.json")
        sf.save(self._make_state())
        with open(sf.path) as f:
            data = json.load(f)
        assert data["version"] == "5.1"
        assert isinstance(data["rounds"], list)
