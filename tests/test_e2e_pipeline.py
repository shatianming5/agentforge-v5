# tests/test_e2e_pipeline.py
"""端到端集成测试：验证完整的 spec -> pipeline -> results 流程。"""
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentforge.agent import StrategySpec, Strategy
from agentforge.experiment import Experiment
from agentforge.pipeline import PipelineOrchestrator, PipelineEvent
from agentforge.state import HardwareInfo, StrategyResult


def _make_hw():
    return HardwareInfo(
        device="cpu", gpu_model="", num_gpus=0,
        cpu_cores=4, ram_gb=16, disk_free_gb=100,
    )


def _make_experiment(index, tmp_path):
    """Create a real Experiment with a simple 'echo' train command."""
    log_dir = tmp_path / ".agentforge" / "runs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return Experiment(
        index=index,
        strategy=Strategy(
            name=f"s{index}",
            branch=f"agentforge/iter-1/exp-{index}",
            confidence=0.8,
            measured_vram_gb=0,
            measured_epoch_seconds=5,
            batch_size=1,
            resume_checkpoint=False,
            category="optim",
            risk="low",
            train_command="echo done",
        ),
        workdir=tmp_path,
        log_path=log_dir / f"exp-{index}.log",
        env=dict(os.environ),
        train_command=["echo", "training_done"],
    )


def test_e2e_pipeline_all_succeed(tmp_path):
    """All workers succeed: 3 specs -> 3 results with scores."""
    specs = [
        StrategySpec(name=f"s{i}", description=f"d{i}", approach=f"a{i}",
                     category="optim", risk="low",
                     estimated_train_command="echo ok")
        for i in range(3)
    ]

    config = MagicMock()
    config.test_benchmark = "echo ok"
    config.test_full = "echo ok"
    hw = _make_hw()
    events = []

    def mock_implement(spec, index, round_num, cwd, config_context, timeout):
        return Strategy(
            name=spec.name, branch=f"agentforge/iter-1/exp-{index}",
            confidence=0.8, measured_vram_gb=0, measured_epoch_seconds=5,
            batch_size=1, resume_checkpoint=False,
            category=spec.category, risk=spec.risk,
            train_command="echo done",
        )

    def mock_create(strategy, index, repo_path, workdir, hw, train_command,
                    default_benchmark=""):
        return _make_experiment(index, tmp_path)

    def mock_score(exp, config, returncode, N=1):
        return 0.85 + exp.index * 0.01

    with patch("agentforge.pipeline.CodexCLI.implement_strategy", side_effect=mock_implement), \
         patch("agentforge.pipeline.ExperimentSetup.create", side_effect=mock_create), \
         patch("agentforge.pipeline.Scorer.score", side_effect=mock_score):
        orch = PipelineOrchestrator(
            specs=specs, config=config, hw=hw,
            round_num=1, workdir=tmp_path,
            timeout=60, config_context="test",
        )
        orch.on_event(lambda e: events.append(e))
        results = orch.run()

    assert len(results) == 3
    scores = sorted([r.score for r in results])
    assert scores == [0.85, 0.86, 0.87]

    # Verify event flow: each worker should emit at least implementing + done
    for i in range(3):
        worker_events = [e for e in events if e.worker_index == i]
        phases = [e.phase for e in worker_events]
        assert "implementing" in phases
        assert "done" in phases or "training" in phases


def test_e2e_pipeline_one_fails(tmp_path):
    """One worker fails during implementation, others succeed."""
    specs = [
        StrategySpec(name=f"s{i}", description=f"d{i}", approach=f"a{i}",
                     category="optim", risk="low",
                     estimated_train_command="echo ok")
        for i in range(2)
    ]
    config = MagicMock()
    config.test_benchmark = "echo ok"
    config.test_full = "echo ok"
    hw = _make_hw()

    def mock_implement(spec, index, round_num, cwd, config_context, timeout):
        if index == 0:
            raise RuntimeError("Codex crashed")
        return Strategy(
            name=spec.name, branch=f"agentforge/iter-1/exp-{index}",
            confidence=0.8, measured_vram_gb=0, measured_epoch_seconds=5,
            batch_size=1, resume_checkpoint=False,
            category=spec.category, risk=spec.risk,
            train_command="echo done",
        )

    def mock_create(strategy, index, repo_path, workdir, hw, train_command,
                    default_benchmark=""):
        return _make_experiment(index, tmp_path)

    with patch("agentforge.pipeline.CodexCLI.implement_strategy", side_effect=mock_implement), \
         patch("agentforge.pipeline.ExperimentSetup.create", side_effect=mock_create), \
         patch("agentforge.pipeline.Scorer.score", return_value=0.9):
        orch = PipelineOrchestrator(
            specs=specs, config=config, hw=hw,
            round_num=1, workdir=tmp_path,
            timeout=60, config_context="test",
        )
        results = orch.run()

    assert len(results) == 2
    failed = [r for r in results if r.status == "error"]
    ok = [r for r in results if r.status == "ok"]
    assert len(failed) == 1
    assert len(ok) == 1
    assert ok[0].score == 0.9


def test_e2e_pipeline_event_ordering(tmp_path):
    """Verify that events are emitted in correct phase order per worker."""
    specs = [
        StrategySpec(name="only_one", description="d", approach="a",
                     category="optim", risk="low",
                     estimated_train_command="echo ok")
    ]
    config = MagicMock()
    config.test_benchmark = "echo ok"
    config.test_full = "echo ok"
    hw = _make_hw()
    events = []

    def mock_implement(spec, index, round_num, cwd, config_context, timeout):
        return Strategy(
            name=spec.name, branch=f"agentforge/iter-1/exp-{index}",
            confidence=0.8, measured_vram_gb=0, measured_epoch_seconds=5,
            batch_size=1, resume_checkpoint=False,
            category=spec.category, risk=spec.risk,
            train_command="echo done",
        )

    def mock_create(strategy, index, repo_path, workdir, hw, train_command,
                    default_benchmark=""):
        return _make_experiment(index, tmp_path)

    def mock_score(exp, config, returncode, N=1):
        return 0.95

    with patch("agentforge.pipeline.CodexCLI.implement_strategy", side_effect=mock_implement), \
         patch("agentforge.pipeline.ExperimentSetup.create", side_effect=mock_create), \
         patch("agentforge.pipeline.Scorer.score", side_effect=mock_score):
        orch = PipelineOrchestrator(
            specs=specs, config=config, hw=hw,
            round_num=1, workdir=tmp_path,
            timeout=60, config_context="test",
        )
        orch.on_event(lambda e: events.append(e))
        results = orch.run()

    assert len(results) == 1
    assert results[0].score == 0.95
    assert results[0].status == "ok"

    # The expected phase order: implementing -> training -> scoring -> done
    worker_events = [e for e in events if e.worker_index == 0]
    phases = [e.phase for e in worker_events]
    assert phases[0] == "implementing"
    # "training" may appear multiple times (log updates); "scoring" follows; "done" is last
    assert "done" in phases
    assert phases[-1] == "done"


def test_e2e_pipeline_empty_specs(tmp_path):
    """Empty specs list should return empty results."""
    config = MagicMock()
    hw = _make_hw()

    orch = PipelineOrchestrator(
        specs=[], config=config, hw=hw,
        round_num=1, workdir=tmp_path,
        timeout=60, config_context="test",
    )
    results = orch.run()
    assert results == []


def test_e2e_pipeline_all_fail(tmp_path):
    """All workers fail: should still return results with error status."""
    specs = [
        StrategySpec(name=f"s{i}", description=f"d{i}", approach=f"a{i}",
                     category="optim", risk="low",
                     estimated_train_command="echo ok")
        for i in range(3)
    ]
    config = MagicMock()
    config.test_benchmark = "echo ok"
    hw = _make_hw()
    events = []

    def mock_implement(spec, index, round_num, cwd, config_context, timeout):
        raise RuntimeError(f"Codex failed for {spec.name}")

    with patch("agentforge.pipeline.CodexCLI.implement_strategy", side_effect=mock_implement):
        orch = PipelineOrchestrator(
            specs=specs, config=config, hw=hw,
            round_num=1, workdir=tmp_path,
            timeout=60, config_context="test",
        )
        orch.on_event(lambda e: events.append(e))
        results = orch.run()

    assert len(results) == 3
    assert all(r.status == "error" for r in results)
    assert all(r.score == 0.0 for r in results)

    # Each worker should emit "implementing" then "failed"
    for i in range(3):
        worker_events = [e for e in events if e.worker_index == i]
        phases = [e.phase for e in worker_events]
        assert "implementing" in phases
        assert "failed" in phases


def test_e2e_pipeline_result_fields(tmp_path):
    """Verify that StrategyResult fields are correctly populated."""
    specs = [
        StrategySpec(name="test_strategy", description="d", approach="a",
                     category="arch", risk="high",
                     estimated_train_command="echo ok")
    ]
    config = MagicMock()
    config.test_benchmark = "echo ok"
    config.test_full = "echo ok"
    hw = _make_hw()

    def mock_implement(spec, index, round_num, cwd, config_context, timeout):
        return Strategy(
            name="test_strategy",
            branch="agentforge/iter-1/exp-0",
            confidence=0.9,
            measured_vram_gb=4.5,
            measured_epoch_seconds=120,
            batch_size=8,
            resume_checkpoint=False,
            category="arch",
            risk="high",
            train_command="echo done",
        )

    def mock_create(strategy, index, repo_path, workdir, hw, train_command,
                    default_benchmark=""):
        return _make_experiment(index, tmp_path)

    def mock_score(exp, config, returncode, N=1):
        return 0.92

    with patch("agentforge.pipeline.CodexCLI.implement_strategy", side_effect=mock_implement), \
         patch("agentforge.pipeline.ExperimentSetup.create", side_effect=mock_create), \
         patch("agentforge.pipeline.Scorer.score", side_effect=mock_score):
        orch = PipelineOrchestrator(
            specs=specs, config=config, hw=hw,
            round_num=1, workdir=tmp_path,
            timeout=60, config_context="test",
        )
        results = orch.run()

    assert len(results) == 1
    r = results[0]
    assert r.id == "exp-0"
    assert r.strategy == "test_strategy"
    assert r.branch == "agentforge/iter-1/exp-0"
    assert r.score == 0.92
    assert r.status == "ok"
    assert r.error is None
    assert r.actual_vram_gb == 4.5
    assert r.actual_epoch_seconds == 120
    assert r.actual_batch_size == 8
