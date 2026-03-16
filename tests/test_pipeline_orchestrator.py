# tests/test_pipeline_orchestrator.py
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentforge.agent import Strategy, StrategySpec
from agentforge.pipeline import (
    PipelineEvent, EventBus, PipelineWorker, PipelineOrchestrator,
)
from agentforge.state import HardwareInfo, StrategyResult


def _make_specs(n=2):
    return [
        StrategySpec(
            name=f"strategy_{i}", description=f"desc {i}",
            approach=f"approach {i}", category="optim",
            risk="low", estimated_train_command="echo ok",
        )
        for i in range(n)
    ]


def _make_hw():
    return HardwareInfo(
        device="cpu", gpu_model="", num_gpus=0,
        cpu_cores=4, ram_gb=16, disk_free_gb=100,
    )


def _make_result(index, score=0.85, status="ok"):
    return StrategyResult(
        id=f"exp-{index}", strategy=f"strategy_{index}",
        branch=f"agentforge/iter-1/exp-{index}",
        score=score, status=status, error=None,
        actual_vram_gb=0.0, actual_epoch_seconds=10.0,
        actual_batch_size=1,
    )


def test_orchestrator_runs_all_workers():
    specs = _make_specs(3)
    hw = _make_hw()
    mock_config = MagicMock()
    mock_config.test_benchmark = "echo ok"

    events = []

    def mock_worker_run(worker):
        worker._result = _make_result(worker.index)
        worker._emit("implementing")
        worker._emit("done", score=0.85)

    with patch.object(PipelineWorker, "run", autospec=True, side_effect=mock_worker_run):
        orch = PipelineOrchestrator(
            specs=specs, config=mock_config, hw=hw,
            round_num=1, workdir=Path("/tmp/test"),
            timeout=60, config_context="test",
        )
        orch.on_event(lambda e: events.append(e))
        results = orch.run()

    assert len(results) == 3
    assert all(isinstance(r, StrategyResult) for r in results)
    assert all(r.score == 0.85 for r in results)


def test_orchestrator_collects_events():
    specs = _make_specs(2)
    hw = _make_hw()
    mock_config = MagicMock()

    events = []

    def mock_worker_run(worker):
        worker._result = _make_result(worker.index)
        worker._emit("implementing")
        worker._emit("training")
        worker._emit("done", score=0.9)

    with patch.object(PipelineWorker, "run", autospec=True, side_effect=mock_worker_run):
        orch = PipelineOrchestrator(
            specs=specs, config=mock_config, hw=hw,
            round_num=1, workdir=Path("/tmp/test"),
            timeout=60, config_context="test",
        )
        orch.on_event(lambda e: events.append(e))
        orch.run()

    # 2 workers * 3 events each = 6 events
    assert len(events) == 6
    phases = [e.phase for e in events]
    assert phases.count("implementing") == 2
    assert phases.count("done") == 2
