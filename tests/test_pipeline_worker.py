# tests/test_pipeline_worker.py
import json
import os
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentforge.agent import Strategy, StrategySpec
from agentforge.pipeline import PipelineEvent, EventBus, PipelineWorker
from agentforge.state import HardwareInfo, StrategyResult


def _make_spec(name="test_strategy"):
    return StrategySpec(
        name=name, description="test", approach="do stuff",
        category="optim", risk="low", estimated_train_command="echo ok",
    )


def _make_hw():
    return HardwareInfo(
        device="cpu", gpu_model="", num_gpus=0,
        cpu_cores=4, ram_gb=16, disk_free_gb=100,
    )


def _make_strategy(name="test_strategy", branch="agentforge/iter-1/exp-0"):
    return Strategy(
        name=name, branch=branch, confidence=0.8,
        measured_vram_gb=0.0, measured_epoch_seconds=10.0,
        batch_size=1, resume_checkpoint=False,
        category="optim", risk="low", train_command="echo done",
    )


def test_worker_emits_events():
    """Worker should emit implementing -> training -> scoring -> done events."""
    bus = EventBus()
    events = []
    bus.subscribe(lambda e: events.append(e))
    consumer = threading.Thread(target=bus.run_consumer, daemon=True)
    consumer.start()

    spec = _make_spec()
    hw = _make_hw()

    mock_strategy = _make_strategy()
    mock_config = MagicMock()
    mock_config.test_benchmark = "echo ok"
    mock_config.test_full = "echo ok"

    worker = PipelineWorker(
        index=0, spec=spec, config=mock_config,
        hw=hw, round_num=1, workdir=Path("/tmp/test"),
        event_bus=bus, timeout=60, config_context="test",
    )

    with patch.object(worker, "_create_impl_worktree", return_value=Path("/tmp/test/impl")), \
         patch.object(worker, "_remove_impl_worktree"), \
         patch.object(worker, "_implement", return_value=mock_strategy), \
         patch.object(worker, "_train", return_value=0), \
         patch.object(worker, "_score", return_value=0.85):
        worker.run()

    bus.shutdown()
    consumer.join(timeout=2)

    phases = [e.phase for e in events]
    assert "implementing" in phases
    assert "training" in phases
    assert "scoring" in phases
    assert "done" in phases


def test_worker_handles_implement_failure():
    """Worker should emit 'failed' when Codex implementation fails."""
    bus = EventBus()
    events = []
    bus.subscribe(lambda e: events.append(e))
    consumer = threading.Thread(target=bus.run_consumer, daemon=True)
    consumer.start()

    spec = _make_spec()
    hw = _make_hw()
    mock_config = MagicMock()

    worker = PipelineWorker(
        index=0, spec=spec, config=mock_config,
        hw=hw, round_num=1, workdir=Path("/tmp/test"),
        event_bus=bus, timeout=60, config_context="test",
    )

    with patch.object(worker, "_create_impl_worktree", return_value=Path("/tmp/test/impl")), \
         patch.object(worker, "_remove_impl_worktree"), \
         patch.object(worker, "_implement", side_effect=RuntimeError("Codex failed")):
        worker.run()

    bus.shutdown()
    consumer.join(timeout=2)

    phases = [e.phase for e in events]
    assert "failed" in phases
    failed_event = [e for e in events if e.phase == "failed"][0]
    assert "Codex failed" in failed_event.error


def test_worker_result_property():
    """Worker.result should return StrategyResult after completion."""
    bus = EventBus()
    consumer = threading.Thread(target=bus.run_consumer, daemon=True)
    consumer.start()

    spec = _make_spec()
    hw = _make_hw()
    mock_strategy = _make_strategy()
    mock_config = MagicMock()

    worker = PipelineWorker(
        index=0, spec=spec, config=mock_config,
        hw=hw, round_num=1, workdir=Path("/tmp/test"),
        event_bus=bus, timeout=60, config_context="test",
    )

    with patch.object(worker, "_create_impl_worktree", return_value=Path("/tmp/test/impl")), \
         patch.object(worker, "_remove_impl_worktree"), \
         patch.object(worker, "_implement", return_value=mock_strategy), \
         patch.object(worker, "_train", return_value=0), \
         patch.object(worker, "_score", return_value=0.85):
        worker.run()

    bus.shutdown()
    consumer.join(timeout=2)

    result = worker.result
    assert isinstance(result, StrategyResult)
    assert result.score == 0.85
    assert result.status == "ok"
