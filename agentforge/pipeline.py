"""流水线并行架构：PipelineWorker, PipelineOrchestrator, EventBus."""
from __future__ import annotations

import os
import queue
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from agentforge.agent import CodexCLI, Strategy, StrategySpec
from agentforge.config import ChallengeConfig
from agentforge.experiment import ExperimentSetup, Experiment
from agentforge.monitor import SingleExperimentMonitor
from agentforge.scorer import Scorer
from agentforge.state import HardwareInfo, StrategyResult

_SENTINEL = object()


@dataclass
class PipelineEvent:
    worker_index: int
    strategy_name: str
    phase: str  # "implementing" | "training" | "scoring" | "done" | "failed"
    timestamp: float
    progress: dict | None = None
    score: float | None = None
    error: str | None = None
    log_tail: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class EventBus:
    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._subscribers: list[Callable[[PipelineEvent], None]] = []

    def subscribe(self, callback: Callable[[PipelineEvent], None]) -> None:
        self._subscribers.append(callback)

    def emit(self, event: PipelineEvent) -> None:
        self._queue.put(event)

    def shutdown(self) -> None:
        self._queue.put(_SENTINEL)

    def run_consumer(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            for cb in self._subscribers:
                try:
                    cb(item)
                except Exception:
                    pass


class PipelineWorker:
    """一个策略的完整流水线：Codex 实现 -> worktree -> 训练 -> 评分。"""

    CHECK_INTERVAL = 10

    def __init__(
        self,
        index: int,
        spec: StrategySpec,
        config: ChallengeConfig,
        hw: HardwareInfo,
        round_num: int,
        workdir: Path,
        event_bus: EventBus,
        timeout: int = 345600,
        config_context: str = "",
    ):
        self.index = index
        self.spec = spec
        self.config = config
        self.hw = hw
        self.round_num = round_num
        self.workdir = workdir
        self.event_bus = event_bus
        self.timeout = timeout
        self.config_context = config_context
        self._result: StrategyResult | None = None
        self._experiment: Experiment | None = None

    @property
    def result(self) -> StrategyResult | None:
        return self._result

    def _emit(self, phase: str, **kwargs) -> None:
        self.event_bus.emit(PipelineEvent(
            worker_index=self.index,
            strategy_name=self.spec.name,
            phase=phase,
            timestamp=time.time(),
            **kwargs,
        ))

    def run(self) -> None:
        try:
            # Phase: Implementing
            self._emit("implementing")
            strategy = self._implement()

            # Phase: Training
            self._emit("training")
            returncode = self._train(strategy)

            # Phase: Scoring
            self._emit("scoring")
            score = self._score(strategy, returncode)

            # Done
            self._result = StrategyResult(
                id=f"exp-{self.index}",
                strategy=strategy.name,
                branch=strategy.branch,
                score=score,
                status="ok" if returncode == 0 else "error",
                error=None if returncode == 0 else f"exit code {returncode}",
                actual_vram_gb=strategy.measured_vram_gb,
                actual_epoch_seconds=strategy.measured_epoch_seconds,
                actual_batch_size=strategy.batch_size,
            )
            self._emit("done", score=score)

        except Exception as e:
            self._result = StrategyResult(
                id=f"exp-{self.index}",
                strategy=self.spec.name,
                branch="",
                score=0.0,
                status="error",
                error=str(e),
                actual_vram_gb=0.0,
                actual_epoch_seconds=0.0,
                actual_batch_size=0,
            )
            self._emit("failed", error=str(e))

    def _implement(self) -> Strategy:
        return CodexCLI.implement_strategy(
            spec=self.spec,
            index=self.index,
            round_num=self.round_num,
            cwd=self.workdir,
            config_context=self.config_context,
            timeout=self.timeout // 2,
        )

    def _train(self, strategy: Strategy) -> int:
        runs_dir = self.workdir / ".agentforge" / "runs"
        self._experiment = ExperimentSetup.create(
            strategy=strategy,
            index=self.index,
            repo_path=self.workdir,
            workdir=runs_dir,
            hw=self.hw,
            train_command=shlex.split(self.config.test_benchmark),
        )

        self._experiment.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(self._experiment.log_path, "w")
        try:
            proc = subprocess.Popen(
                self._experiment.train_command,
                cwd=str(self._experiment.workdir),
                env=self._experiment.env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp,
            )
            monitor = SingleExperimentMonitor(
                index=self.index,
                log_path=self._experiment.log_path,
                timeout=self.timeout,
            )
            while proc.poll() is None:
                if monitor.is_timed_out():
                    self._kill_proc(proc)
                    return -1
                if monitor.check_nan():
                    self._kill_proc(proc)
                    return -2
                new_lines = monitor.read_new_lines()
                if new_lines:
                    self._emit("training", log_tail="\n".join(new_lines[-3:]),
                               progress={"elapsed_s": int(monitor.elapsed_seconds())})
                time.sleep(self.CHECK_INTERVAL)
            return proc.returncode
        finally:
            log_file.close()

    def _score(self, strategy: Strategy, returncode: int) -> float:
        if returncode != 0:
            return 0.0
        if self._experiment is None:
            return 0.0
        return Scorer.score(self._experiment, self.config, returncode, N=1)

    @staticmethod
    def _kill_proc(proc: subprocess.Popen) -> None:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            proc.wait(timeout=5)
        except (ProcessLookupError, PermissionError, OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
