"""流水线并行架构：PipelineWorker, PipelineOrchestrator, EventBus."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

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
