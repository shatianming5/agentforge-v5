from __future__ import annotations
import os
import re
import shutil
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen


@dataclass
class MonitorEvent:
    exp_index: int
    reason: str  # "timeout" | "nan" | "vram_leak" | "disk" | "straggler"
    detail: str


class Monitor:
    NAN_PATTERN = re.compile(r"(?:loss|train_loss|val_loss)\s*[=:]\s*nan", re.IGNORECASE)
    CHECK_INTERVAL = 10

    def __init__(self, processes, timeout, log_paths=None, workdir=None, disk_threshold=0.9):
        self._processes = processes  # list of (index, proc, log_path)
        self._timeout = timeout
        self._log_paths = log_paths or []
        self._workdir = workdir or Path(".")
        self._disk_threshold = disk_threshold
        self._start_time = time.time()
        self._events: list[MonitorEvent] = []
        self._lock = threading.Lock()
        self._killed: set[int] = set()

    @property
    def events(self):
        with self._lock:
            return list(self._events)

    def run(self):
        thread = threading.Thread(target=self._monitor_loop, daemon=True)
        thread.start()
        self._wait_all()
        thread.join(timeout=5)

    def _wait_all(self):
        for idx, proc, _ in self._processes:
            proc.wait()

    def _monitor_loop(self):
        counter = 0
        while self._any_alive():
            for idx, proc, log_path in self._processes:
                if idx in self._killed or proc.poll() is not None:
                    continue
                if self._is_timed_out(proc):
                    self._kill(idx, proc, "timeout", f"exceeded {self._timeout}s")
                    continue
                if counter % 3 == 0 and self._check_nan_in_log(idx, log_path):
                    self._kill(idx, proc, "nan", "NaN detected in loss")
                    continue
            if counter % 6 == 0 and self._check_disk_critical():
                self._add_event(-1, "disk", "Disk usage critical")
            self._check_stragglers()
            counter += 1
            time.sleep(self.CHECK_INTERVAL)

    def _any_alive(self):
        return any(proc.poll() is None for _, proc, _ in self._processes)

    def _is_timed_out(self, proc):
        if proc.poll() is not None:
            return False
        return (time.time() - self._start_time) > self._timeout

    def _check_nan_in_log(self, index, log_path):
        if not log_path.exists():
            return False
        try:
            with open(log_path) as f:
                lines = f.readlines()[-100:]
            return any(self.NAN_PATTERN.search(line) for line in lines)
        except OSError:
            return False

    def _check_disk_critical(self):
        usage = shutil.disk_usage(self._workdir)
        return (usage.used / usage.total) > self._disk_threshold

    def _check_stragglers(self):
        alive = [(i, p) for i, p, _ in self._processes if p.poll() is None and i not in self._killed]
        done = [(i, p) for i, p, _ in self._processes if p.poll() is not None or i in self._killed]
        if len(alive) == 1 and len(done) >= 2:
            elapsed = time.time() - self._start_time
            if elapsed > self._timeout * 0.8:
                idx, proc = alive[0]
                self._kill(idx, proc, "straggler", "last experiment, 80% of timeout")

    def _kill(self, index, proc, reason, detail):
        with self._lock:
            if index in self._killed:
                return
            self._killed.add(index)
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
        self._add_event(index, reason, detail)

    def _add_event(self, index, reason, detail):
        with self._lock:
            self._events.append(MonitorEvent(index, reason, detail))
