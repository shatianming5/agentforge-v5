from __future__ import annotations
import os
import re
import shutil
import signal
import subprocess
import time
import threading
from dataclasses import dataclass, field
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

    NAN_CHECK_EVERY = 3   # every 30s (3 * 10s interval)
    DISK_CHECK_EVERY = 6  # every 60s
    VRAM_CHECK_EVERY = 6  # every 60s
    LOG_TAIL_EVERY = 3    # every 30s

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
        self._vram_history: dict[int, list[tuple[float, float]]] = {}  # idx -> [(time, vram_mb)]
        self._completion_times: list[float] = []  # for straggler median
        self._last_log_pos: dict[int, int] = {}  # idx -> last read position

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
                    # Track completion time for straggler median
                    if proc.poll() is not None and idx not in self._killed:
                        elapsed = time.time() - self._start_time
                        if elapsed not in self._completion_times:
                            self._completion_times.append(elapsed)
                    continue
                if self._is_timed_out(proc):
                    self._kill(idx, proc, "timeout", f"exceeded {self._timeout}s")
                    continue
                if counter % self.NAN_CHECK_EVERY == 0 and self._check_nan_in_log(idx, log_path):
                    self._kill(idx, proc, "nan", "NaN detected in loss")
                    continue
                if counter % self.VRAM_CHECK_EVERY == 0:
                    self._check_vram_trend(idx, proc)
            if counter % self.DISK_CHECK_EVERY == 0 and self._check_disk_critical():
                self._add_event(-1, "disk", "Disk usage critical")
            if counter % self.LOG_TAIL_EVERY == 0:
                self._print_log_tails()
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

    def _check_vram_trend(self, index, proc):
        """Check for VRAM leak by querying nvidia-smi and extrapolating."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return
        now = time.time()
        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split(", ")
            if len(parts) == 2:
                try:
                    pid, mem_mb = int(parts[0]), float(parts[1])
                except ValueError:
                    continue
                if pid == proc.pid:
                    history = self._vram_history.setdefault(index, [])
                    history.append((now, mem_mb))
                    if len(history) >= 3:
                        self._extrapolate_oom(index, proc, history)
                    break

    def _extrapolate_oom(self, index, proc, history):
        """Linear extrapolation: if VRAM will exceed GPU total before timeout, kill."""
        times = [h[0] - history[0][0] for h in history]
        mems = [h[1] for h in history]
        if len(times) < 3 or times[-1] - times[0] < 60:
            return
        slope = (mems[-1] - mems[0]) / (times[-1] - times[0])
        if slope <= 0:
            return  # No leak
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            total_mb = float(result.stdout.strip().split("\n")[0])
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            return
        remaining_time = self._timeout - (time.time() - self._start_time)
        projected = mems[-1] + slope * remaining_time
        if projected > total_mb * 0.95:
            self._kill(index, proc, "vram_leak",
                       f"VRAM trend {slope:.0f}MB/s, projected {projected:.0f}MB > {total_mb:.0f}MB")

    def _check_stragglers(self):
        alive = [(i, p) for i, p, _ in self._processes if p.poll() is None and i not in self._killed]
        done = [(i, p) for i, p, _ in self._processes if p.poll() is not None or i in self._killed]
        if len(alive) == 1 and len(done) >= 2:
            if self._completion_times:
                sorted_times = sorted(self._completion_times)
                mid = len(sorted_times) // 2
                median_time = sorted_times[mid]
                grace = median_time * 0.2
                elapsed = time.time() - self._start_time
                if elapsed > median_time + grace:
                    idx, proc = alive[0]
                    self._kill(idx, proc, "straggler",
                               f"exceeded median({median_time:.0f}s) + 20% grace")
            else:
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
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            proc.wait(timeout=5)
        except (ProcessLookupError, PermissionError, OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
        self._add_event(index, reason, detail)

    def _print_log_tails(self):
        """定期打印各实验日志的最新内容。"""
        import sys as _sys
        elapsed = int(time.time() - self._start_time)
        for idx, proc, log_path in self._processes:
            if idx in self._killed or proc.poll() is not None:
                continue
            if not log_path.exists():
                continue
            try:
                last_pos = self._last_log_pos.get(idx, 0)
                with open(log_path) as f:
                    f.seek(last_pos)
                    new_content = f.read()
                    new_pos = f.tell()
                self._last_log_pos[idx] = new_pos
                if not new_content.strip():
                    continue
                # 只显示最后几行有意义的内容
                lines = new_content.strip().split("\n")
                tail = lines[-3:] if len(lines) > 3 else lines
                for line in tail:
                    line = line.strip()
                    if line:
                        _sys.stdout.write(f"  [exp-{idx} {elapsed}s] {line}\n")
                _sys.stdout.flush()
            except OSError:
                continue

    def _add_event(self, index, reason, detail):
        with self._lock:
            self._events.append(MonitorEvent(index, reason, detail))


class SingleExperimentMonitor:
    """单个实验的监控器，供 PipelineWorker 使用。"""

    NAN_PATTERN = Monitor.NAN_PATTERN

    def __init__(self, index: int, log_path: Path, timeout: int):
        self.index = index
        self.log_path = log_path
        self.timeout = timeout
        self._start_time = time.time()
        self._last_pos = 0

    def is_timed_out(self) -> bool:
        return (time.time() - self._start_time) > self.timeout

    def check_nan(self) -> bool:
        if not self.log_path.exists():
            return False
        try:
            with open(self.log_path) as f:
                lines = f.readlines()[-100:]
            return any(self.NAN_PATTERN.search(line) for line in lines)
        except OSError:
            return False

    def read_new_lines(self) -> list[str]:
        if not self.log_path.exists():
            return []
        try:
            with open(self.log_path) as f:
                f.seek(self._last_pos)
                content = f.read()
                self._last_pos = f.tell()
            if not content.strip():
                return []
            return [l for l in content.strip().split("\n") if l.strip()]
        except OSError:
            return []

    def elapsed_seconds(self) -> float:
        return time.time() - self._start_time
