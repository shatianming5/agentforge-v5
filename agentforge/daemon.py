from __future__ import annotations
import os
import signal
import sys
from pathlib import Path
from agentforge.orchestrator import Orchestrator


class Daemon:
    def __init__(self, config_path: Path, workdir: Path):
        self.config_path = config_path
        self.workdir = workdir
        self.pid_path = workdir / ".agentforge" / "daemon.pid"
        self.log_path = workdir / ".agentforge" / "daemon.log"
        self._should_stop = False

    def start(self):
        if self.is_running():
            print(f"AgentForge already running (PID {self.read_pid()})")
            return
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid = os.fork()
        if pid > 0:
            print(f"AgentForge daemon started (PID {pid})")
            return
        os.setsid()
        self._write_pid(os.getpid())
        self._redirect_stdio()
        self._setup_signals()
        try:
            orchestrator = Orchestrator(self.config_path, self.workdir,
                                        stop_flag=lambda: self._should_stop)
            orchestrator.run()
        finally:
            self._cleanup_pid()

    def stop(self):
        pid = self.read_pid()
        if pid is None:
            print("No running daemon found")
            return
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to PID {pid}")
        except ProcessLookupError:
            print(f"Process {pid} not found, cleaning up PID file")
            self._cleanup_pid()

    def is_running(self):
        pid = self.read_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def read_pid(self):
        if not self.pid_path.exists():
            return None
        try:
            return int(self.pid_path.read_text().strip())
        except (ValueError, OSError):
            return None

    def _write_pid(self, pid):
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(str(pid))

    def _cleanup_pid(self):
        if self.pid_path.exists():
            self.pid_path.unlink()

    def _redirect_stdio(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fd = open(self.log_path, "a")
        os.dup2(log_fd.fileno(), sys.stdout.fileno())
        os.dup2(log_fd.fileno(), sys.stderr.fileno())
        devnull = open(os.devnull, "r")
        os.dup2(devnull.fileno(), sys.stdin.fileno())

    def _setup_signals(self):
        signal.signal(signal.SIGTERM, self._handle_term)

    def _handle_term(self, signum, frame):
        self._should_stop = True
