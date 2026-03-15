from __future__ import annotations
import time
from pathlib import Path
from unittest.mock import MagicMock
from agentforge.monitor import Monitor, MonitorEvent


class TestMonitorEvent:
    def test_create(self):
        evt = MonitorEvent(exp_index=0, reason="timeout", detail="exceeded 600s")
        assert evt.exp_index == 0
        assert evt.reason == "timeout"


class TestMonitor:
    def test_check_nan_found(self, tmp_path):
        log = tmp_path / "exp-0.log"
        log.write_text("epoch 1 loss=0.5\nepoch 2 loss=nan\n")
        m = Monitor([], 600, [log])
        assert m._check_nan_in_log(0, log) is True

    def test_check_nan_not_found(self, tmp_path):
        log = tmp_path / "exp-0.log"
        log.write_text("epoch 1 loss=0.5\nepoch 2 loss=0.4\n")
        m = Monitor([], 600, [log])
        assert m._check_nan_in_log(0, log) is False

    def test_check_timeout_true(self):
        m = Monitor([], 60, [])
        m._start_time = time.time() - 100
        proc = MagicMock()
        proc.poll.return_value = None
        assert m._is_timed_out(proc) is True

    def test_check_timeout_false(self):
        m = Monitor([], 600, [])
        m._start_time = time.time()
        proc = MagicMock()
        proc.poll.return_value = None
        assert m._is_timed_out(proc) is False

    def test_check_disk_not_critical(self, tmp_path):
        m = Monitor([], 600, [], workdir=tmp_path)
        assert m._check_disk_critical() is False
