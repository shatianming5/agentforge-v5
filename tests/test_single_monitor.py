# tests/test_single_monitor.py
import os
import time
from pathlib import Path
from unittest.mock import MagicMock
from agentforge.monitor import SingleExperimentMonitor


def test_check_nan_detects_nan(tmp_path):
    log_path = tmp_path / "exp.log"
    log_path.write_text("step 1 loss=0.5\nstep 2 loss=nan\n")

    mon = SingleExperimentMonitor(
        index=0, log_path=log_path, timeout=3600,
    )
    assert mon.check_nan() is True


def test_check_nan_no_nan(tmp_path):
    log_path = tmp_path / "exp.log"
    log_path.write_text("step 1 loss=0.5\nstep 2 loss=0.3\n")

    mon = SingleExperimentMonitor(
        index=0, log_path=log_path, timeout=3600,
    )
    assert mon.check_nan() is False


def test_is_timed_out():
    mon = SingleExperimentMonitor(
        index=0, log_path=Path("/dev/null"), timeout=1,
    )
    mon._start_time = time.time() - 10
    assert mon.is_timed_out() is True


def test_read_new_log_lines(tmp_path):
    log_path = tmp_path / "exp.log"
    log_path.write_text("line1\nline2\n")

    mon = SingleExperimentMonitor(
        index=0, log_path=log_path, timeout=3600,
    )
    lines = mon.read_new_lines()
    assert len(lines) == 2
    assert "line1" in lines[0]

    # Second read returns nothing (no new content)
    lines2 = mon.read_new_lines()
    assert len(lines2) == 0

    # Append new content
    with open(log_path, "a") as f:
        f.write("line3\n")
    lines3 = mon.read_new_lines()
    assert len(lines3) == 1
    assert "line3" in lines3[0]
