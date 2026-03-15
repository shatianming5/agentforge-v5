from __future__ import annotations
from pathlib import Path
from agentforge.daemon import Daemon


class TestDaemon:
    def test_pid_path(self, tmp_path):
        d = Daemon(config_path=tmp_path / "af.yaml", workdir=tmp_path)
        assert d.pid_path == tmp_path / ".agentforge" / "daemon.pid"

    def test_write_and_read_pid(self, tmp_path):
        d = Daemon(config_path=tmp_path / "af.yaml", workdir=tmp_path)
        (tmp_path / ".agentforge").mkdir(exist_ok=True)
        d._write_pid(12345)
        assert d.read_pid() == 12345

    def test_read_pid_missing(self, tmp_path):
        d = Daemon(config_path=tmp_path / "af.yaml", workdir=tmp_path)
        assert d.read_pid() is None

    def test_is_running_false(self, tmp_path):
        d = Daemon(config_path=tmp_path / "af.yaml", workdir=tmp_path)
        assert d.is_running() is False

    def test_cleanup_pid(self, tmp_path):
        d = Daemon(config_path=tmp_path / "af.yaml", workdir=tmp_path)
        (tmp_path / ".agentforge").mkdir(exist_ok=True)
        d._write_pid(99999)
        assert d.pid_path.exists()
        d._cleanup_pid()
        assert not d.pid_path.exists()
