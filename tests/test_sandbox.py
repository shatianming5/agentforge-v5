from __future__ import annotations
import os
import stat
from pathlib import Path
from agentforge.sandbox import Sandbox


class TestSandbox:
    def test_make_readonly(self, tmp_path):
        target = tmp_path / "protected"
        target.mkdir()
        (target / "file.py").write_text("pass")
        sb = Sandbox(workdir=tmp_path, read_only=["protected"])
        sb.setup()
        mode = os.stat(target / "file.py").st_mode
        assert not (mode & stat.S_IWUSR)
        sb.teardown()
        mode = os.stat(target / "file.py").st_mode
        assert mode & stat.S_IWUSR

    def test_teardown_restores(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        orig = os.stat(f).st_mode
        sb = Sandbox(tmp_path, [])
        sb._saved_permissions[str(f)] = orig
        os.chmod(f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        sb.teardown()
        assert os.stat(f).st_mode == orig
