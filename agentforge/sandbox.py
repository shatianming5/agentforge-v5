from __future__ import annotations
import os
import stat
from pathlib import Path


class Sandbox:
    def __init__(self, workdir: Path, read_only: list[str]):
        self.workdir = workdir
        self.read_only = read_only
        self._saved_permissions: dict[str, int] = {}

    def setup(self):
        self._protect_readonly()

    def teardown(self):
        self._restore_permissions()

    def _protect_readonly(self):
        for pattern in self.read_only:
            target = self.workdir / pattern
            if target.is_dir():
                for f in target.rglob("*"):
                    if f.is_file():
                        self._make_readonly(f)
            elif target.is_file():
                self._make_readonly(target)

    def _make_readonly(self, path):
        current = os.stat(path).st_mode
        self._saved_permissions[str(path)] = current
        readonly = current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        os.chmod(path, readonly)

    def _restore_permissions(self):
        for path_str, mode in self._saved_permissions.items():
            try:
                os.chmod(path_str, mode)
            except OSError:
                pass
        self._saved_permissions.clear()
