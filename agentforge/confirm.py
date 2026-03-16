from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


class InteractiveConfirm:
    """交互式逐步确认生成的配置文件。"""

    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)

    def confirm_each(self, files: dict[str, str]) -> dict[str, str]:
        """逐个展示文件内容，让用户确认 Y/n/edit。

        Returns: {filename: "accepted" | "rejected" | "edited"}
        """
        results = {}
        total = len(files)
        for i, (filename, content) in enumerate(files.items(), 1):
            print(f"\n{'━' * 40}")
            print(f"  {i}/{total}: {filename}")
            print(f"{'━' * 40}")
            print(content)
            print(f"{'━' * 40}")

            choice = input("确认？[Y/n/edit] > ").strip().lower()

            if choice in ("", "y", "yes"):
                self._write_file(filename, content)
                results[filename] = "accepted"
            elif choice in ("n", "no"):
                results[filename] = "rejected"
            elif choice == "edit":
                edited = self._open_editor(filename, content)
                if edited is not None:
                    self._write_file(filename, edited)
                    results[filename] = "edited"
                else:
                    results[filename] = "rejected"
            else:
                self._write_file(filename, content)
                results[filename] = "accepted"

        return results

    def _write_file(self, filename: str, content: str) -> None:
        path = self.workdir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def _open_editor(self, filename: str, content: str) -> str | None:
        """打开 $EDITOR 让用户编辑，返回编辑后内容。"""
        editor = os.environ.get("EDITOR", "vi")
        suffix = Path(filename).suffix or ".txt"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, prefix="agentforge_"
        ) as f:
            f.write(content)
            tmp_path = f.name
        try:
            subprocess.run([editor, tmp_path], check=True)
            return Path(tmp_path).read_text()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
