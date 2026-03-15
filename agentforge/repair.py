from __future__ import annotations
import subprocess
from pathlib import Path
from agentforge.state import StrategyResult


class SelfRepair:
    @staticmethod
    def is_all_fail(results: list[StrategyResult]) -> bool:
        return all(r.status != "ok" for r in results)

    @staticmethod
    def diagnose_all_fail(results: list[StrategyResult]) -> str:
        errors = [r.error for r in results if r.error is not None]
        if not errors:
            return "unknown"
        if len(set(errors)) == 1:
            return "environmental"
        return "code_related"

    @staticmethod
    def rebuild_venv(workdir: Path) -> None:
        import shutil
        venv_dir = workdir / ".agentforge" / "venv"
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
        subprocess.run(["python", "-m", "venv", str(venv_dir)],
                      check=True, capture_output=True)

    @staticmethod
    def rollback_to_commit(repo_path: Path, commit: str) -> None:
        subprocess.run(["git", "checkout", commit],
                      cwd=str(repo_path), check=True, capture_output=True)
