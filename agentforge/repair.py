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
        # Install dependencies from lockfile if it exists
        req_file = workdir / "requirements.txt"
        if req_file.exists():
            subprocess.run(
                [str(venv_dir / "bin" / "pip"), "install", "-r", str(req_file)],
                check=False, capture_output=True, timeout=300,
            )

    @staticmethod
    def rollback_to_commit(repo_path: Path, commit: str) -> None:
        subprocess.run(["git", "checkout", commit],
                      cwd=str(repo_path), check=True, capture_output=True)

    @staticmethod
    def collect_error_context(results: list[StrategyResult]) -> str:
        """Collect error details for a new Agent session to fix code-related failures."""
        lines = [
            "ALL experiments failed during full training.",
            "The 2-epoch trial did not catch this.",
            "Fix the underlying issue and produce corrected strategies.",
            "",
            "Error details:",
        ]
        for r in results:
            lines.append(f"  - {r.id} [{r.strategy}]: {r.error or 'unknown error'}")
        return "\n".join(lines)
