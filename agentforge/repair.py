from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

from agentforge.state import StrategyResult


class SelfRepair:
    MAX_REPAIR_ATTEMPTS = 2

    @staticmethod
    def is_all_fail(results: list[StrategyResult]) -> bool:
        return all(r.status != "ok" for r in results)

    @staticmethod
    def has_scoring_failures(results: list[StrategyResult]) -> bool:
        """Training succeeded (status ok) but score=0.0 — scoring pipeline broken."""
        return any(r.status == "ok" and r.score == 0.0 for r in results)

    @staticmethod
    def diagnose_all_fail(results: list[StrategyResult]) -> str:
        errors = [r.error for r in results if r.error is not None]
        if not errors:
            return "unknown"
        if len(set(errors)) == 1:
            return "environmental"
        return "code_related"

    @staticmethod
    def diagnose_scoring(exp_workdir: Path, config) -> tuple[str, str]:
        """Determine why scoring failed for a successful training.

        Returns (failure_type, error_detail):
          - "test_failed":     test_suite.py crashed
          - "benchmark_error": benchmark.py crashed
          - "no_results":      benchmark.json not created or metric missing
          - "unknown":         can't determine
        """
        from agentforge.config import ChallengeConfig

        from agentforge.stream import stream_run
        # 1. Try test suite
        try:
            stream_run(
                shlex.split(config.test_full),
                cwd=exp_workdir, timeout=120, prefix="Repair-Test", check=True,
            )
        except subprocess.CalledProcessError as e:
            return "test_failed", (e.output or "")[:500]
        except subprocess.TimeoutExpired:
            return "test_failed", "test timed out"

        # 2. Try benchmark
        try:
            stream_run(
                shlex.split(config.test_benchmark),
                cwd=exp_workdir, timeout=120, prefix="Repair-Bench", check=True,
            )
        except subprocess.CalledProcessError as e:
            return "benchmark_error", (e.output or "")[:500]
        except subprocess.TimeoutExpired:
            return "benchmark_error", "benchmark timed out"

        # 3. Check results file
        results_path = exp_workdir / "results" / "benchmark.json"
        if not results_path.exists():
            return "no_results", "results/benchmark.json was not created"

        import json
        try:
            data = json.loads(results_path.read_text())
            if config.target_metric not in data:
                return "no_results", f"metric '{config.target_metric}' not in {data}"
        except (json.JSONDecodeError, ValueError) as e:
            return "no_results", f"bad JSON: {e}"

        return "unknown", ""

    @staticmethod
    def build_repair_prompt(
        failure_type: str,
        error_detail: str,
        workdir: Path,
        config,
    ) -> str:
        """Build Codex prompt to fix scoring pipeline."""
        lines = []

        if failure_type == "benchmark_error" or failure_type == "no_results":
            target_file = "benchmark.py"
            source = (workdir / "benchmark.py").read_text() if (workdir / "benchmark.py").exists() else "(missing)"
        elif failure_type == "test_failed":
            target_file = "test_suite.py"
            source = (workdir / "test_suite.py").read_text() if (workdir / "test_suite.py").exists() else "(missing)"
        else:
            return ""

        # Collect training output directory listing
        out_listing = ""
        for d in sorted(workdir.iterdir()):
            if d.name.startswith("out") and d.is_dir():
                files = [f.name for f in sorted(d.iterdir())]
                out_listing += f"  {d.name}/: {files}\n"

        # Collect training log (last 30 lines)
        log_tail = ""
        log_dir = workdir / ".agentforge" / "runs" / "logs"
        if log_dir.is_dir():
            for log_file in sorted(log_dir.glob("*.log")):
                text = log_file.read_text()
                last_lines = text.strip().split("\n")[-30:]
                log_tail += f"\n--- {log_file.name} ---\n" + "\n".join(last_lines)

        lines.append(f"{target_file} failed during scoring.")
        lines.append(f"Error: {error_detail}")
        lines.append("")
        lines.append(f"Current {target_file}:")
        lines.append(source)
        lines.append("")
        lines.append(f"Target metric: {config.target_metric} (direction: {config.target_direction})")
        lines.append(f"Benchmark command: {config.test_benchmark}")
        lines.append(f"Expected output: results/benchmark.json with {{\"{config.target_metric}\": <value>}}")
        if out_listing:
            lines.append(f"\nTraining output directories:\n{out_listing}")
        if log_tail:
            lines.append(f"\nTraining log (last 30 lines):{log_tail}")
        lines.append(f"\nFix {target_file} so it correctly extracts the metric. Do NOT modify any other file.")

        return "\n".join(lines)

    @staticmethod
    def rebuild_venv(workdir: Path) -> None:
        import shutil
        from agentforge.stream import stream_run
        venv_dir = workdir / ".agentforge" / "venv"
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
        stream_run([sys.executable, "-m", "venv", str(venv_dir)],
                   prefix="Venv", check=True)
        req_file = workdir / "requirements.txt"
        if req_file.exists():
            stream_run(
                [str(venv_dir / "bin" / "pip"), "install", "-r", str(req_file)],
                prefix="Pip", timeout=300,
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
