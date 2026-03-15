from __future__ import annotations
import subprocess
from pathlib import Path
from agentforge.agent import Strategy


class StrategyValidator:
    MIN_CATEGORIES = 3
    MIN_HIGH_RISK_RATIO = 0.25

    @staticmethod
    def validate(strategies: list[Strategy]) -> list[str]:
        errors = []
        if not strategies:
            return ["No strategies provided"]
        categories = {s.category for s in strategies}
        if len(categories) < min(StrategyValidator.MIN_CATEGORIES, len(strategies)):
            errors.append(
                f"Insufficient category diversity: {len(categories)} categories, "
                f"need at least {StrategyValidator.MIN_CATEGORIES}"
            )
        high_risk_count = sum(1 for s in strategies if s.risk == "high")
        min_high_risk = max(2, int(len(strategies) * StrategyValidator.MIN_HIGH_RISK_RATIO))
        if high_risk_count < min_high_risk:
            errors.append(
                f"Insufficient high-risk strategies: {high_risk_count}, need at least {min_high_risk}"
            )
        return errors

    @staticmethod
    def compute_diff_fingerprint(repo_path: Path, branch: str, base: str = "HEAD") -> set[str]:
        """Compute a set of changed lines (fingerprint) for a branch vs base."""
        try:
            result = subprocess.run(
                ["git", "diff", base, branch, "--", "*.py"],
                cwd=str(repo_path), capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return set()
            lines = set()
            for line in result.stdout.split("\n"):
                if line.startswith("+") and not line.startswith("+++"):
                    lines.add(line[1:].strip())
                elif line.startswith("-") and not line.startswith("---"):
                    lines.add(line[1:].strip())
            return lines
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return set()

    @staticmethod
    def check_fingerprint_overlap(new_fp: set[str], tried_fps: list[set[str]],
                                   threshold: float = 0.7) -> bool:
        """Check if new fingerprint overlaps > threshold with any tried fingerprint."""
        if not new_fp:
            return False
        for tried in tried_fps:
            if not tried:
                continue
            intersection = new_fp & tried
            union = new_fp | tried
            if union and len(intersection) / len(union) > threshold:
                return True
        return False
