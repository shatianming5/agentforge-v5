from __future__ import annotations
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
    def check_fingerprint_overlap(new_fp, tried_fps, threshold=0.7):
        return new_fp in tried_fps
