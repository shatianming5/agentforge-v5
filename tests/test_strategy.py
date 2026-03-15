from __future__ import annotations
from agentforge.strategy import StrategyValidator
from agentforge.agent import Strategy


def _make_strategies(n, categories=None):
    cats = categories or ["arch", "optim", "data", "reg"]
    return [
        Strategy(f"s{i}", f"b{i}", 0.5, 40, 45, 64, False,
                cats[i % len(cats)], "high" if i < 2 else "low")
        for i in range(n)
    ]


class TestStrategyValidator:
    def test_valid(self):
        assert len(StrategyValidator.validate(_make_strategies(8))) == 0

    def test_insufficient_categories(self):
        strats = _make_strategies(4, categories=["optim", "optim"])
        errors = StrategyValidator.validate(strats)
        assert any("categor" in e.lower() for e in errors)

    def test_insufficient_high_risk(self):
        strats = [Strategy(f"s{i}", f"b{i}", 0.5, 40, 45, 64, False, "cat", "low") for i in range(8)]
        errors = StrategyValidator.validate(strats)
        assert any("risk" in e.lower() for e in errors)

    def test_fingerprint_overlap(self):
        assert StrategyValidator.check_fingerprint_overlap("fp_abc", {"fp_abc": "sgd"}) is True

    def test_no_overlap(self):
        assert StrategyValidator.check_fingerprint_overlap("fp_xyz", {"fp_abc": "sgd"}) is False
