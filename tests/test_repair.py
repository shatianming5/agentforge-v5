from __future__ import annotations
from agentforge.repair import SelfRepair
from agentforge.state import StrategyResult


def _make_results(statuses, errors):
    return [
        StrategyResult(f"exp-{i}", f"s{i}", f"b{i}", 0.0, s, e, 0, 0, 32)
        for i, (s, e) in enumerate(zip(statuses, errors))
    ]


class TestSelfRepair:
    def test_diagnose_environmental(self):
        results = _make_results(["error"]*4, ["OOM"]*4)
        assert SelfRepair.diagnose_all_fail(results) == "environmental"

    def test_diagnose_code_related(self):
        results = _make_results(["error"]*4, ["OOM", "NaN", "Timeout", "OOM"])
        assert SelfRepair.diagnose_all_fail(results) == "code_related"

    def test_all_fail_true(self):
        results = _make_results(["error"]*4, ["OOM"]*4)
        assert SelfRepair.is_all_fail(results) is True

    def test_all_fail_false(self):
        results = _make_results(["ok", "error"], [None, "OOM"])
        assert SelfRepair.is_all_fail(results) is False
