from __future__ import annotations

from agentforge.display import Display, DisplayConfig
from agentforge.state import HardwareInfo, StrategyResult


def _hw_cpu():
    return HardwareInfo("cpu", "", 0, 8, 16, 100)


def _hw_gpu():
    return HardwareInfo("cuda", "A100", 4, 64, 256, 500)


def _results():
    return [
        StrategyResult("exp-0", "adamw_cosine", "b0", 0.95, "ok", None, 40.0, 30.0, 64),
        StrategyResult("exp-1", "lion_high_lr", "b1", 0.0, "oom", "out of memory", 0.0, 0.0, 64),
        StrategyResult("exp-2", "rmsnorm_fast", "b2", 0.88, "ok", None, 38.0, 28.0, 64),
    ]


class TestDisplay:
    def test_create_cpu(self):
        d = Display(_hw_cpu(), N=2)
        assert d.hw.device == "cpu"
        assert d.N == 2

    def test_create_gpu(self):
        d = Display(_hw_gpu(), N=8, cfg=DisplayConfig(direction="minimize"))
        assert d.cfg.direction == "minimize"

    def test_header_no_crash(self, capsys):
        d = Display(_hw_cpu(), N=2)
        d.header()
        out = capsys.readouterr().out
        assert "AgentForge" in out
        assert "cpu" in out

    def test_round_lifecycle_no_crash(self, capsys):
        d = Display(_hw_cpu(), N=2)
        d.round_start(1)
        d.phase1_start()
        d.phase1_done(2.5, 3)
        d.phase2_start(3)
        d.phase2_done(1.2)
        out = capsys.readouterr().out
        assert "Round 1" in out
        assert "3 strategies" in out

    def test_round_results_no_crash(self, capsys):
        d = Display(_hw_cpu(), N=2)
        results = _results()
        d.round_results(results, winners=["exp-0"], best_score=0.95, best_round=1)
        out = capsys.readouterr().out
        assert "adamw_cosine" in out
        assert "0.9500" in out
        assert "oom" in out

    def test_final_summary_no_crash(self, capsys):
        d = Display(_hw_cpu(), N=2)
        d.final_summary("completed", 0.95, 3, 12.5)
        out = capsys.readouterr().out
        assert "completed" in out
        assert "0.950000" in out

    def test_skip_round(self, capsys):
        d = Display(_hw_cpu(), N=2)
        d.skip_round("no branches")
        out = capsys.readouterr().out
        assert "no branches" in out

    def test_warn(self, capsys):
        d = Display(_hw_cpu(), N=2)
        d.warn("something wrong")
        out = capsys.readouterr().out
        assert "WARNING" in out
