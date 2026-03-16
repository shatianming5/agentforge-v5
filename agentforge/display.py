from __future__ import annotations

import sys
from dataclasses import dataclass

from agentforge.state import HardwareInfo, StrategyResult


def _is_tty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


@dataclass
class DisplayConfig:
    agent_name: str = "Codex"
    direction: str = "minimize"
    target_metric: str = "score"
    target_value: float = 0.0


class Display:
    """Orchestrator 输出层。有 rich 用 rich，没有用 print。"""

    def __init__(self, hw: HardwareInfo, N: int, cfg: DisplayConfig | None = None):
        self.hw = hw
        self.N = N
        self.cfg = cfg or DisplayConfig()
        self._use_rich = _HAS_RICH and _is_tty()
        if self._use_rich:
            self._console = Console()

    # ── Header ──────────────────────────────────────────

    def header(self) -> None:
        hw_desc = (
            f"{self.hw.num_gpus}x {self.hw.gpu_model}"
            if self.hw.device == "cuda"
            else f"cpu, {self.hw.cpu_cores} cores"
        )
        line = (
            f"AgentForge v5.1 | Agent: {self.cfg.agent_name} | "
            f"Hardware: {hw_desc} | N={self.N}"
        )
        if self._use_rich:
            self._console.rule(line, style="bold cyan")
        else:
            print(f"\n{'=' * 60}")
            print(f"  {line}")
            print(f"{'=' * 60}\n")

    # ── Round ───────────────────────────────────────────

    def round_start(self, round_num: int) -> None:
        if self._use_rich:
            self._console.rule(f"Round {round_num}", style="bold")
        else:
            print(f"\n--- Round {round_num} {'─' * 40}")

    def phase1_start(self) -> None:
        self._print("  Phase 1: Agent 生成策略...")

    def phase1_done(self, minutes: float, num_strategies: int) -> None:
        self._print(
            f"  Phase 1: Agent 生成策略... done "
            f"({minutes:.1f}m, {num_strategies} strategies)"
        )

    def phase2_start(self, num_experiments: int) -> None:
        self._print(f"  Phase 2: 并行训练 {num_experiments} 个实验...")

    def phase2_done(self, minutes: float) -> None:
        self._print(f"  Phase 2: 训练完成 ({minutes:.1f}m)")

    # ── Results Table ───────────────────────────────────

    def round_results(
        self,
        results: list[StrategyResult],
        winners: list[str],
        best_score: float,
        best_round: int,
    ) -> None:
        if self._use_rich:
            self._rich_table(results, winners, best_score, best_round)
        else:
            self._plain_table(results, winners, best_score, best_round)

    def _rich_table(
        self,
        results: list[StrategyResult],
        winners: list[str],
        best_score: float,
        best_round: int,
    ) -> None:
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=4)
        table.add_column("Strategy", min_width=20)
        table.add_column("Score", justify="right", width=10)
        table.add_column("Status", width=12)

        for r in results:
            is_winner = r.id in winners
            status_str = self._status_str(r.status, is_winner)
            score_str = f"{r.score:.4f}" if r.score > 0 else "—"
            idx = r.id.replace("exp-", "")
            name = r.strategy
            if is_winner:
                name = f"[bold green]{name}[/bold green]"
                score_str = f"[bold green]{score_str}[/bold green]"
            table.add_row(idx, name, score_str, status_str)

        self._console.print(table)
        self._console.print(
            f"  Best so far: [bold]{best_score:.4f}[/bold] (round {best_round})"
        )

    def _plain_table(
        self,
        results: list[StrategyResult],
        winners: list[str],
        best_score: float,
        best_round: int,
    ) -> None:
        print(f"\n  {'#':<4} {'Strategy':<24} {'Score':>10} {'Status':<12}")
        print(f"  {'─'*4} {'─'*24} {'─'*10} {'─'*12}")
        for r in results:
            is_winner = r.id in winners
            status_str = self._status_str(r.status, is_winner)
            score_str = f"{r.score:.4f}" if r.score > 0 else "—"
            idx = r.id.replace("exp-", "")
            marker = " *" if is_winner else ""
            print(f"  {idx:<4} {r.strategy:<24} {score_str:>10} {status_str}{marker}")
        print(f"\n  Best so far: {best_score:.4f} (round {best_round})")

    # ── Final Summary ───────────────────────────────────

    def final_summary(self, status: str, best_score: float, rounds: int,
                      elapsed_minutes: float) -> None:
        if self._use_rich:
            self._console.rule("RESULTS", style="bold green" if status == "completed" else "bold yellow")
            self._console.print(f"  Status:     {status}")
            self._console.print(f"  Best score: [bold]{best_score:.6f}[/bold]")
            self._console.print(f"  Rounds:     {rounds}")
            self._console.print(f"  Time:       {elapsed_minutes:.1f}m")
        else:
            print(f"\n{'=' * 60}")
            print(f"  RESULTS")
            print(f"{'=' * 60}")
            print(f"  Status:     {status}")
            print(f"  Best score: {best_score:.6f}")
            print(f"  Rounds:     {rounds}")
            print(f"  Time:       {elapsed_minutes:.1f}m")
            print(f"{'=' * 60}")

    # ── Helpers ─────────────────────────────────────────

    def skip_round(self, reason: str) -> None:
        self._print(f"  Skipped: {reason}")

    def warn(self, msg: str) -> None:
        if self._use_rich:
            self._console.print(f"  [yellow]WARNING:[/yellow] {msg}")
        else:
            print(f"  WARNING: {msg}")

    @staticmethod
    def _status_str(status: str, is_winner: bool) -> str:
        if status == "ok":
            return "winner" if is_winner else "ok"
        return status

    def _print(self, msg: str) -> None:
        if self._use_rich:
            self._console.print(msg)
        else:
            print(msg)
