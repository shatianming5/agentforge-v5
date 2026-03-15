from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentforge.config import ChallengeConfig
    from agentforge.state import SessionState


@dataclass
class Strategy:
    name: str
    branch: str
    confidence: float
    measured_vram_gb: float
    measured_epoch_seconds: float
    batch_size: int
    resume_checkpoint: bool
    category: str
    risk: str
    train_command: str = ""  # Agent-provided training command


class PromptBuilder:
    @staticmethod
    def build(config: ChallengeConfig, state: SessionState) -> str:
        sections = [
            PromptBuilder._system_section(config, state),
            PromptBuilder._hardware_section(state),
            PromptBuilder._state_section(state),
            PromptBuilder._last_round_section(state),
            PromptBuilder._compressed_history_section(state),
            PromptBuilder._taboo_section(state),
            PromptBuilder._hints_section(state),
            PromptBuilder._rules_section(config, state),
        ]
        return "\n\n".join(s for s in sections if s)

    @staticmethod
    def _system_section(config, state):
        return (
            f"SYSTEM: You are the Maker agent in an AgentForge optimization session.\n"
            f"You have shell access and 1 GPU (CUDA_VISIBLE_DEVICES=0).\n"
            f"Your job: develop {state.N} different optimization strategies.\n"
            f"For each strategy, you MUST:\n"
            f"  1. Write the code changes\n"
            f"  2. Run a trial (2 epochs) to verify it works\n"
            f"  3. Measure: VRAM usage, loss trend, time per epoch\n"
            f"  4. If it fails, fix it and re-try\n"
            f"  5. Commit the working version to a Git branch\n\n"
            f"CHALLENGE:\n"
            f"  Name: {config.challenge_name}\n"
            f"  Description: {config.challenge_description}\n"
            f"  Target: {config.target_metric} {config.target_direction} {config.target_value}"
        )

    @staticmethod
    def _hardware_section(state):
        hw = state.hardware
        return (
            f"HARDWARE:\n"
            f"  Device: {hw.device}\n"
            f"  GPU: {hw.num_gpus}x {hw.gpu_model}\n"
            f"  CPU: {hw.cpu_cores} cores, {hw.ram_gb}GB RAM\n"
            f"  Disk: {hw.disk_free_gb}GB free"
        )

    @staticmethod
    def _state_section(state):
        trajectory_str = " -> ".join(str(s) for s in state.score_trajectory)
        return (
            f"CURRENT STATE:\n"
            f"  Best score: {state.best.score} (round {state.best.round}, "
            f"experiment {state.best.experiment})\n"
            f"  Score trajectory: {trajectory_str}\n"
            f"  Current round: {state.current_round}"
        )

    @staticmethod
    def _last_round_section(state):
        if not state.rounds:
            return ""
        last = state.rounds[-1]
        lines = [f"LAST ROUND RESULTS (round {last.round}, {len(last.experiments)} experiments):"]
        for e in last.experiments:
            line = f"  - {e.id} [{e.strategy}]: score={e.score}, status={e.status}"
            if e.actual_vram_gb > 0:
                line += f", vram={e.actual_vram_gb:.1f}GB"
            if e.actual_epoch_seconds > 0:
                line += f", epoch_time={e.actual_epoch_seconds:.0f}s"
            if e.error:
                line += f", error={e.error}"
            lines.append(line)
        lines.append(f"  Winners: {', '.join(last.winners) if last.winners else 'none'}")
        return "\n".join(lines)

    @staticmethod
    def _compressed_history_section(state):
        if len(state.rounds) <= 1:
            return ""
        lines = ["COMPRESSED HISTORY:"]
        for r in state.rounds[:-1]:
            best_exp = max(r.experiments, key=lambda e: e.score) if r.experiments else None
            best_info = f"best={best_exp.score:.2f} ({best_exp.strategy})" if best_exp else "no results"
            ok_count = sum(1 for e in r.experiments if e.status == "ok")
            fail_count = len(r.experiments) - ok_count
            lines.append(
                f"  Round {r.round}: {best_info}, {ok_count} ok / {fail_count} failed, "
                f"phase1={r.phase1_minutes:.0f}m, phase2={r.phase2_minutes:.0f}m"
            )
        return "\n".join(lines)

    @staticmethod
    def _taboo_section(state):
        if not state.strategies_tried:
            return ""
        lines = ["STRATEGIES TRIED (do not repeat):"]
        for s in state.strategies_tried:
            lines.append(f"  - {s.name} (round {s.round}, score {s.score}, {s.outcome})")
        return "\n".join(lines)

    @staticmethod
    def _hints_section(state):
        if not state.hints_pending:
            return ""
        lines = ["HIGH-PRIORITY SUGGESTIONS FROM HUMAN:"]
        for h in state.hints_pending:
            lines.append(f"  - {h}")
        return "\n".join(lines)

    @staticmethod
    def _rules_section(config, state):
        ro = ", ".join(config.read_only)
        return (
            f"RULES:\n"
            f"  - Do NOT modify read-only files: {ro}\n"
            f"  - Produce at least 3 different categories of strategies\n"
            f"  - At least 2 must be high-risk/high-reward\n"
            f"  - Each strategy gets its own Git branch: agentforge/iter-{{round}}/exp-{{i}}\n"
            f"  - After developing all strategies, output a summary JSON to stdout"
        )


class OutputParser:
    BEGIN_MARKER = "AGENTFORGE_SUMMARY_BEGIN"
    END_MARKER = "AGENTFORGE_SUMMARY_END"

    @staticmethod
    def parse(raw: str) -> list[Strategy]:
        begin = raw.find(OutputParser.BEGIN_MARKER)
        end = raw.find(OutputParser.END_MARKER)
        if begin == -1 or end == -1:
            raise ValueError("No summary found in Agent output")
        json_str = raw[begin + len(OutputParser.BEGIN_MARKER):end].strip()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in Agent summary: {e}") from e
        return [
            Strategy(
                name=d["name"], branch=d["branch"], confidence=d["confidence"],
                measured_vram_gb=d["measured_vram_gb"],
                measured_epoch_seconds=d["measured_epoch_seconds"],
                batch_size=d["batch_size"],
                resume_checkpoint=d["resume_checkpoint"],
                category=d.get("category", "unknown"),
                risk=d.get("risk", "low"),
                train_command=d.get("train_command", ""),
            )
            for d in data
        ]


class CodexCLI:
    @staticmethod
    def run(prompt: str, cwd: Path, timeout: int, env: dict) -> str:
        import os as _os
        merged_env = {**_os.environ, **env} if env else None
        result = subprocess.run(
            ["codex", "--approval-policy", "auto-edit", "--quiet", "-p", prompt],
            cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout, env=merged_env,
        )
        if result.returncode != 0 and not result.stdout.strip():
            raise RuntimeError(f"Codex CLI failed (exit {result.returncode}): {result.stderr[:500]}")
        return result.stdout


class AgentSession:
    def __init__(self, config, state, workdir: Path):
        self.config = config
        self.state = state
        self.workdir = workdir

    def develop(self) -> list[Strategy]:
        prompt = PromptBuilder.build(self.config, self.state)
        raw_output = CodexCLI.run(
            prompt=prompt, cwd=self.workdir,
            timeout=1800, env={"CUDA_VISIBLE_DEVICES": "0"},
        )
        return OutputParser.parse(raw_output)
