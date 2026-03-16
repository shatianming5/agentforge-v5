from __future__ import annotations

import json
import os
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
            f"  2. Run a VERY SHORT trial (max_iters=50, eval_interval=25) to verify it works\n"
            f"     IMPORTANT: Keep trials SHORT (<30 seconds). Do NOT run full training.\n"
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
        rnd = state.current_round + 1
        return (
            f"RULES:\n"
            f"  - Do NOT modify read-only files: {ro}\n"
            f"  - Produce at least {min(state.N, 3)} different categories of strategies\n"
            f"  - At least {min(state.N, 2)} must be high-risk/high-reward\n"
            f"  - Each strategy gets its own Git branch: agentforge/iter-{rnd}/exp-{{i}}\n"
            f"  - For each strategy: create the branch, commit code, run a quick trial to verify\n"
            f"  - Return to the main branch when done\n\n"
            f"OUTPUT (CRITICAL — you MUST do this as your FINAL step):\n"
            f"  mkdir -p .agentforge\n"
            f"  Write a JSON array to the file: .agentforge/agent_output.json\n"
            f"  Each element must have these fields:\n"
            f'  {{\n'
            f'    "name": "descriptive_name",\n'
            f'    "branch": "agentforge/iter-{rnd}/exp-0",\n'
            f'    "confidence": 0.8,\n'
            f'    "measured_vram_gb": 0,\n'
            f'    "measured_epoch_seconds": 5,\n'
            f'    "batch_size": 1,\n'
            f'    "resume_checkpoint": false,\n'
            f'    "category": "optim",\n'
            f'    "risk": "high",\n'
            f'    "train_command": "python3 train.py"\n'
            f'  }}\n'
            f"  category must be one of: optim, arch, data, reg\n"
            f"  risk must be: high or low\n"
            f"  train_command: the shell command to run full training for this strategy"
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
        if not isinstance(data, list) or not data:
            raise ValueError("Agent summary must be a non-empty JSON array")
        strategies = []
        for d in data:
            if "name" not in d or "branch" not in d:
                continue  # Skip entries without required fields
            strategies.append(Strategy(
                name=d["name"],
                branch=d["branch"],
                confidence=float(d.get("confidence", 0.5)),
                measured_vram_gb=float(d.get("measured_vram_gb", 0.0)),
                measured_epoch_seconds=float(d.get("measured_epoch_seconds", 0.0)),
                batch_size=int(d.get("batch_size", 2)),
                resume_checkpoint=bool(d.get("resume_checkpoint", False)),
                category=d.get("category", "unknown"),
                risk=d.get("risk", "medium"),
                train_command=d.get("train_command", ""),
            ))
        if not strategies:
            raise ValueError("No valid strategies found in Agent output")
        return strategies


class CodexCLI:
    @staticmethod
    def _find_summary_json(agentforge_dir: Path) -> str | None:
        """Search .agentforge/ for any JSON file containing strategy data."""
        if not agentforge_dir.is_dir():
            return None
        # Try exact name first, then any JSON
        candidates = []
        exact = agentforge_dir / "agent_output.json"
        if exact.exists():
            candidates.insert(0, exact)
        for f in sorted(agentforge_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
            if f.name != "state.json" and f not in candidates:
                candidates.append(f)
        for f in candidates:
            try:
                content = f.read_text().strip()
                if not content:
                    continue
                data = json.loads(content)
                # Accept: direct array, or object with "strategies" key
                if isinstance(data, list):
                    return content
                if isinstance(data, dict) and "strategies" in data:
                    return json.dumps(data["strategies"])
                return None
            except (json.JSONDecodeError, OSError):
                continue
        return None

    @staticmethod
    def run(prompt: str, cwd: Path, timeout: int, env: dict) -> str:
        merged_env = {**os.environ, **(env or {})}
        agentforge_dir = Path(cwd) / ".agentforge"
        # Use pre-seeded agent_output.json if present (skip Codex)
        agent_output_path = agentforge_dir / "agent_output.json"
        if agent_output_path.exists():
            try:
                raw = agent_output_path.read_text().strip()
                data = json.loads(raw)
                content = None
                if isinstance(data, list) and data:
                    content = raw
                elif isinstance(data, dict) and "strategies" in data:
                    content = json.dumps(data["strategies"])
                if content:
                    agent_output_path.unlink()  # consume so next round runs Codex
                    return f"AGENTFORGE_SUMMARY_BEGIN\n{content}\nAGENTFORGE_SUMMARY_END"
            except (json.JSONDecodeError, OSError):
                pass
        try:
            result = subprocess.run(
                ["codex", "exec", "--full-auto", prompt],
                cwd=str(cwd), capture_output=True, text=True,
                timeout=timeout, env=merged_env,
            )
        except subprocess.TimeoutExpired:
            content = CodexCLI._find_summary_json(agentforge_dir)
            if content:
                return f"AGENTFORGE_SUMMARY_BEGIN\n{content}\nAGENTFORGE_SUMMARY_END"
            raise
        # Primary: read structured output from any JSON in .agentforge/
        content = CodexCLI._find_summary_json(agentforge_dir)
        if content:
            return f"AGENTFORGE_SUMMARY_BEGIN\n{content}\nAGENTFORGE_SUMMARY_END"
        # Fallback: look for markers in stdout
        stdout = result.stdout or ""
        if "AGENTFORGE_SUMMARY_BEGIN" in stdout:
            return stdout
        if result.returncode != 0:
            raise RuntimeError(
                f"Codex CLI failed (exit {result.returncode}): {(result.stderr or '')[:500]}")
        raise RuntimeError("Codex produced no strategy output")


class BranchDetector:
    """Fallback: detect strategies from git branches if Codex output parsing fails."""

    @staticmethod
    def detect(workdir: Path, current_round: int) -> list[Strategy]:
        prefix = f"agentforge/iter-{current_round}/exp-"
        try:
            result = subprocess.run(
                ["git", "branch", "--list", f"{prefix}*"],
                cwd=str(workdir), capture_output=True, text=True, timeout=10,
            )
            branches = [
                b.strip().lstrip("* ")
                for b in result.stdout.strip().split("\n")
                if b.strip()
            ]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []
        strategies = []
        for branch in sorted(branches):
            # Only include branches with actual commits beyond main
            try:
                diff = subprocess.run(
                    ["git", "log", f"main..{branch}", "--oneline"],
                    cwd=str(workdir), capture_output=True, text=True, timeout=10,
                )
                if not diff.stdout.strip():
                    continue  # Empty branch, skip
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
            strategies.append(Strategy(
                name=f"auto-{branch.split('/')[-1]}",
                branch=branch,
                confidence=0.5,
                measured_vram_gb=0.0,
                measured_epoch_seconds=0.0,
                batch_size=2,
                resume_checkpoint=False,
                category="auto-detected",
                risk="medium",
                train_command="",
            ))
        return strategies


class AgentSession:
    def __init__(self, config, state, workdir: Path):
        self.config = config
        self.state = state
        self.workdir = workdir

    def develop(self) -> list[Strategy]:
        prompt = PromptBuilder.build(self.config, self.state)
        first_gpu = os.environ.get(
            "CUDA_VISIBLE_DEVICES", "0"
        ).split(",")[0].strip()
        try:
            raw_output = CodexCLI.run(
                prompt=prompt, cwd=self.workdir,
                timeout=43200,  # 12 hours
                env={"CUDA_VISIBLE_DEVICES": first_gpu},
            )
            return OutputParser.parse(raw_output)
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as e:
            # Fallback: detect branches created by Codex
            print(f"[AgentForge] Codex output parsing failed: {e}")
            print("[AgentForge] Falling back to branch detection...")
            strategies = BranchDetector.detect(
                self.workdir, self.state.current_round
            )
            if strategies:
                print(f"[AgentForge] Found {len(strategies)} branches via fallback")
                return strategies
            raise
