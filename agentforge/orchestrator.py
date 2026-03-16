from __future__ import annotations

import math
import shlex
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from agentforge.agent import AgentSession, CodexCLI, OutputParser
from agentforge.analyzer import ProjectAnalyzer
from agentforge.anti_oscillation import AntiOscillation
from agentforge.cleanup import Cleanup
from agentforge.config import ChallengeConfig, is_better, best_initial_score, load_config
from agentforge.data import prepare_data
from agentforge.confirm import InteractiveConfirm
from agentforge.generator import ConfigGenerator
from agentforge.experiment import ExperimentSetup
from agentforge.hardware import HardwareDetector
from agentforge.repair import SelfRepair
from agentforge.runner import ParallelRunner
from agentforge.sandbox import Sandbox
from agentforge.state import (
    BestResult, Budget, Credentials, RoundResult, SessionState,
    StateFile, StrategyRecord, StrategyResult,
)
from agentforge.strategy import StrategyValidator


class Orchestrator:
    def __init__(self, config_path: Path | None, workdir: Path,
                 stop_flag: Callable[[], bool] | None = None):
        self.workdir = workdir
        self.state_file = StateFile(workdir / ".agentforge" / "state.json")
        self._stop_flag = stop_flag or (lambda: False)

        # Auto-setup: 如果没有 config_path 且 workdir 下没有 challenge.yaml
        if config_path is None:
            config_path = self._auto_setup()

        self.config = load_config(config_path)

    def _auto_setup(self) -> Path:
        """自动分析项目并生成配置文件。"""
        challenge_path = self.workdir / "challenge.yaml"
        if challenge_path.exists():
            return challenge_path

        print("[AgentForge] 未找到 challenge.yaml，启动自动配置...")

        # Step 1: 分析项目
        print("[AgentForge] 正在分析项目结构（Codex read-only）...")
        analyzer = ProjectAnalyzer(workdir=self.workdir)
        profile = analyzer.analyze()
        print(f"[AgentForge] 分析完成: {profile.description}")

        # Step 2: 生成配置文件
        generator = ConfigGenerator(profile)
        files = generator.generate_all()

        # Step 3: 交互确认
        confirm = InteractiveConfirm(workdir=self.workdir)
        results = confirm.confirm_each(files)

        rejected = [f for f, r in results.items() if r == "rejected"]
        if "challenge.yaml" in rejected:
            raise RuntimeError("challenge.yaml 被拒绝，无法继续")

        print("[AgentForge] 配置已保存。开始优化...")
        return challenge_path

    def run(self) -> None:
        state = self._init_or_resume()
        while not self._done(state) and not self._stop_flag():
            state = self._run_round(state)
            self.state_file.save(state)
        d = self.config.target_direction
        done = is_better(state.best.score, self.config.target_value, d) or state.best.score == self.config.target_value
        state.status = "completed" if done else "paused"
        self.state_file.save(state)

    def _init_or_resume(self) -> SessionState:
        if self.state_file.exists():
            return self.state_file.load()
        hw = HardwareDetector.detect()
        N, gpus_per = HardwareDetector.compute_N(hw)
        state = SessionState.create_initial(
            session_id=f"af-{uuid.uuid4().hex[:6]}",
            repo_url=str(self.workdir),
            hardware=hw, N=N, gpus_per_experiment=gpus_per,
            rounds_max=25, gpu_hours_max=200,
            direction=self.config.target_direction,
        )
        # 数据准备
        prepare_data(self.workdir, self.config.data)
        # Snapshot credentials
        state.credentials = self._snapshot_credentials()
        # Snapshot env lockfile
        state.env_lockfile_hash = self._compute_lockfile_hash()
        self.state_file.save(state)
        return state

    def _run_round(self, state: SessionState) -> SessionState:
        state.current_round += 1
        t0 = time.time()

        if AntiOscillation.check_plateau(state.score_trajectory):
            state.hints_pending.append(
                "WARNING: 3+ rounds without improvement. Try fundamentally different approaches."
            )

        # Sandbox: protect read-only files
        sandbox = Sandbox(self.workdir, self.config.read_only)
        sandbox.setup()

        # Phase 1
        agent = AgentSession(self.config, state, self.workdir)
        strategies = agent.develop()

        # Validate strategies
        warnings = StrategyValidator.validate(strategies)
        if warnings:
            state.hints_pending.extend(
                f"Strategy validation: {w}" for w in warnings
            )

        sandbox.teardown()
        t1 = time.time()

        # Cleanup between phases
        Cleanup(self.workdir).between_phases()

        # Phase 2
        experiments = []
        for i, s in enumerate(strategies):
            try:
                exp = ExperimentSetup.create(
                    strategy=s, index=i, repo_path=self.workdir,
                    workdir=self.workdir / ".agentforge" / "runs",
                    hw=state.hardware,
                    train_command=shlex.split(self.config.test_benchmark),
                )
                experiments.append(exp)
            except subprocess.CalledProcessError as e:
                print(f"[AgentForge] 跳过策略 {s.name}: 分支 {s.branch} 不可用")
        if not experiments:
            print("[AgentForge] 所有策略分支均不可用，跳过本轮")
            t2 = time.time()
            round_result = RoundResult(
                round=state.current_round, experiments=[],
                winners=[], phase1_minutes=(t1 - t0) / 60,
                phase2_minutes=(t2 - t1) / 60,
            )
            state.rounds.append(round_result)
            state.budget.rounds_used += 1
            state.current_round += 1
            self.state_file.save(state)
            return state
        runner = ParallelRunner(experiments, self.config, timeout=345600, workdir=self.workdir)
        results = runner.run(N=state.N)
        t2 = time.time()

        # All-fail handling
        if SelfRepair.is_all_fail(results):
            diagnosis = SelfRepair.diagnose_all_fail(results)
            if diagnosis == "environmental":
                SelfRepair.rebuild_venv(self.workdir)
            elif diagnosis == "code_related":
                # Launch new Agent session with error context
                error_context = SelfRepair.collect_error_context(results)
                try:
                    repair_output = CodexCLI.run(
                        prompt=error_context, cwd=self.workdir,
                        timeout=1800, env={"CUDA_VISIBLE_DEVICES": "0"},
                    )
                    repair_strategies = OutputParser.parse(repair_output)
                    if repair_strategies:
                        strategies = repair_strategies
                except (RuntimeError, ValueError):
                    # Repair failed, rollback to best commit
                    if state.best.commit:
                        SelfRepair.rollback_to_commit(self.workdir, state.best.commit)

        # Select winners
        d = self.config.target_direction
        winners = self.select_winners(results, d)

        # Update state
        round_result = RoundResult(
            round=state.current_round, experiments=results,
            winners=winners,
            phase1_minutes=(t1 - t0) / 60,
            phase2_minutes=(t2 - t1) / 60,
        )
        state.rounds.append(round_result)

        for r in results:
            if r.status == "ok" and r.score > 0 and is_better(r.score, state.best.score, d):
                commit = self._get_branch_commit(r.branch)
                state.best = BestResult(
                    score=r.score, round=state.current_round,
                    experiment=r.id, commit=commit, checkpoint="",
                )

        ok_scores = [r.score for r in results if r.status == "ok" and r.score > 0]
        if ok_scores:
            best_this_round = min(ok_scores) if d == "minimize" else max(ok_scores)
        else:
            best_this_round = best_initial_score(d)
        state.score_trajectory.append(best_this_round)

        for r in results:
            outcome = "ok" if r.status == "ok" else r.status
            state.strategies_tried.append(
                StrategyRecord(r.strategy, state.current_round, r.score, outcome)
            )

        state.budget.rounds_used += 1
        round_hours = (t2 - t0) / 3600 * max(state.hardware.num_gpus, 1)
        state.budget.gpu_hours_used += round_hours
        state.hints_pending.clear()

        loser_workdirs = [e.workdir for e in experiments if e.strategy.name not in winners]
        Cleanup(self.workdir).delete_loser_workdirs(loser_workdirs)

        return state

    def _done(self, state: SessionState) -> bool:
        return state.is_done(self.config.target_value, self.config.target_direction)

    def _get_branch_commit(self, branch: str) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", branch],
                cwd=str(self.workdir), capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""

    def _snapshot_credentials(self) -> Credentials:
        """Detect git credential method at startup."""
        method = "none"
        try:
            result = subprocess.run(
                ["git", "config", "remote.origin.url"],
                cwd=str(self.workdir), capture_output=True, text=True, timeout=5,
            )
            url = result.stdout.strip()
            if url.startswith("git@") or url.startswith("ssh://"):
                method = "ssh_key"
            elif url.startswith("https://"):
                method = "https_token"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return Credentials(
            git_method=method,
            last_check=datetime.now(timezone.utc).isoformat(),
        )

    def _compute_lockfile_hash(self) -> str:
        """Hash requirements.txt or similar lockfile for env reproducibility."""
        import hashlib
        for name in ["requirements.txt", "Pipfile.lock", "poetry.lock"]:
            path = self.workdir / name
            if path.exists():
                content = path.read_bytes()
                return f"sha256:{hashlib.sha256(content).hexdigest()[:16]}"
        return ""

    @staticmethod
    def select_winners(results: list[StrategyResult], direction: str = "maximize") -> list[str]:
        ok_results = [r for r in results if r.status == "ok" and r.score > 0]
        if not ok_results:
            return []
        top_k = max(1, math.ceil(len(results) / 4))
        reverse = (direction != "minimize")
        sorted_results = sorted(ok_results, key=lambda r: r.score, reverse=reverse)
        return [r.id for r in sorted_results[:top_k]]
