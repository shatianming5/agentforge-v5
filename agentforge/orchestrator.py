from __future__ import annotations

import math
import shlex
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from agentforge.agent import AgentSession, CodexCLI, OutputParser, Strategy
from agentforge.pipeline import PipelineOrchestrator
from agentforge.analyzer import ProjectAnalyzer
from agentforge.anti_oscillation import AntiOscillation
from agentforge.cleanup import Cleanup
from agentforge.config import ChallengeConfig, is_better, best_initial_score, load_config
from agentforge.data import prepare_data
from agentforge.display import Display, DisplayConfig
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
        self.display = Display(
            hw=state.hardware, N=state.N,
            cfg=DisplayConfig(
                direction=self.config.target_direction,
                target_metric=self.config.target_metric,
                target_value=self.config.target_value,
            ),
        )
        self.display.header()
        t_start = time.time()
        while not self._done(state) and not self._stop_flag():
            state = self._run_round(state)
            self.state_file.save(state)
        d = self.config.target_direction
        done = is_better(state.best.score, self.config.target_value, d) or state.best.score == self.config.target_value
        state.status = "completed" if done else "paused"
        self.state_file.save(state)
        elapsed = (time.time() - t_start) / 60
        self.display.final_summary(state.status, state.best.score,
                                   state.budget.rounds_used, elapsed)

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
        self.display.round_start(state.current_round)

        if AntiOscillation.check_plateau(state.score_trajectory):
            state.hints_pending.append(
                "WARNING: 3+ rounds without improvement. Try fundamentally different approaches."
            )
            self.display.warn("3+ rounds without improvement")

        # Sandbox: protect read-only files
        sandbox = Sandbox(self.workdir, self.config.read_only)
        sandbox.setup()

        # Phase 1: Generate strategy specs only (no code)
        self.display.phase1_start()
        agent = AgentSession(self.config, state, self.workdir)
        try:
            specs = agent.develop_specs()
        except (RuntimeError, ValueError) as e:
            self.display.warn(f"策略规格生成失败: {e}")
            sandbox.teardown()
            return state

        # Validate spec diversity
        pseudo_strategies = [
            Strategy(
                name=s.name, branch="", confidence=0.5,
                measured_vram_gb=0, measured_epoch_seconds=0,
                batch_size=1, resume_checkpoint=False,
                category=s.category, risk=s.risk,
            )
            for s in specs
        ]
        warnings = StrategyValidator.validate(pseudo_strategies)
        if warnings:
            state.hints_pending.extend(
                f"Strategy validation: {w}" for w in warnings
            )
            for w in warnings:
                self.display.warn(w)

        sandbox.teardown()
        t1 = time.time()
        self.display.phase1_done((t1 - t0) / 60, len(specs))

        # Cleanup between phases
        Cleanup(self.workdir).between_phases()

        # Build config context for workers
        config_context = (
            f"Challenge: {self.config.challenge_name}\n"
            f"Description: {self.config.challenge_description}\n"
            f"Target: {self.config.target_metric} {self.config.target_direction} "
            f"{self.config.target_value}\n"
            f"Read-only: {', '.join(self.config.read_only)}"
        )

        # Pipeline Phase: parallel implement -> train -> score
        self.display.phase2_start(len(specs))
        pipeline = PipelineOrchestrator(
            specs=specs,
            config=self.config,
            hw=state.hardware,
            round_num=state.current_round,
            workdir=self.workdir,
            timeout=345600,
            config_context=config_context,
        )

        # Try to setup live progress display (LiveProgressDisplay may not be available yet)
        live_display = None
        try:
            from agentforge.display import LiveProgressDisplay
            live_display = LiveProgressDisplay(num_workers=len(specs))
            pipeline.on_event(live_display.handle_event)
            live_display.start()
        except (ImportError, Exception):
            pass

        try:
            results = pipeline.run()
        finally:
            if live_display:
                live_display.stop()

        t2 = time.time()
        self.display.phase2_done((t2 - t1) / 60)

        # All-fail handling
        if SelfRepair.is_all_fail(results):
            diagnosis = SelfRepair.diagnose_all_fail(results)
            if diagnosis == "environmental":
                SelfRepair.rebuild_venv(self.workdir)

        # Select winners
        d = self.config.target_direction
        winners = self.select_winners(results, d)

        # Display results table
        self.display.round_results(
            results, winners, state.best.score, state.best.round,
        )

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

    def _repair_scoring(
        self,
        results: list[StrategyResult],
        experiments,
        state: SessionState,
    ) -> list[StrategyResult]:
        """Auto-repair scoring pipeline: diagnose failure, let Codex fix, re-score."""
        # Find first experiment with scoring failure to diagnose
        failed_exp = None
        for r, exp in zip(results, experiments):
            if r.status == "ok" and r.score == 0.0:
                failed_exp = exp
                break
        if failed_exp is None:
            return results

        for attempt in range(SelfRepair.MAX_REPAIR_ATTEMPTS):
            failure_type, error_detail = SelfRepair.diagnose_scoring(
                failed_exp.workdir, self.config,
            )
            if failure_type == "unknown":
                break

            self.display.warn(
                f"评分失败 ({failure_type}), 自动修复 #{attempt + 1}..."
            )
            prompt = SelfRepair.build_repair_prompt(
                failure_type, error_detail, self.workdir, self.config,
            )
            if not prompt:
                break

            try:
                CodexCLI.run(
                    prompt=prompt, cwd=self.workdir,
                    timeout=300, env={"CUDA_VISIBLE_DEVICES": "0"},
                )
            except (RuntimeError, subprocess.TimeoutExpired):
                self.display.warn("修复尝试失败")
                break

            # Re-score all failed experiments
            new_results = []
            for r, exp in zip(results, experiments):
                if r.status == "ok" and r.score == 0.0:
                    from agentforge.scorer import Scorer
                    new_score = Scorer.score(exp, self.config, 0, N=state.N)
                    new_results.append(StrategyResult(
                        id=r.id, strategy=r.strategy, branch=r.branch,
                        score=new_score, status=r.status, error=r.error,
                        actual_vram_gb=r.actual_vram_gb,
                        actual_epoch_seconds=r.actual_epoch_seconds,
                        actual_batch_size=r.actual_batch_size,
                    ))
                else:
                    new_results.append(r)
            results = new_results

            if not SelfRepair.has_scoring_failures(results):
                self.display.warn("修复成功")
                break

        return results

    @staticmethod
    def select_winners(results: list[StrategyResult], direction: str = "maximize") -> list[str]:
        ok_results = [r for r in results if r.status == "ok" and r.score > 0]
        if not ok_results:
            return []
        top_k = max(1, math.ceil(len(results) / 4))
        reverse = (direction != "minimize")
        sorted_results = sorted(ok_results, key=lambda r: r.score, reverse=reverse)
        return [r.id for r in sorted_results[:top_k]]
