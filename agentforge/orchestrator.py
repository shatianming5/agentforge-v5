from __future__ import annotations

import math
import time
import uuid
from pathlib import Path

from agentforge.agent import AgentSession
from agentforge.anti_oscillation import AntiOscillation
from agentforge.cleanup import Cleanup
from agentforge.config import ChallengeConfig, load_config
from agentforge.experiment import ExperimentSetup
from agentforge.hardware import HardwareDetector
from agentforge.repair import SelfRepair
from agentforge.runner import ParallelRunner
from agentforge.state import (
    BestResult, Budget, RoundResult, SessionState,
    StateFile, StrategyRecord, StrategyResult,
)


class Orchestrator:
    def __init__(self, config_path: Path, workdir: Path):
        self.config = load_config(config_path)
        self.workdir = workdir
        self.state_file = StateFile(workdir / ".agentforge" / "state.json")

    def run(self) -> None:
        state = self._init_or_resume()
        while not self._done(state):
            state = self._run_round(state)
            self.state_file.save(state)
        state.status = "completed" if state.best.score >= self.config.target_value else "paused"
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
        )
        self.state_file.save(state)
        return state

    def _run_round(self, state: SessionState) -> SessionState:
        state.current_round += 1
        t0 = time.time()

        if AntiOscillation.check_plateau(state.score_trajectory):
            state.hints_pending.append(
                "WARNING: 3+ rounds without improvement. Try fundamentally different approaches."
            )

        # Phase 1
        agent = AgentSession(self.config, state, self.workdir)
        strategies = agent.develop()
        t1 = time.time()

        # Cleanup
        Cleanup(self.workdir).between_phases()

        # Phase 2
        experiments = [
            ExperimentSetup.create(
                strategy=s, index=i, repo_path=self.workdir,
                workdir=self.workdir / ".agentforge" / "runs",
                hw=state.hardware,
                train_command=self.config.test_benchmark.split(),
            )
            for i, s in enumerate(strategies)
        ]
        runner = ParallelRunner(experiments, self.config, timeout=7200, workdir=self.workdir)
        results = runner.run()
        t2 = time.time()

        # All-fail handling
        if SelfRepair.is_all_fail(results):
            diagnosis = SelfRepair.diagnose_all_fail(results)
            if diagnosis == "environmental":
                SelfRepair.rebuild_venv(self.workdir)

        # Select winners
        winners = self.select_winners(results)

        # Update state
        round_result = RoundResult(
            round=state.current_round, experiments=results,
            winners=winners,
            phase1_minutes=(t1 - t0) / 60,
            phase2_minutes=(t2 - t1) / 60,
        )
        state.rounds.append(round_result)

        for r in results:
            if r.score > state.best.score:
                state.best = BestResult(
                    score=r.score, round=state.current_round,
                    experiment=r.id, commit="", checkpoint="",
                )

        best_this_round = max((r.score for r in results), default=0.0)
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
        return state.is_done(self.config.target_value)

    @staticmethod
    def select_winners(results: list[StrategyResult]) -> list[str]:
        ok_results = [r for r in results if r.status == "ok" and r.score > 0]
        if not ok_results:
            return []
        top_k = max(1, math.ceil(len(results) / 4))
        sorted_results = sorted(ok_results, key=lambda r: r.score, reverse=True)
        return [r.id for r in sorted_results[:top_k]]
