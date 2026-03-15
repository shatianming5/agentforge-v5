from __future__ import annotations
import os
import subprocess
import time
from pathlib import Path

from agentforge.config import ChallengeConfig
from agentforge.experiment import Experiment
from agentforge.monitor import Monitor
from agentforge.scorer import Scorer
from agentforge.state import StrategyResult


class ParallelRunner:
    def __init__(self, experiments, config, timeout, workdir):
        self._experiments = experiments
        self._config = config
        self._timeout = timeout
        self._workdir = workdir

    def run(self, N: int = 1) -> list[StrategyResult]:
        self._N = N
        launched = self._launch_all()
        monitor = Monitor(
            processes=[(exp.index, proc, exp.log_path) for exp, proc, _ in launched],
            timeout=self._timeout,
            log_paths=[exp.log_path for exp, _, _ in launched],
            workdir=self._workdir,
        )
        monitor.run()
        try:
            return self._collect_results(launched, monitor)
        finally:
            for _, _, log_file in launched:
                try:
                    log_file.close()
                except OSError:
                    pass

    def _launch_all(self):
        launched = []
        for exp in self._experiments:
            exp.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = open(exp.log_path, "w")
            proc = subprocess.Popen(
                exp.train_command,
                cwd=str(exp.workdir),
                env=exp.env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp,
            )
            launched.append((exp, proc, log_file))
        return launched

    def _collect_results(self, launched, monitor):
        killed = {e.exp_index: e for e in monitor.events if e.reason != "disk"}
        results = []
        for exp, proc, _ in launched:
            if exp.index in killed:
                evt = killed[exp.index]
                results.append(StrategyResult(
                    id=f"exp-{exp.index}", strategy=exp.strategy.name,
                    branch=exp.strategy.branch, score=0.0,
                    status=evt.reason, error=evt.detail,
                    actual_vram_gb=0.0, actual_epoch_seconds=0.0,
                    actual_batch_size=exp.strategy.batch_size,
                ))
            elif proc.returncode != 0:
                results.append(StrategyResult(
                    id=f"exp-{exp.index}", strategy=exp.strategy.name,
                    branch=exp.strategy.branch, score=0.0,
                    status="error", error=f"exit code {proc.returncode}",
                    actual_vram_gb=0.0, actual_epoch_seconds=0.0,
                    actual_batch_size=exp.strategy.batch_size,
                ))
            else:
                score = Scorer.score(exp, self._config, proc.returncode, N=self._N)
                results.append(StrategyResult(
                    id=f"exp-{exp.index}", strategy=exp.strategy.name,
                    branch=exp.strategy.branch, score=score,
                    status="ok", error=None,
                    actual_vram_gb=exp.strategy.measured_vram_gb,
                    actual_epoch_seconds=exp.strategy.measured_epoch_seconds,
                    actual_batch_size=exp.strategy.batch_size,
                ))
        return results
