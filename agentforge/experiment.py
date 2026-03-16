from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from agentforge.agent import Strategy
from agentforge.anti_oscillation import AntiOscillation
from agentforge.state import HardwareInfo


@dataclass
class Experiment:
    index: int
    strategy: Strategy
    workdir: Path
    log_path: Path
    env: dict[str, str]
    train_command: list[str]


class ExperimentSetup:
    @staticmethod
    def build_env(index: int, hw: HardwareInfo, strategy: Strategy | None = None) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Use anti-oscillation seed if strategy provided, else simple seed
        if strategy:
            env["PYTHONHASHSEED"] = str(AntiOscillation.compute_seed(strategy.name, index))
        else:
            env["PYTHONHASHSEED"] = str(42 + index)
        if hw.device == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = str(index % hw.num_gpus)
        return env

    @staticmethod
    def create_clone(repo_path: Path, branch: str, workdir: Path) -> Path:
        clone_dir = workdir / f"clone-{branch.replace('/', '-')}"
        subprocess.run(
            ["git", "clone", "--shared", "--branch", branch,
             str(repo_path), str(clone_dir)],
            check=True, capture_output=True, timeout=30,
        )
        subprocess.run(
            ["git", "config", "gc.auto", "0"],
            cwd=str(clone_dir), check=True, capture_output=True,
        )
        # Symlink untracked files/dirs from source repo (e.g. data, models)
        ExperimentSetup._symlink_untracked(repo_path, clone_dir)
        return clone_dir

    @staticmethod
    def _symlink_untracked(src: Path, dst: Path) -> None:
        """Symlink items in src that are missing in dst (untracked data files, etc.)."""
        for item in src.iterdir():
            if item.name.startswith("."):
                continue
            target = dst / item.name
            if not target.exists():
                # Missing entirely in clone → symlink the whole thing
                os.symlink(str(item.resolve()), str(target))
            elif item.is_dir() and target.is_dir():
                # Directory exists in both → recurse to catch untracked files inside
                ExperimentSetup._symlink_untracked(item, target)

    @staticmethod
    def set_cpu_affinity(index: int, hw: HardwareInfo) -> list[str]:
        """Return taskset prefix command for CPU affinity, if applicable."""
        if hw.device != "cpu" or hw.cpu_cores < 4:
            return []
        # taskset is Linux-only
        if not shutil.which("taskset"):
            return []
        cores_per_exp = max(1, hw.cpu_cores // 8)
        start = index * cores_per_exp
        end = start + cores_per_exp - 1
        if end >= hw.cpu_cores:
            return []
        return ["taskset", "-c", f"{start}-{end}"]

    @staticmethod
    def create(strategy, index, repo_path, workdir, hw, train_command,
               default_benchmark: str = "") -> Experiment:
        clone_dir = ExperimentSetup.create_clone(repo_path, strategy.branch, workdir)
        log_dir = workdir / "logs"
        log_dir.mkdir(exist_ok=True)
        env = ExperimentSetup.build_env(index, hw, strategy)
        # Use strategy's train_command if provided, else fall back to default
        if strategy.train_command:
            cmd = shlex.split(strategy.train_command)
        else:
            cmd = train_command
        # Add CPU affinity prefix if applicable
        affinity = ExperimentSetup.set_cpu_affinity(index, hw)
        if affinity:
            cmd = affinity + cmd
        return Experiment(
            index=index, strategy=strategy, workdir=clone_dir,
            log_path=log_dir / f"exp-{index}.log",
            env=env, train_command=cmd,
        )
