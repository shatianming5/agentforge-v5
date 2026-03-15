from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from agentforge.agent import Strategy
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
    def build_env(index: int, hw: HardwareInfo) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
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
        return clone_dir

    @staticmethod
    def create(strategy, index, repo_path, workdir, hw, train_command) -> Experiment:
        clone_dir = ExperimentSetup.create_clone(repo_path, strategy.branch, workdir)
        log_dir = workdir / "logs"
        log_dir.mkdir(exist_ok=True)
        env = ExperimentSetup.build_env(index, hw)
        return Experiment(
            index=index, strategy=strategy, workdir=clone_dir,
            log_path=log_dir / f"exp-{index}.log",
            env=env, train_command=train_command,
        )
