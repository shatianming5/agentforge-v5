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
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if visible:
                gpu_ids = [x.strip() for x in visible.split(",") if x.strip()]
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[index % len(gpu_ids)]
            else:
                env["CUDA_VISIBLE_DEVICES"] = str(index % hw.num_gpus)
        return env

    @staticmethod
    def create_clone(repo_path: Path, branch: str, workdir: Path) -> Path:
        clone_dir = workdir / f"clone-{branch.replace('/', '-')}"
        subprocess.run(
            ["git", "worktree", "add", str(clone_dir), branch],
            cwd=str(repo_path),
            check=True, capture_output=True, timeout=30,
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
    def _find_strategy_config(clone_dir: Path) -> str | None:
        """Find the strategy-specific config in configs/agentforge/."""
        ag_configs = clone_dir / "configs" / "agentforge"
        if ag_configs.is_dir():
            py_files = sorted(ag_configs.glob("*.py"))
            if py_files:
                return str(py_files[0].relative_to(clone_dir))
        return None

    @staticmethod
    def create(strategy, index, repo_path, workdir, hw, train_command,
               default_benchmark: str = "") -> Experiment:
        clone_dir = ExperimentSetup.create_clone(repo_path, strategy.branch, workdir)
        log_dir = workdir / "logs"
        log_dir.mkdir(exist_ok=True)
        env = ExperimentSetup.build_env(index, hw, strategy)
        # Use strategy's train_command if provided
        if strategy.train_command:
            cmd = shlex.split(strategy.train_command)
        else:
            # Auto-discover config from configs/agentforge/ in clone
            config_path = ExperimentSetup._find_strategy_config(clone_dir)
            if config_path:
                work_dir_name = f"work_dirs/{strategy.branch.replace('/', '_')}"
                cmd = [
                    "conda", "run", "-n", "iraod", "--no-capture-output",
                    "python", "train.py", config_path,
                    "--work-dir", work_dir_name,
                ]
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
