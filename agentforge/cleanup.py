from __future__ import annotations
import shutil
import subprocess
from pathlib import Path


class Cleanup:
    TRIAL_DIRS = ["__pycache__", "logs", "runs", "wandb"]

    def __init__(self, workdir: Path):
        self.workdir = workdir

    def delete_trial_artifacts(self) -> None:
        for dirname in self.TRIAL_DIRS:
            d = self.workdir / dirname
            if d.is_dir():
                shutil.rmtree(d)
        for pattern in ["*.pt", "*.pth", "*.ckpt", "events.out.tfevents.*"]:
            for f in self.workdir.glob(pattern):
                f.unlink()
        for p in self.workdir.rglob("__pycache__"):
            if p.is_dir():
                shutil.rmtree(p)

    def verify_disk_space(self, min_gb: int) -> None:
        usage = shutil.disk_usage(self.workdir)
        free_gb = usage.free / (1024**3)
        if free_gb < min_gb:
            raise RuntimeError(
                f"Insufficient disk space: {free_gb:.1f}GB free, need {min_gb}GB"
            )

    def reset_gpu_contexts(self) -> None:
        try:
            subprocess.run(["nvidia-smi", "--gpu-reset"],
                          capture_output=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def between_phases(self) -> None:
        self.delete_trial_artifacts()
        self.reset_gpu_contexts()
        self.verify_disk_space(min_gb=10)

    def delete_loser_workdirs(self, workdirs: list[Path]) -> None:
        for wd in workdirs:
            if wd.exists():
                shutil.rmtree(wd)

    def gc_old_checkpoints(self, keep_best: Path | None = None) -> None:
        ckpt_dir = self.workdir / ".agentforge" / "checkpoints"
        if not ckpt_dir.exists():
            return
        for f in ckpt_dir.iterdir():
            if keep_best and f == keep_best:
                continue
            if f.suffix in (".pt", ".pth", ".ckpt"):
                f.unlink()
