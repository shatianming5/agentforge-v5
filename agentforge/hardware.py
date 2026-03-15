from __future__ import annotations
import os
import subprocess
from agentforge.state import HardwareInfo

try:
    import psutil
except ImportError:
    psutil = None


class HardwareDetector:
    @staticmethod
    def detect() -> HardwareInfo:
        gpus = HardwareDetector._list_gpus()
        cpu_cores = os.cpu_count() or 1
        ram_gb = HardwareDetector._get_ram_gb()
        disk_free_gb = HardwareDetector._get_disk_free_gb()
        if gpus:
            return HardwareInfo(
                device="cuda", gpu_model=gpus[0]["name"],
                num_gpus=len(gpus), cpu_cores=cpu_cores,
                ram_gb=ram_gb, disk_free_gb=disk_free_gb,
            )
        return HardwareInfo(
            device="cpu", gpu_model="", num_gpus=0,
            cpu_cores=cpu_cores, ram_gb=ram_gb, disk_free_gb=disk_free_gb,
        )

    @staticmethod
    def compute_N(hw: HardwareInfo) -> tuple[int, int]:
        if hw.device == "cpu":
            return max(1, min(hw.cpu_cores // 4, 8)), 0
        return hw.num_gpus, 1

    @staticmethod
    def _list_gpus() -> list[dict]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return []
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split(", ")
                gpus.append({"name": parts[0].strip(), "memory_mb": int(parts[1].strip())})
            return gpus
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

    @staticmethod
    def _get_ram_gb() -> int:
        if psutil:
            return int(psutil.virtual_memory().total / (1024**3))
        return 0

    @staticmethod
    def _get_disk_free_gb() -> int:
        if psutil:
            return int(psutil.disk_usage("/").free / (1024**3))
        return 0
