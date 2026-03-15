from __future__ import annotations
from unittest.mock import patch, MagicMock
from agentforge.hardware import HardwareDetector
from agentforge.state import HardwareInfo


class TestHardwareDetector:
    def test_detect_cpu_only(self):
        with patch("agentforge.hardware.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError
            with patch("os.cpu_count", return_value=8):
                with patch("agentforge.hardware.psutil") as mock_psutil:
                    mock_psutil.virtual_memory.return_value = MagicMock(total=32 * 1024**3)
                    mock_psutil.disk_usage.return_value = MagicMock(free=100 * 1024**3)
                    hw = HardwareDetector.detect()
        assert hw.device == "cpu"
        assert hw.num_gpus == 0
        assert hw.cpu_cores == 8
        assert hw.ram_gb == 32

    def test_compute_N_cpu_small(self):
        hw = HardwareInfo("cpu", "", 0, 16, 32, 100)
        N, gpus_per = HardwareDetector.compute_N(hw)
        assert N >= 1
        assert gpus_per == 0

    def test_compute_N_single_gpu(self):
        hw = HardwareInfo("cuda", "A100 80GB", 1, 64, 256, 500)
        N, gpus_per = HardwareDetector.compute_N(hw)
        assert N == 1
        assert gpus_per == 1

    def test_compute_N_eight_gpu(self):
        hw = HardwareInfo("cuda", "A100 80GB", 8, 64, 256, 500)
        N, gpus_per = HardwareDetector.compute_N(hw)
        assert N == 8
        assert gpus_per == 1
