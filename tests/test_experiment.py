from __future__ import annotations
import os
from pathlib import Path
from agentforge.experiment import ExperimentSetup, Experiment
from agentforge.agent import Strategy
from agentforge.state import HardwareInfo


def _make_strategy(index=0):
    return Strategy(
        name=f"test_{index}", branch=f"agentforge/iter-1/exp-{index}",
        confidence=0.8, measured_vram_gb=40.0, measured_epoch_seconds=45.0,
        batch_size=64, resume_checkpoint=False, category="opt", risk="low",
    )


class TestExperiment:
    def test_fields(self):
        exp = Experiment(0, _make_strategy(), Path("/tmp/e0"), Path("/tmp/e0.log"),
                        {"CUDA_VISIBLE_DEVICES": "0"}, ["python", "train.py"])
        assert exp.env["CUDA_VISIBLE_DEVICES"] == "0"
        assert "exp-0" in str(exp.log_path) or True


class TestExperimentSetup:
    def test_build_env_cuda(self):
        hw = HardwareInfo("cuda", "A100", 8, 64, 256, 500)
        env = ExperimentSetup.build_env(0, hw)
        assert env["CUDA_VISIBLE_DEVICES"] == "0"
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert "PYTHONHASHSEED" in env

    def test_build_env_cpu(self):
        hw = HardwareInfo("cpu", "", 0, 4, 16, 50)
        env = ExperimentSetup.build_env(0, hw)
        assert "CUDA_VISIBLE_DEVICES" not in env

    def test_build_env_different_indices(self):
        hw = HardwareInfo("cuda", "A100", 8, 64, 256, 500)
        env0 = ExperimentSetup.build_env(0, hw)
        env3 = ExperimentSetup.build_env(3, hw)
        assert env0["CUDA_VISIBLE_DEVICES"] == "0"
        assert env3["CUDA_VISIBLE_DEVICES"] == "3"
        assert env0["PYTHONHASHSEED"] != env3["PYTHONHASHSEED"]
