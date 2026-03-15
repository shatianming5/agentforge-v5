from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentforge.orchestrator import Orchestrator
from agentforge.agent import Strategy
from agentforge.state import StateFile, HardwareInfo, StrategyResult


def _make_mock_strategy():
    return Strategy(
        name="baseline",
        branch="agentforge/iter-1/exp-0",
        confidence=0.9,
        measured_vram_gb=10,
        measured_epoch_seconds=30,
        batch_size=32,
        resume_checkpoint=False,
        category="opt",
        risk="high",
    )


class TestIntegration:
    def test_single_round_n1(self, tmp_path, sample_config_path):
        """Run one complete round with N=1, mocked Agent.
        Score 0.96 exceeds target 0.95, so session should complete in 1 round."""
        workdir = tmp_path / "project"
        workdir.mkdir()

        mock_strategy = _make_mock_strategy()

        mock_experiment = MagicMock()
        mock_experiment.strategy = mock_strategy
        mock_experiment.workdir = workdir / ".agentforge" / "runs" / "clone-exp-0"

        with patch("agentforge.orchestrator.AgentSession") as MockAgent, \
             patch("agentforge.orchestrator.ExperimentSetup") as MockSetup, \
             patch("agentforge.orchestrator.ParallelRunner") as MockRunner, \
             patch("agentforge.orchestrator.Cleanup") as MockCleanup, \
             patch("agentforge.orchestrator.HardwareDetector") as MockHW, \
             patch("agentforge.orchestrator.SelfRepair") as MockRepair:

            MockHW.detect.return_value = HardwareInfo("cpu", "", 0, 4, 16, 50)
            MockHW.compute_N.return_value = (1, 0)
            MockAgent.return_value.develop.return_value = [mock_strategy]
            MockSetup.create.return_value = mock_experiment
            MockRunner.return_value.run.return_value = [
                StrategyResult("exp-0", "baseline", "b0", 0.96, "ok", None, 10, 30, 32)
            ]
            MockRepair.is_all_fail.return_value = False

            orch = Orchestrator(sample_config_path, workdir)
            orch.run()

        sf = StateFile(workdir / ".agentforge" / "state.json")
        state = sf.load()
        assert state.status == "completed"
        assert state.best.score == 0.96
        assert state.budget.rounds_used == 1

    def test_multiple_rounds_until_budget(self, tmp_path, sample_config_path):
        """Run until budget exhausted (3 rounds), score never reaches target 0.95."""
        workdir = tmp_path / "project2"
        workdir.mkdir()

        mock_strategy = Strategy(
            name="s1",
            branch="b1",
            confidence=0.5,
            measured_vram_gb=10,
            measured_epoch_seconds=30,
            batch_size=32,
            resume_checkpoint=False,
            category="opt",
            risk="high",
        )

        mock_experiment = MagicMock()
        mock_experiment.strategy = mock_strategy
        mock_experiment.workdir = workdir / ".agentforge" / "runs" / "clone-exp-0"

        call_count = 0

        def mock_run_side_effect():
            nonlocal call_count
            call_count += 1
            return [
                StrategyResult(
                    f"exp-0", "s1", "b1", 0.5 + call_count * 0.1,
                    "ok", None, 10, 30, 32,
                )
            ]

        with patch("agentforge.orchestrator.AgentSession") as MockAgent, \
             patch("agentforge.orchestrator.ExperimentSetup") as MockSetup, \
             patch("agentforge.orchestrator.ParallelRunner") as MockRunner, \
             patch("agentforge.orchestrator.Cleanup") as MockCleanup, \
             patch("agentforge.orchestrator.HardwareDetector") as MockHW, \
             patch("agentforge.orchestrator.SelfRepair") as MockRepair:

            MockHW.detect.return_value = HardwareInfo("cpu", "", 0, 4, 16, 50)
            MockHW.compute_N.return_value = (1, 0)
            MockAgent.return_value.develop.return_value = [mock_strategy]
            MockSetup.create.return_value = mock_experiment
            MockRunner.return_value.run.side_effect = mock_run_side_effect
            MockRepair.is_all_fail.return_value = False

            orch = Orchestrator(sample_config_path, workdir)
            # Let init create state with default 25 rounds, then override to 3
            orch.state_file.path.parent.mkdir(parents=True, exist_ok=True)
            state = orch._init_or_resume()
            state.budget.rounds_max = 3
            orch.state_file.save(state)

            orch.run()

        sf = StateFile(workdir / ".agentforge" / "state.json")
        state = sf.load()
        assert state.status == "paused"  # didn't reach target
        assert state.budget.rounds_used == 3
        assert len(state.score_trajectory) == 3

    def test_all_fail_triggers_repair(self, tmp_path, sample_config_path):
        """When all experiments fail, SelfRepair should be invoked."""
        workdir = tmp_path / "project3"
        workdir.mkdir()

        mock_strategy = _make_mock_strategy()

        mock_experiment = MagicMock()
        mock_experiment.strategy = mock_strategy
        mock_experiment.workdir = workdir / ".agentforge" / "runs" / "clone-exp-0"

        with patch("agentforge.orchestrator.AgentSession") as MockAgent, \
             patch("agentforge.orchestrator.ExperimentSetup") as MockSetup, \
             patch("agentforge.orchestrator.ParallelRunner") as MockRunner, \
             patch("agentforge.orchestrator.Cleanup") as MockCleanup, \
             patch("agentforge.orchestrator.HardwareDetector") as MockHW, \
             patch("agentforge.orchestrator.SelfRepair") as MockRepair:

            MockHW.detect.return_value = HardwareInfo("cpu", "", 0, 4, 16, 50)
            MockHW.compute_N.return_value = (1, 0)
            MockAgent.return_value.develop.return_value = [mock_strategy]
            MockSetup.create.return_value = mock_experiment

            # All experiments fail
            fail_result = StrategyResult(
                "exp-0", "baseline", "b0", 0.0, "error",
                "same error", 0, 0, 32,
            )
            MockRunner.return_value.run.return_value = [fail_result]
            MockRepair.is_all_fail.return_value = True
            MockRepair.diagnose_all_fail.return_value = "environmental"

            orch = Orchestrator(sample_config_path, workdir)
            # Set rounds_max=1 so it stops after one round
            state = orch._init_or_resume()
            state.budget.rounds_max = 1
            orch.state_file.save(state)

            orch.run()

        # Verify self-repair was triggered
        MockRepair.diagnose_all_fail.assert_called_once()
        MockRepair.rebuild_venv.assert_called_once_with(workdir)

    def test_state_persists_across_resume(self, tmp_path, sample_config_path):
        """Verify that state is correctly saved and can be resumed."""
        workdir = tmp_path / "project4"
        workdir.mkdir()

        mock_strategy = _make_mock_strategy()
        mock_experiment = MagicMock()
        mock_experiment.strategy = mock_strategy
        mock_experiment.workdir = workdir / ".agentforge" / "runs" / "clone-exp-0"

        with patch("agentforge.orchestrator.AgentSession") as MockAgent, \
             patch("agentforge.orchestrator.ExperimentSetup") as MockSetup, \
             patch("agentforge.orchestrator.ParallelRunner") as MockRunner, \
             patch("agentforge.orchestrator.Cleanup"), \
             patch("agentforge.orchestrator.HardwareDetector") as MockHW, \
             patch("agentforge.orchestrator.SelfRepair") as MockRepair:

            MockHW.detect.return_value = HardwareInfo("cpu", "", 0, 4, 16, 50)
            MockHW.compute_N.return_value = (1, 0)
            MockAgent.return_value.develop.return_value = [mock_strategy]
            MockSetup.create.return_value = mock_experiment
            MockRunner.return_value.run.return_value = [
                StrategyResult("exp-0", "baseline", "b0", 0.80, "ok", None, 10, 30, 32)
            ]
            MockRepair.is_all_fail.return_value = False

            # Run 1 round, budget=1 => paused
            orch = Orchestrator(sample_config_path, workdir)
            state = orch._init_or_resume()
            state.budget.rounds_max = 1
            orch.state_file.save(state)
            orch.run()

        sf = StateFile(workdir / ".agentforge" / "state.json")
        state = sf.load()
        assert state.status == "paused"
        assert state.best.score == 0.80
        assert state.current_round == 1

        # Verify state file has correct structure by reloading
        state2 = sf.load()
        assert state2.session_id == state.session_id
        assert state2.budget.rounds_used == 1
