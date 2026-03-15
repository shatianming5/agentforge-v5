from __future__ import annotations
from pathlib import Path
import pytest
from agentforge.cleanup import Cleanup


class TestCleanup:
    def test_delete_trial_artifacts(self, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "foo.pyc").touch()
        (tmp_path / "trial_checkpoint.pt").touch()
        (tmp_path / "events.out.tfevents.123").touch()
        c = Cleanup(tmp_path)
        c.delete_trial_artifacts()
        assert not (tmp_path / "__pycache__").exists()
        assert not (tmp_path / "trial_checkpoint.pt").exists()
        assert not (tmp_path / "events.out.tfevents.123").exists()

    def test_verify_disk_space_ok(self, tmp_path):
        c = Cleanup(tmp_path)
        c.verify_disk_space(min_gb=0)

    def test_verify_disk_space_insufficient(self, tmp_path):
        c = Cleanup(tmp_path)
        with pytest.raises(RuntimeError, match="Insufficient disk"):
            c.verify_disk_space(min_gb=999999)

    def test_delete_loser_workdirs(self, tmp_path):
        loser = tmp_path / "loser"
        loser.mkdir()
        (loser / "file.txt").touch()
        c = Cleanup(tmp_path)
        c.delete_loser_workdirs([loser])
        assert not loser.exists()

    def test_gc_old_checkpoints(self, tmp_path):
        ckpt_dir = tmp_path / ".agentforge" / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        best = ckpt_dir / "best.pt"
        best.touch()
        old = ckpt_dir / "old.pt"
        old.touch()
        c = Cleanup(tmp_path)
        c.gc_old_checkpoints(keep_best=best)
        assert best.exists()
        assert not old.exists()
