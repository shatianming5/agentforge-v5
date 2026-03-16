from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from agentforge.validate import validate_challenge


def _write_challenge(workdir: Path, overrides: dict | None = None) -> None:
    """写一个完整的 challenge 目录。"""
    config = {
        "challenge": {"name": "Test", "description": "Test desc"},
        "target": {"metric": "val_loss", "value": 1.8, "direction": "minimize"},
        "tests": {"smoke": "echo ok", "full": "echo ok", "benchmark": "python benchmark.py"},
        "constraints": {"writable": ["train.py"], "read_only": ["data/"]},
    }
    if overrides:
        for k, v in overrides.items():
            keys = k.split(".")
            d = config
            for key in keys[:-1]:
                d = d[key]
            d[keys[-1]] = v

    (workdir / "challenge.yaml").write_text(yaml.dump(config))
    (workdir / "benchmark.py").write_text(
        'import json\nscore = 1.5\nwith open("results/benchmark.json","w") as f:\n'
        '    json.dump({"val_loss": score}, f)\n'
    )
    (workdir / "test_suite.py").write_text("print('ok')\n")
    (workdir / "train.py").write_text("print('train')\n")
    # 模拟 git repo
    (workdir / ".git").mkdir(exist_ok=True)


class TestValidateChallenge:
    def test_valid(self, tmp_path: Path):
        _write_challenge(tmp_path)
        errors = validate_challenge(tmp_path)
        assert errors == []

    def test_no_challenge_yaml(self, tmp_path: Path):
        errors = validate_challenge(tmp_path)
        assert any("challenge.yaml 不存在" in e for e in errors)

    def test_no_benchmark(self, tmp_path: Path):
        _write_challenge(tmp_path)
        (tmp_path / "benchmark.py").unlink()
        errors = validate_challenge(tmp_path)
        assert any("benchmark.py 不存在" in e for e in errors)

    def test_no_test_suite(self, tmp_path: Path):
        _write_challenge(tmp_path)
        (tmp_path / "test_suite.py").unlink()
        errors = validate_challenge(tmp_path)
        assert any("test_suite.py 不存在" in e for e in errors)

    def test_invalid_direction(self, tmp_path: Path):
        _write_challenge(tmp_path, {"target.direction": "up"})
        errors = validate_challenge(tmp_path)
        assert any("direction 无效" in e for e in errors)

    def test_no_git(self, tmp_path: Path):
        _write_challenge(tmp_path)
        (tmp_path / ".git").rmdir()
        errors = validate_challenge(tmp_path)
        assert any("不是 git 仓库" in e for e in errors)

    def test_metric_not_in_benchmark(self, tmp_path: Path):
        _write_challenge(tmp_path)
        (tmp_path / "benchmark.py").write_text(
            'import json\nwith open("results/benchmark.json","w") as f:\n'
            '    json.dump({"accuracy": 0.9}, f)\n'
        )
        errors = validate_challenge(tmp_path)
        assert any("val_loss" in e for e in errors)
