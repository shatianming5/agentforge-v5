import json
import pytest
from agentforge.analyzer import ProjectProfile


def test_profile_from_dict():
    data = {
        "description": "nanoGPT Shakespeare training",
        "run_command": "python train.py",
        "run_args": ["--max_iters=1000"],
        "eval_metric": "val_loss",
        "eval_direction": "minimize",
        "eval_method": "Read val_loss from stdout",
        "baseline_value": 2.5,
        "suggested_target": 1.8,
        "writable": ["train.py", "model.py"],
        "readonly": ["data/", "benchmark.py"],
        "python_cmd": "python3",
        "needs_gpu": False,
        "result_location": "stdout",
        "result_pattern": "",
        "metric_extraction": "import re\nlog = open('log.txt').read()\nscore = float(re.search(r'val_loss=(\\d+\\.\\d+)', log).group(1))",
        "import_checks": "import torch",
    }
    profile = ProjectProfile.from_dict(data)
    assert profile.description == "nanoGPT Shakespeare training"
    assert profile.eval_metric == "val_loss"
    assert profile.eval_direction == "minimize"
    assert profile.baseline_value == 2.5
    assert profile.writable == ["train.py", "model.py"]
    assert profile.needs_gpu is False


def test_profile_from_dict_defaults():
    """最小字段也能构造 profile。"""
    data = {
        "description": "A project",
        "run_command": "python main.py",
        "eval_metric": "score",
        "eval_direction": "maximize",
        "eval_method": "parse stdout",
        "suggested_target": 0.9,
        "writable": ["main.py"],
        "readonly": [],
        "metric_extraction": "score = 0.5",
    }
    profile = ProjectProfile.from_dict(data)
    assert profile.run_args == []
    assert profile.baseline_value is None
    assert profile.python_cmd == "python3"
    assert profile.needs_gpu is False
    assert profile.result_location == "stdout"
    assert profile.result_pattern == ""
    assert profile.import_checks == ""


def test_profile_to_dict_roundtrip():
    data = {
        "description": "test",
        "run_command": "python x.py",
        "run_args": [],
        "eval_metric": "acc",
        "eval_direction": "maximize",
        "eval_method": "parse",
        "baseline_value": None,
        "suggested_target": 0.9,
        "writable": ["x.py"],
        "readonly": [],
        "python_cmd": "python3",
        "needs_gpu": False,
        "result_location": "stdout",
        "result_pattern": "",
        "metric_extraction": "score = 0.9",
        "import_checks": "",
    }
    profile = ProjectProfile.from_dict(data)
    assert profile.to_dict() == data


def test_profile_missing_required_field():
    with pytest.raises(KeyError):
        ProjectProfile.from_dict({"description": "test"})
