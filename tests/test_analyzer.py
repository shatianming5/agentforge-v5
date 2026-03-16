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


from unittest.mock import patch, MagicMock
from agentforge.analyzer import ProjectAnalyzer


def test_analyzer_builds_prompt(tmp_path):
    """检查 prompt 包含关键指令。"""
    analyzer = ProjectAnalyzer(workdir=tmp_path)
    prompt = analyzer._build_prompt()
    assert "project_profile.json" in prompt
    assert "description" in prompt
    assert "run_command" in prompt
    assert "eval_metric" in prompt
    assert "metric_extraction" in prompt


def test_analyzer_parse_profile_json(tmp_path):
    """从 .agentforge/project_profile.json 解析 profile。"""
    profile_data = {
        "description": "Test project",
        "run_command": "python train.py",
        "eval_metric": "loss",
        "eval_direction": "minimize",
        "eval_method": "parse stdout",
        "suggested_target": 1.0,
        "writable": ["train.py"],
        "readonly": ["data/"],
        "metric_extraction": "score = 1.0",
    }
    af_dir = tmp_path / ".agentforge"
    af_dir.mkdir()
    (af_dir / "project_profile.json").write_text(json.dumps(profile_data))

    analyzer = ProjectAnalyzer(workdir=tmp_path)
    profile = analyzer._read_profile()
    assert profile.description == "Test project"
    assert profile.eval_metric == "loss"


def test_analyzer_analyze_calls_codex(tmp_path):
    """analyze() 调用 codex exec 并返回 profile。"""
    profile_data = {
        "description": "Analyzed project",
        "run_command": "python main.py",
        "eval_metric": "acc",
        "eval_direction": "maximize",
        "eval_method": "read from output",
        "suggested_target": 0.95,
        "writable": ["main.py"],
        "readonly": [],
        "metric_extraction": "score = 0.95",
    }
    af_dir = tmp_path / ".agentforge"
    af_dir.mkdir()

    def fake_codex_run(*args, **kwargs):
        (af_dir / "project_profile.json").write_text(json.dumps(profile_data))
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=fake_codex_run):
        analyzer = ProjectAnalyzer(workdir=tmp_path)
        profile = analyzer.analyze()
        assert profile.description == "Analyzed project"
        assert profile.eval_metric == "acc"


def test_analyzer_analyze_no_profile_raises(tmp_path):
    """Codex 没写 profile 文件时 raise。"""
    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        analyzer = ProjectAnalyzer(workdir=tmp_path)
        with pytest.raises(FileNotFoundError):
            analyzer.analyze()
