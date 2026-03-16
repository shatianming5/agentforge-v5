import yaml
from agentforge.analyzer import ProjectProfile
from agentforge.generator import ConfigGenerator


def _make_profile(**overrides):
    defaults = {
        "description": "nanoGPT Shakespeare training",
        "run_command": "python train.py",
        "run_args": ["--max_iters=1000"],
        "eval_metric": "val_loss",
        "eval_direction": "minimize",
        "eval_method": "parse stdout",
        "baseline_value": 2.5,
        "suggested_target": 1.8,
        "writable": ["train.py", "model.py"],
        "readonly": ["data/", "benchmark.py"],
        "python_cmd": "python3",
        "needs_gpu": False,
        "result_location": "stdout",
        "result_pattern": "",
        "metric_extraction": "import re\nlog = open('log.txt').read()\nscore = float(re.search(r'val_loss=(\\\\d+\\\\.\\\\d+)', log).group(1))",
        "import_checks": "import torch",
    }
    defaults.update(overrides)
    return ProjectProfile.from_dict(defaults)


def test_generate_challenge_yaml():
    profile = _make_profile()
    gen = ConfigGenerator(profile)
    content = gen.generate_challenge_yaml()
    parsed = yaml.safe_load(content)
    assert parsed["challenge"]["name"] == "nanoGPT Shakespeare training"
    assert parsed["target"]["metric"] == "val_loss"
    assert parsed["target"]["value"] == 1.8
    assert parsed["target"]["direction"] == "minimize"
    assert "train.py" in parsed["constraints"]["writable"]
    assert "data/" in parsed["constraints"]["read_only"]


def test_generate_benchmark_py():
    profile = _make_profile(
        metric_extraction="score = 1.5",
        eval_metric="val_loss",
    )
    gen = ConfigGenerator(profile)
    content = gen.generate_benchmark_py()
    assert "score = 1.5" in content
    assert "val_loss" in content
    assert "results/benchmark.json" in content
    # 必须是合法 Python
    compile(content, "benchmark.py", "exec")


def test_generate_test_suite_py():
    profile = _make_profile(
        run_command="python train.py",
        import_checks="import torch\nimport numpy",
    )
    gen = ConfigGenerator(profile)
    content = gen.generate_test_suite_py()
    assert "train.py" in content
    assert "import torch" in content
    # 必须是合法 Python
    compile(content, "test_suite.py", "exec")


def test_generate_all():
    profile = _make_profile()
    gen = ConfigGenerator(profile)
    files = gen.generate_all()
    assert "challenge.yaml" in files
    assert "benchmark.py" in files
    assert "test_suite.py" in files
    assert len(files) == 3
