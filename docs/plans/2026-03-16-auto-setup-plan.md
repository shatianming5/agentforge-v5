# AgentForge Auto-Setup 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 用户将 AgentForge 指向任意 repo，框架自动分析项目并生成 challenge.yaml、benchmark.py、test_suite.py，用户逐个确认后启动优化循环。零手动配置。

**Architecture:** 三个新模块完全解耦——ProjectAnalyzer 调用 Codex CLI read-only 分析 repo 输出 ProjectProfile JSON；ConfigGenerator 用模板渲染 ProjectProfile 生成 3 个文件；InteractiveConfirm 逐个展示让用户 Y/n/edit。在 Orchestrator._init_or_resume() 开头检测 challenge.yaml 是否存在，不存在则触发 auto-setup 流程。

**Tech Stack:** Python 3.10+, dataclasses, Codex CLI (`codex exec -s read-only`), Jinja2-like string templates, click CLI

**设计文档:** `docs/plans/2026-03-16-auto-setup-design.md`

---

### Task 1: ProjectProfile 数据结构

**Files:**
- Create: `agentforge/analyzer.py`
- Test: `tests/test_analyzer.py`

**Step 1: 写失败测试**

```python
# tests/test_analyzer.py
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
```

**Step 2: 运行测试确认失败**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_analyzer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentforge.analyzer'`

**Step 3: 实现 ProjectProfile**

```python
# agentforge/analyzer.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ProjectProfile:
    description: str
    run_command: str
    eval_metric: str
    eval_direction: str
    eval_method: str
    suggested_target: float
    writable: list[str]
    readonly: list[str]
    metric_extraction: str

    run_args: list[str] = field(default_factory=list)
    baseline_value: float | None = None
    python_cmd: str = "python3"
    needs_gpu: bool = False
    result_location: str = "stdout"
    result_pattern: str = ""
    import_checks: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectProfile:
        return cls(
            description=data["description"],
            run_command=data["run_command"],
            eval_metric=data["eval_metric"],
            eval_direction=data["eval_direction"],
            eval_method=data["eval_method"],
            suggested_target=float(data["suggested_target"]),
            writable=list(data["writable"]),
            readonly=list(data["readonly"]),
            metric_extraction=data["metric_extraction"],
            run_args=list(data.get("run_args", [])),
            baseline_value=data.get("baseline_value"),
            python_cmd=data.get("python_cmd", "python3"),
            needs_gpu=bool(data.get("needs_gpu", False)),
            result_location=data.get("result_location", "stdout"),
            result_pattern=data.get("result_pattern", ""),
            import_checks=data.get("import_checks", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

**Step 4: 运行测试确认通过**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_analyzer.py -v`
Expected: 4 passed

**Step 5: 提交**

```bash
git add agentforge/analyzer.py tests/test_analyzer.py
git commit -m "feat: add ProjectProfile dataclass for auto-setup"
```

---

### Task 2: ProjectAnalyzer（Codex CLI 分析）

**Files:**
- Modify: `agentforge/analyzer.py`
- Test: `tests/test_analyzer.py`

**Step 1: 写失败测试**

在 `tests/test_analyzer.py` 追加：

```python
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
    import json
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

    import json

    def fake_codex_run(*args, **kwargs):
        # Codex 会写 profile json
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
```

**Step 2: 运行测试确认失败**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_analyzer.py::test_analyzer_builds_prompt -v`
Expected: FAIL — `ImportError: cannot import name 'ProjectAnalyzer'`

**Step 3: 实现 ProjectAnalyzer**

在 `agentforge/analyzer.py` 追加：

```python
import json
import subprocess


class ProjectAnalyzer:
    """调用 Codex CLI read-only 分析 repo，生成 ProjectProfile。"""

    PROMPT_TEMPLATE = """\
你是一个项目分析器。分析当前目录的代码，返回结构化 JSON。

你可以：
  - 读取任何文件（cat, head, find, ls）
  - 跑简短的测试（python -c "import torch"）
你不能：
  - 修改任何文件（除了写入 .agentforge/project_profile.json）

请分析并将 JSON 写入 .agentforge/project_profile.json，字段如下：
{{
  "description": "项目简短描述",
  "run_command": "主运行命令，如 python train.py",
  "run_args": ["默认参数列表"],
  "eval_metric": "评估指标名，如 val_loss / accuracy / sharpe_ratio",
  "eval_direction": "minimize 或 maximize",
  "eval_method": "如何获取指标的描述",
  "baseline_value": null 或当前基准值(float),
  "suggested_target": 建议目标值(float),
  "writable": ["可修改的文件/目录列表"],
  "readonly": ["不可修改的文件/目录列表"],
  "python_cmd": "python3 或 conda run -n env python",
  "needs_gpu": false,
  "result_location": "stdout / checkpoint / output_file",
  "result_pattern": "结果文件 glob 模式",
  "metric_extraction": "Python 代码片段，执行后将指标值赋给变量 score",
  "import_checks": "Python 代码片段，验证依赖可导入"
}}

重点分析：
1. 找到主运行入口和启动方式
2. 找到评估指标（看 eval/val/metric 相关代码）
3. 写一段 Python 代码片段放入 metric_extraction，该代码执行后将指标值赋给变量 score
4. 识别哪些文件是数据/评估（不该改），哪些是可优化的（可以改）
5. 根据项目现状建议合理的优化目标值
6. 写一段 import 检查代码放入 import_checks
"""

    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)

    def _build_prompt(self) -> str:
        return self.PROMPT_TEMPLATE

    def _read_profile(self) -> ProjectProfile:
        profile_path = self.workdir / ".agentforge" / "project_profile.json"
        if not profile_path.exists():
            raise FileNotFoundError(
                f"Codex 未生成 profile 文件: {profile_path}"
            )
        data = json.loads(profile_path.read_text())
        return ProjectProfile.from_dict(data)

    def analyze(self) -> ProjectProfile:
        af_dir = self.workdir / ".agentforge"
        af_dir.mkdir(parents=True, exist_ok=True)
        prompt = self._build_prompt()
        subprocess.run(
            ["codex", "exec", "-s", "read-only", prompt],
            cwd=str(self.workdir),
            timeout=600,
            capture_output=True,
            text=True,
        )
        return self._read_profile()
```

**Step 4: 运行测试确认通过**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_analyzer.py -v`
Expected: 8 passed

**Step 5: 提交**

```bash
git add agentforge/analyzer.py tests/test_analyzer.py
git commit -m "feat: add ProjectAnalyzer with Codex CLI integration"
```

---

### Task 3: ConfigGenerator 模板渲染

**Files:**
- Create: `agentforge/generator.py`
- Test: `tests/test_generator.py`

**Step 1: 写失败测试**

```python
# tests/test_generator.py
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
        "metric_extraction": "import re\\nlog = open('log.txt').read()\\nscore = float(re.search(r'val_loss=(\\\\d+\\\\.\\\\d+)', log).group(1))",
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
        import_checks="import torch\\nimport numpy",
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
```

**Step 2: 运行测试确认失败**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_generator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentforge.generator'`

**Step 3: 实现 ConfigGenerator**

```python
# agentforge/generator.py
from __future__ import annotations

from agentforge.analyzer import ProjectProfile


class ConfigGenerator:
    """从 ProjectProfile 渲染生成 challenge.yaml、benchmark.py、test_suite.py。"""

    def __init__(self, profile: ProjectProfile):
        self.profile = profile

    def generate_challenge_yaml(self) -> str:
        p = self.profile
        run_args_str = " ".join(p.run_args) if p.run_args else ""
        writable_list = "\n".join(f"    - {w}" for w in p.writable)
        readonly_list = "\n".join(f"    - {r}" for r in p.readonly)
        baseline_line = f"    Baseline: {p.baseline_value}" if p.baseline_value is not None else ""

        return (
            f"challenge:\n"
            f"  name: '{p.description[:60]}'\n"
            f"  description: |\n"
            f"    {p.description}\n"
            f"    Run command: {p.run_command} {run_args_str}\n"
            f"{baseline_line}\n"
            f"target:\n"
            f"  metric: {p.eval_metric}\n"
            f"  value: {p.suggested_target}\n"
            f"  direction: {p.eval_direction}\n"
            f"tests:\n"
            f"  smoke: '{p.python_cmd} test_suite.py'\n"
            f"  full: '{p.python_cmd} test_suite.py'\n"
            f"  benchmark: '{p.python_cmd} benchmark.py'\n"
            f"constraints:\n"
            f"  writable:\n"
            f"{writable_list}\n"
            f"  read_only:\n"
            f"{readonly_list}\n"
        )

    def generate_benchmark_py(self) -> str:
        p = self.profile
        return (
            '#!/usr/bin/env python3\n'
            '"""Auto-generated benchmark by AgentForge."""\n'
            'import json\n'
            'import os\n'
            '\n'
            '# ---- metric extraction (generated by Codex analysis) ----\n'
            f'{p.metric_extraction}\n'
            '# ---- end metric extraction ----\n'
            '\n'
            'os.makedirs("results", exist_ok=True)\n'
            'with open("results/benchmark.json", "w") as f:\n'
            f'    json.dump({{"{p.eval_metric}": round(score, 6)}}, f, indent=2)\n'
            f'print(f"[benchmark] {p.eval_metric}={{score:.6f}}")\n'
        )

    def generate_test_suite_py(self) -> str:
        p = self.profile
        # 从 run_command 提取脚本名
        parts = p.run_command.split()
        run_script = parts[-1] if parts else "main.py"

        return (
            '#!/usr/bin/env python3\n'
            '"""Auto-generated sanity tests by AgentForge."""\n'
            'import os\n'
            'import sys\n'
            '\n'
            '\n'
            'def test_entry_exists():\n'
            f'    assert os.path.exists("{run_script}"), "入口脚本不存在: {run_script}"\n'
            '    return True\n'
            '\n'
            '\n'
            'def test_imports():\n'
            f'    {p.import_checks}\n'
            '    return True\n'
            '\n'
            '\n'
            'tests = [test_entry_exists, test_imports]\n'
            'for t in tests:\n'
            '    try:\n'
            '        assert t()\n'
            '    except Exception as e:\n'
            '        print(f"FAIL: {t.__name__}: {e}")\n'
            '        sys.exit(1)\n'
            'print(f"All {len(tests)} tests passed")\n'
        )

    def generate_all(self) -> dict[str, str]:
        return {
            "challenge.yaml": self.generate_challenge_yaml(),
            "benchmark.py": self.generate_benchmark_py(),
            "test_suite.py": self.generate_test_suite_py(),
        }
```

**Step 4: 运行测试确认通过**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_generator.py -v`
Expected: 4 passed

**Step 5: 提交**

```bash
git add agentforge/generator.py tests/test_generator.py
git commit -m "feat: add ConfigGenerator for template rendering"
```

---

### Task 4: InteractiveConfirm 交互确认

**Files:**
- Create: `agentforge/confirm.py`
- Test: `tests/test_confirm.py`

**Step 1: 写失败测试**

```python
# tests/test_confirm.py
from unittest.mock import patch
from agentforge.confirm import InteractiveConfirm


def test_confirm_yes(tmp_path):
    """用户输入 Y，文件被写入。"""
    files = {"test.yaml": "content: hello"}
    with patch("builtins.input", return_value="Y"):
        confirm = InteractiveConfirm(workdir=tmp_path)
        result = confirm.confirm_each(files)
    assert (tmp_path / "test.yaml").read_text() == "content: hello"
    assert result == {"test.yaml": "accepted"}


def test_confirm_empty_input_means_yes(tmp_path):
    """空回车 = Y。"""
    files = {"a.py": "# code"}
    with patch("builtins.input", return_value=""):
        confirm = InteractiveConfirm(workdir=tmp_path)
        result = confirm.confirm_each(files)
    assert (tmp_path / "a.py").read_text() == "# code"
    assert result == {"a.py": "accepted"}


def test_confirm_no_skips(tmp_path):
    """用户输入 n，文件不写入。"""
    files = {"skip.yaml": "content: skip"}
    with patch("builtins.input", return_value="n"):
        confirm = InteractiveConfirm(workdir=tmp_path)
        result = confirm.confirm_each(files)
    assert not (tmp_path / "skip.yaml").exists()
    assert result == {"skip.yaml": "rejected"}


def test_confirm_multiple_files(tmp_path):
    """多个文件逐个确认。"""
    files = {
        "challenge.yaml": "challenge: test",
        "benchmark.py": "# bench",
        "test_suite.py": "# tests",
    }
    responses = iter(["Y", "n", "Y"])
    with patch("builtins.input", side_effect=responses):
        confirm = InteractiveConfirm(workdir=tmp_path)
        result = confirm.confirm_each(files)
    assert (tmp_path / "challenge.yaml").exists()
    assert not (tmp_path / "benchmark.py").exists()
    assert (tmp_path / "test_suite.py").exists()
    assert result == {
        "challenge.yaml": "accepted",
        "benchmark.py": "rejected",
        "test_suite.py": "accepted",
    }
```

**Step 2: 运行测试确认失败**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_confirm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentforge.confirm'`

**Step 3: 实现 InteractiveConfirm**

```python
# agentforge/confirm.py
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


class InteractiveConfirm:
    """交互式逐步确认生成的配置文件。"""

    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)

    def confirm_each(self, files: dict[str, str]) -> dict[str, str]:
        """逐个展示文件内容，让用户确认 Y/n/edit。

        Returns: {filename: "accepted" | "rejected" | "edited"}
        """
        results = {}
        total = len(files)
        for i, (filename, content) in enumerate(files.items(), 1):
            print(f"\n{'━' * 40}")
            print(f"  {i}/{total}: {filename}")
            print(f"{'━' * 40}")
            print(content)
            print(f"{'━' * 40}")

            choice = input("确认？[Y/n/edit] > ").strip().lower()

            if choice in ("", "y", "yes"):
                self._write_file(filename, content)
                results[filename] = "accepted"
            elif choice in ("n", "no"):
                results[filename] = "rejected"
            elif choice == "edit":
                edited = self._open_editor(filename, content)
                if edited is not None:
                    self._write_file(filename, edited)
                    results[filename] = "edited"
                else:
                    results[filename] = "rejected"
            else:
                # 不认识的输入当 Y
                self._write_file(filename, content)
                results[filename] = "accepted"

        return results

    def _write_file(self, filename: str, content: str) -> None:
        path = self.workdir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def _open_editor(self, filename: str, content: str) -> str | None:
        """打开 $EDITOR 让用户编辑，返回编辑后内容。"""
        editor = os.environ.get("EDITOR", "vi")
        suffix = Path(filename).suffix or ".txt"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, prefix="agentforge_"
        ) as f:
            f.write(content)
            tmp_path = f.name
        try:
            subprocess.run([editor, tmp_path], check=True)
            return Path(tmp_path).read_text()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
```

**Step 4: 运行测试确认通过**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_confirm.py -v`
Expected: 4 passed

**Step 5: 提交**

```bash
git add agentforge/confirm.py tests/test_confirm.py
git commit -m "feat: add InteractiveConfirm for step-by-step file confirmation"
```

---

### Task 5: CLI 修改——config_path 变为可选

**Files:**
- Modify: `agentforge/cli.py:24-31`
- Modify: `agentforge/daemon.py:10-11,31`
- Test: `tests/test_cli.py`

**Step 1: 写失败测试**

在 `tests/test_cli.py` 追加：

```python
def test_run_without_config_path(tmp_path):
    """不传 config_path 也能启动（触发 auto-setup）。"""
    from click.testing import CliRunner
    from agentforge.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--workdir", str(tmp_path)])
    # 不应该报 "Missing argument" 错误
    assert "Missing argument" not in (result.output or "")
```

**Step 2: 运行测试确认失败**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_cli.py::test_run_without_config_path -v`
Expected: FAIL — `Missing argument 'CONFIG_PATH'`

**Step 3: 修改 cli.py 和 daemon.py**

`agentforge/cli.py` 修改 `run` 命令：

```python
@cli.command()
@click.argument("config_path", type=click.Path(exists=True), required=False, default=None)
@click.option("--workdir", type=click.Path(), default=None)
def run(config_path, workdir):
    """Start the optimization daemon. If config_path is omitted, auto-setup runs first."""
    wd = _get_workdir(workdir)
    config = Path(config_path) if config_path else None
    daemon = Daemon(config_path=config, workdir=wd)
    daemon.start()
```

`agentforge/daemon.py` 修改 `__init__` 和 `start`：

```python
class Daemon:
    def __init__(self, config_path: Path | None, workdir: Path):
        self.config_path = config_path
        # ...（其余不变）

    def start(self):
        # ...（fork 部分不变）
        try:
            orchestrator = Orchestrator(self.config_path, self.workdir,
                                        stop_flag=lambda: self._should_stop)
            orchestrator.run()
        finally:
            self._cleanup_pid()
```

**Step 4: 运行测试确认通过**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_cli.py -v`
Expected: ALL passed

**Step 5: 提交**

```bash
git add agentforge/cli.py agentforge/daemon.py tests/test_cli.py
git commit -m "feat: make config_path optional in CLI run command"
```

---

### Task 6: Orchestrator 集成 auto-setup

**Files:**
- Modify: `agentforge/orchestrator.py:28-60`
- Test: `tests/test_orchestrator.py`

**Step 1: 写失败测试**

在 `tests/test_orchestrator.py` 追加：

```python
def test_orchestrator_auto_setup_when_no_config(tmp_path):
    """config_path=None 且 workdir 无 challenge.yaml 时触发 auto-setup。"""
    from unittest.mock import patch, MagicMock
    import json

    profile_data = {
        "description": "Test auto-setup project",
        "run_command": "python main.py",
        "eval_metric": "accuracy",
        "eval_direction": "maximize",
        "eval_method": "parse stdout",
        "suggested_target": 0.9,
        "writable": ["main.py"],
        "readonly": ["data/"],
        "metric_extraction": "score = 0.9",
        "import_checks": "",
    }
    af_dir = tmp_path / ".agentforge"
    af_dir.mkdir()

    def fake_codex(*args, **kwargs):
        (af_dir / "project_profile.json").write_text(json.dumps(profile_data))
        return MagicMock(returncode=0)

    def fake_confirm(files):
        # 模拟全部接受
        for fname, content in files.items():
            (tmp_path / fname).write_text(content)
        return {f: "accepted" for f in files}

    with patch("subprocess.run", side_effect=fake_codex), \
         patch("agentforge.orchestrator.InteractiveConfirm") as MockConfirm:
        mock_instance = MagicMock()
        mock_instance.confirm_each = fake_confirm
        MockConfirm.return_value = mock_instance

        orch = Orchestrator(config_path=None, workdir=tmp_path)
        # auto_setup 应该被调用，生成 challenge.yaml
        assert (tmp_path / "challenge.yaml").exists()


def test_orchestrator_skips_setup_when_config_provided(tmp_path):
    """config_path 已提供时不触发 auto-setup。"""
    config_content = {
        "challenge": {"name": "test", "description": "test desc"},
        "target": {"metric": "acc", "value": 0.9, "direction": "maximize"},
        "tests": {"smoke": "echo ok", "full": "echo ok", "benchmark": "echo ok"},
        "constraints": {"writable": ["x.py"], "read_only": ["data/"]},
    }
    import yaml
    config_path = tmp_path / "challenge.yaml"
    config_path.write_text(yaml.dump(config_content))

    orch = Orchestrator(config_path=config_path, workdir=tmp_path)
    assert orch.config.challenge_name == "test"
```

**Step 2: 运行测试确认失败**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_orchestrator.py::test_orchestrator_auto_setup_when_no_config -v`
Expected: FAIL — `Orchestrator.__init__` 不接受 `config_path=None`

**Step 3: 修改 Orchestrator**

```python
# agentforge/orchestrator.py 修改部分

from agentforge.analyzer import ProjectAnalyzer
from agentforge.generator import ConfigGenerator
from agentforge.confirm import InteractiveConfirm

class Orchestrator:
    def __init__(self, config_path: Path | None, workdir: Path,
                 stop_flag: Callable[[], bool] | None = None):
        self.workdir = workdir
        self.state_file = StateFile(workdir / ".agentforge" / "state.json")
        self._stop_flag = stop_flag or (lambda: False)

        # Auto-setup: 如果没有 config_path 且 workdir 下没有 challenge.yaml
        if config_path is None:
            config_path = self._auto_setup()

        self.config = load_config(config_path)

    def _auto_setup(self) -> Path:
        """自动分析项目并生成配置文件。"""
        challenge_path = self.workdir / "challenge.yaml"
        if challenge_path.exists():
            return challenge_path

        print("[AgentForge] 未找到 challenge.yaml，启动自动配置...")

        # Step 1: 分析项目
        print("[AgentForge] 正在分析项目结构（Codex read-only）...")
        analyzer = ProjectAnalyzer(workdir=self.workdir)
        profile = analyzer.analyze()
        print(f"[AgentForge] 分析完成: {profile.description}")

        # Step 2: 生成配置文件
        generator = ConfigGenerator(profile)
        files = generator.generate_all()

        # Step 3: 交互确认
        confirm = InteractiveConfirm(workdir=self.workdir)
        results = confirm.confirm_each(files)

        rejected = [f for f, r in results.items() if r == "rejected"]
        if "challenge.yaml" in rejected:
            raise RuntimeError("challenge.yaml 被拒绝，无法继续")

        print("[AgentForge] 配置已保存。开始优化...")
        return challenge_path
```

**Step 4: 运行测试确认通过**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_orchestrator.py -v`
Expected: ALL passed

**Step 5: 运行全量测试**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/ -v`
Expected: ALL passed（原有 102 + 新增测试全部通过）

**Step 6: 提交**

```bash
git add agentforge/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: integrate auto-setup into Orchestrator initialization"
```

---

### Task 7: 全量集成测试

**Files:**
- Test: `tests/test_integration.py`

**Step 1: 写集成测试**

在 `tests/test_integration.py` 追加：

```python
def test_auto_setup_end_to_end(tmp_path):
    """完整 auto-setup 流程：分析 → 生成 → 确认 → 加载配置。"""
    from unittest.mock import patch, MagicMock
    import json

    # 创建一个假项目
    (tmp_path / "train.py").write_text("print('training')")
    (tmp_path / "model.py").write_text("class Model: pass")
    (tmp_path / "data").mkdir()

    profile_data = {
        "description": "Fake ML training project",
        "run_command": "python train.py",
        "run_args": [],
        "eval_metric": "val_loss",
        "eval_direction": "minimize",
        "eval_method": "parse stdout for val_loss",
        "baseline_value": 2.5,
        "suggested_target": 1.8,
        "writable": ["train.py", "model.py"],
        "readonly": ["data/"],
        "python_cmd": "python3",
        "needs_gpu": False,
        "result_location": "stdout",
        "result_pattern": "",
        "metric_extraction": "score = 1.5",
        "import_checks": "",
    }

    af_dir = tmp_path / ".agentforge"
    af_dir.mkdir()

    def fake_codex(*args, **kwargs):
        (af_dir / "project_profile.json").write_text(json.dumps(profile_data))
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_codex), \
         patch("builtins.input", return_value="Y"):
        from agentforge.orchestrator import Orchestrator
        orch = Orchestrator(config_path=None, workdir=tmp_path)

    # 验证文件已生成
    assert (tmp_path / "challenge.yaml").exists()
    assert (tmp_path / "benchmark.py").exists()
    assert (tmp_path / "test_suite.py").exists()

    # 验证配置已正确加载
    assert orch.config.target_metric == "val_loss"
    assert orch.config.target_direction == "minimize"
    assert orch.config.target_value == 1.8
```

**Step 2: 运行测试**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_integration.py::test_auto_setup_end_to_end -v`
Expected: PASS

**Step 3: 运行全量测试确认无回归**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/ -v`
Expected: ALL passed

**Step 4: 提交**

```bash
git add tests/test_integration.py
git commit -m "test: add auto-setup end-to-end integration test"
```

---

## 文件修改汇总

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `agentforge/analyzer.py` | ProjectProfile + ProjectAnalyzer |
| 新建 | `agentforge/generator.py` | ConfigGenerator 模板渲染 |
| 新建 | `agentforge/confirm.py` | InteractiveConfirm 交互确认 |
| 新建 | `tests/test_analyzer.py` | 8 个测试 |
| 新建 | `tests/test_generator.py` | 4 个测试 |
| 新建 | `tests/test_confirm.py` | 4 个测试 |
| 修改 | `agentforge/cli.py` | config_path 变为可选参数 |
| 修改 | `agentforge/daemon.py` | config_path 类型改为 `Path | None` |
| 修改 | `agentforge/orchestrator.py` | 加入 `_auto_setup()` 方法 |
| 修改 | `tests/test_orchestrator.py` | 2 个新测试 |
| 修改 | `tests/test_integration.py` | 1 个新测试 |
