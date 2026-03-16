from __future__ import annotations

import json
import subprocess
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

    @staticmethod
    def _unescape_code(s: str) -> str:
        """处理 LLM 生成 JSON 中代码片段的双转义换行符。"""
        if "\n" not in s and "\\n" in s:
            s = s.replace("\\n", "\n")
        if "\\t" in s and "\t" not in s:
            s = s.replace("\\t", "\t")
        return s

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
            metric_extraction=cls._unescape_code(data["metric_extraction"]),
            run_args=list(data.get("run_args", [])),
            baseline_value=data.get("baseline_value"),
            python_cmd=data.get("python_cmd", "python3"),
            needs_gpu=bool(data.get("needs_gpu", False)),
            result_location=data.get("result_location", "stdout"),
            result_pattern=data.get("result_pattern", ""),
            import_checks=cls._unescape_code(data.get("import_checks", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        from agentforge.stream import stream_run
        af_dir = self.workdir / ".agentforge"
        af_dir.mkdir(parents=True, exist_ok=True)
        prompt = self._build_prompt()
        result = stream_run(
            ["codex", "exec", "-s", "workspace-write", prompt],
            cwd=self.workdir, timeout=600, prefix="Codex",
        )
        if result.returncode != 0:
            print(f"[AgentForge] Codex 分析警告 (exit {result.returncode}): {result.stdout[-500:]}")
        return self._read_profile()
