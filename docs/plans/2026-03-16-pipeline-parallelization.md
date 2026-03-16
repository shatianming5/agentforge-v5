# Pipeline 并行化 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 AgentForge 的 Phase 1+2 重构为流水线式并行架构，每个策略独立走完「Codex 实现 → 训练 → 评分」全流程，同时提供 CLI Rich Live 进度表格和结构化事件接口。

**Architecture:** 线程池 Pipeline。Phase 1 精简为只生成策略规格列表。新增 `PipelineOrchestrator` 管理 N 个 `PipelineWorker` 线程，每个 Worker 独立运行 Codex 实现 + 训练 + 评分。`EventBus`（`queue.Queue`）收集进度事件，分发给 Rich Live 显示和外部回调。

**Tech Stack:** Python threading, queue.Queue, subprocess.Popen, rich.live.Live, dataclasses

---

## Task 1: StrategySpec 数据类 + spec-only prompt

**Files:**
- Modify: `agentforge/agent.py:15-27` (在 Strategy 之前新增 StrategySpec)
- Modify: `agentforge/agent.py:29-168` (PromptBuilder 新增 build_spec_only 方法)
- Modify: `agentforge/agent.py:171-206` (OutputParser 新增 parse_specs 方法)
- Test: `tests/test_agent_spec.py`

**Step 1: Write failing tests**

```python
# tests/test_agent_spec.py
from agentforge.agent import StrategySpec, PromptBuilder, OutputParser


def test_strategy_spec_fields():
    spec = StrategySpec(
        name="cosine_lr",
        description="Use cosine annealing LR schedule",
        approach="Replace StepLR with CosineAnnealingLR, warmup 5 epochs",
        category="optim",
        risk="low",
        estimated_train_command="python train.py --lr-schedule=cosine",
    )
    assert spec.name == "cosine_lr"
    assert spec.category == "optim"


def test_parse_specs_valid():
    raw = (
        'AGENTFORGE_SUMMARY_BEGIN\n'
        '[{"name": "cosine_lr", "description": "cosine schedule", '
        '"approach": "replace step with cosine", "category": "optim", '
        '"risk": "low", "estimated_train_command": "python train.py"}]\n'
        'AGENTFORGE_SUMMARY_END'
    )
    specs = OutputParser.parse_specs(raw)
    assert len(specs) == 1
    assert isinstance(specs[0], StrategySpec)
    assert specs[0].name == "cosine_lr"


def test_parse_specs_missing_name_skipped():
    raw = (
        'AGENTFORGE_SUMMARY_BEGIN\n'
        '[{"description": "no name field", "approach": "x", '
        '"category": "optim", "risk": "low", "estimated_train_command": ""},'
        '{"name": "valid", "description": "ok", "approach": "y", '
        '"category": "arch", "risk": "high", "estimated_train_command": ""}]\n'
        'AGENTFORGE_SUMMARY_END'
    )
    specs = OutputParser.parse_specs(raw)
    assert len(specs) == 1
    assert specs[0].name == "valid"


def test_parse_specs_empty_raises():
    raw = 'AGENTFORGE_SUMMARY_BEGIN\n[]\nAGENTFORGE_SUMMARY_END'
    try:
        OutputParser.parse_specs(raw)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_agent_spec.py -v`
Expected: FAIL — `StrategySpec` not defined, `parse_specs` not defined

**Step 3: Implement StrategySpec + parse_specs + build_spec_only**

在 `agentforge/agent.py` 中：

1. 在 `Strategy` 类之前添加：

```python
@dataclass
class StrategySpec:
    """Phase 1 精简输出：只有策略规格，不含分支/运行时数据。"""
    name: str
    description: str
    approach: str
    category: str
    risk: str
    estimated_train_command: str = ""
```

2. 在 `PromptBuilder` 类中添加：

```python
    @staticmethod
    def build_spec_only(config: ChallengeConfig, state: SessionState) -> str:
        sections = [
            PromptBuilder._system_spec_section(config, state),
            PromptBuilder._hardware_section(state),
            PromptBuilder._state_section(state),
            PromptBuilder._last_round_section(state),
            PromptBuilder._compressed_history_section(state),
            PromptBuilder._taboo_section(state),
            PromptBuilder._hints_section(state),
            PromptBuilder._rules_spec_section(config, state),
        ]
        return "\n\n".join(s for s in sections if s)

    @staticmethod
    def _system_spec_section(config, state):
        return (
            f"SYSTEM: You are the Strategist agent in an AgentForge optimization session.\n"
            f"Your job: propose {state.N} different optimization strategies.\n"
            f"You do NOT need to write code or create branches.\n"
            f"Just describe each strategy clearly so another agent can implement it.\n\n"
            f"CHALLENGE:\n"
            f"  Name: {config.challenge_name}\n"
            f"  Description: {config.challenge_description}\n"
            f"  Target: {config.target_metric} {config.target_direction} {config.target_value}"
        )

    @staticmethod
    def _rules_spec_section(config, state):
        ro = ", ".join(config.read_only)
        return (
            f"RULES:\n"
            f"  - Do NOT modify read-only files: {ro}\n"
            f"  - Produce at least {min(state.N, 3)} different categories of strategies\n"
            f"  - At least {min(state.N, 2)} must be high-risk/high-reward\n"
            f"  - Do NOT write any code or create branches\n"
            f"  - Just describe each strategy\n\n"
            f"OUTPUT (CRITICAL — you MUST do this as your FINAL step):\n"
            f"  mkdir -p .agentforge\n"
            f"  Write a JSON array to the file: .agentforge/agent_output.json\n"
            f"  Each element must have these fields:\n"
            f'  {{\n'
            f'    "name": "descriptive_name",\n'
            f'    "description": "what this strategy does",\n'
            f'    "approach": "detailed implementation steps for another agent",\n'
            f'    "category": "optim",\n'
            f'    "risk": "high",\n'
            f'    "estimated_train_command": "python3 train.py"\n'
            f'  }}\n'
            f"  category must be one of: optim, arch, data, reg\n"
            f"  risk must be: high or low"
        )
```

3. 在 `OutputParser` 类中添加：

```python
    @staticmethod
    def parse_specs(raw: str) -> list[StrategySpec]:
        begin = raw.find(OutputParser.BEGIN_MARKER)
        end = raw.find(OutputParser.END_MARKER)
        if begin == -1 or end == -1:
            raise ValueError("No summary found in Agent output")
        json_str = raw[begin + len(OutputParser.BEGIN_MARKER):end].strip()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in Agent summary: {e}") from e
        if not isinstance(data, list) or not data:
            raise ValueError("Agent summary must be a non-empty JSON array")
        specs = []
        for d in data:
            if "name" not in d:
                continue
            specs.append(StrategySpec(
                name=d["name"],
                description=d.get("description", ""),
                approach=d.get("approach", ""),
                category=d.get("category", "unknown"),
                risk=d.get("risk", "medium"),
                estimated_train_command=d.get("estimated_train_command", ""),
            ))
        if not specs:
            raise ValueError("No valid strategy specs found in Agent output")
        return specs
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_agent_spec.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add agentforge/agent.py tests/test_agent_spec.py
git commit -m "feat: add StrategySpec, parse_specs, and build_spec_only for spec-only Phase 1"
```

---

## Task 2: CodexCLI.implement_strategy() — 单策略实现

**Files:**
- Modify: `agentforge/agent.py:209-282` (CodexCLI 类新增 implement_strategy)
- Test: `tests/test_implement_strategy.py`

**Step 1: Write failing test**

```python
# tests/test_implement_strategy.py
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from agentforge.agent import CodexCLI, StrategySpec, Strategy


def test_implement_strategy_with_preseeded_output(tmp_path):
    """Pre-seeded agent_output.json should be consumed and return a Strategy."""
    af_dir = tmp_path / ".agentforge"
    af_dir.mkdir()
    strategy_data = [{
        "name": "cosine_lr",
        "branch": "agentforge/iter-1/exp-0",
        "confidence": 0.8,
        "measured_vram_gb": 2.5,
        "measured_epoch_seconds": 30.0,
        "batch_size": 32,
        "resume_checkpoint": False,
        "category": "optim",
        "risk": "low",
        "train_command": "python train.py",
    }]
    (af_dir / "agent_output.json").write_text(json.dumps(strategy_data))

    spec = StrategySpec(
        name="cosine_lr",
        description="Use cosine annealing",
        approach="Replace StepLR with CosineAnnealingLR",
        category="optim",
        risk="low",
        estimated_train_command="python train.py",
    )
    result = CodexCLI.implement_strategy(
        spec=spec, index=0, round_num=1,
        cwd=tmp_path, config_context="test challenge", timeout=60,
    )
    assert isinstance(result, Strategy)
    assert result.name == "cosine_lr"
    assert result.branch == "agentforge/iter-1/exp-0"


def test_implement_strategy_builds_correct_prompt():
    """Prompt should contain strategy spec details."""
    spec = StrategySpec(
        name="wider_backbone",
        description="Increase model width",
        approach="Double hidden_dim in model.py",
        category="arch",
        risk="high",
        estimated_train_command="python train.py --width=512",
    )
    prompt = CodexCLI._build_implement_prompt(
        spec, index=2, round_num=3,
        config_context="Maximize accuracy on CIFAR-10",
    )
    assert "wider_backbone" in prompt
    assert "Double hidden_dim" in prompt
    assert "agentforge/iter-3/exp-2" in prompt
    assert "CIFAR-10" in prompt
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_implement_strategy.py -v`
Expected: FAIL — `implement_strategy` and `_build_implement_prompt` not defined

**Step 3: Implement CodexCLI.implement_strategy**

在 `agentforge/agent.py` 的 `CodexCLI` 类中添加：

```python
    @staticmethod
    def _build_implement_prompt(
        spec: StrategySpec, index: int, round_num: int,
        config_context: str,
    ) -> str:
        branch = f"agentforge/iter-{round_num}/exp-{index}"
        return (
            f"SYSTEM: You are the Implementer agent in an AgentForge optimization session.\n"
            f"You have shell access and 1 GPU (CUDA_VISIBLE_DEVICES=0).\n"
            f"Your job: implement exactly ONE strategy and verify it works.\n\n"
            f"CHALLENGE CONTEXT:\n{config_context}\n\n"
            f"STRATEGY TO IMPLEMENT:\n"
            f"  Name: {spec.name}\n"
            f"  Description: {spec.description}\n"
            f"  Approach: {spec.approach}\n"
            f"  Category: {spec.category}\n"
            f"  Risk: {spec.risk}\n"
            f"  Suggested train command: {spec.estimated_train_command}\n\n"
            f"INSTRUCTIONS:\n"
            f"  1. Create branch: {branch}\n"
            f"  2. Write the code changes on that branch\n"
            f"  3. Run a VERY SHORT trial (max_iters=50) to verify it works\n"
            f"  4. Measure: VRAM usage, loss trend, time per epoch\n"
            f"  5. Commit the working version\n"
            f"  6. Return to main branch when done\n\n"
            f"OUTPUT (CRITICAL — you MUST do this as your FINAL step):\n"
            f"  mkdir -p .agentforge\n"
            f"  Write a JSON array to .agentforge/agent_output.json with ONE element:\n"
            f'  [{{\n'
            f'    "name": "{spec.name}",\n'
            f'    "branch": "{branch}",\n'
            f'    "confidence": 0.8,\n'
            f'    "measured_vram_gb": <measured>,\n'
            f'    "measured_epoch_seconds": <measured>,\n'
            f'    "batch_size": <used>,\n'
            f'    "resume_checkpoint": false,\n'
            f'    "category": "{spec.category}",\n'
            f'    "risk": "{spec.risk}",\n'
            f'    "train_command": "<actual command used>"\n'
            f'  }}]'
        )

    @staticmethod
    def implement_strategy(
        spec: StrategySpec, index: int, round_num: int,
        cwd: Path, config_context: str, timeout: int = 43200,
    ) -> Strategy:
        prompt = CodexCLI._build_implement_prompt(
            spec, index, round_num, config_context,
        )
        raw_output = CodexCLI.run(
            prompt=prompt, cwd=cwd, timeout=timeout,
            env={"CUDA_VISIBLE_DEVICES": "0"},
        )
        strategies = OutputParser.parse(raw_output)
        if not strategies:
            raise RuntimeError(f"Codex produced no strategy for {spec.name}")
        return strategies[0]
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_implement_strategy.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add agentforge/agent.py tests/test_implement_strategy.py
git commit -m "feat: add CodexCLI.implement_strategy for single-strategy implementation"
```

---

## Task 3: EventBus + PipelineEvent

**Files:**
- Create: `agentforge/pipeline.py`
- Test: `tests/test_pipeline_event.py`

**Step 1: Write failing tests**

```python
# tests/test_pipeline_event.py
import time
import threading
from agentforge.pipeline import PipelineEvent, EventBus


def test_pipeline_event_creation():
    evt = PipelineEvent(
        worker_index=0, strategy_name="cosine_lr",
        phase="training", timestamp=time.time(),
    )
    assert evt.phase == "training"
    assert evt.score is None
    assert evt.error is None


def test_event_to_dict():
    evt = PipelineEvent(
        worker_index=1, strategy_name="mixup",
        phase="done", timestamp=1234567890.0,
        score=0.85,
    )
    d = evt.to_dict()
    assert d["worker_index"] == 1
    assert d["phase"] == "done"
    assert d["score"] == 0.85


def test_eventbus_emit_and_subscribe():
    bus = EventBus()
    received = []
    bus.subscribe(lambda e: received.append(e))

    consumer = threading.Thread(target=bus.run_consumer, daemon=True)
    consumer.start()

    evt = PipelineEvent(
        worker_index=0, strategy_name="test",
        phase="implementing", timestamp=time.time(),
    )
    bus.emit(evt)
    bus.shutdown()
    consumer.join(timeout=2)

    assert len(received) == 1
    assert received[0].strategy_name == "test"


def test_eventbus_multiple_subscribers():
    bus = EventBus()
    received_a = []
    received_b = []
    bus.subscribe(lambda e: received_a.append(e))
    bus.subscribe(lambda e: received_b.append(e))

    consumer = threading.Thread(target=bus.run_consumer, daemon=True)
    consumer.start()

    bus.emit(PipelineEvent(
        worker_index=0, strategy_name="s1",
        phase="done", timestamp=time.time(), score=0.9,
    ))
    bus.shutdown()
    consumer.join(timeout=2)

    assert len(received_a) == 1
    assert len(received_b) == 1
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_pipeline_event.py -v`
Expected: FAIL — `agentforge.pipeline` module not found

**Step 3: Create pipeline.py with EventBus + PipelineEvent**

```python
# agentforge/pipeline.py
"""流水线并行架构：PipelineWorker, PipelineOrchestrator, EventBus."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

_SENTINEL = object()


@dataclass
class PipelineEvent:
    worker_index: int
    strategy_name: str
    phase: str  # "implementing" | "training" | "scoring" | "done" | "failed"
    timestamp: float
    progress: dict | None = None
    score: float | None = None
    error: str | None = None
    log_tail: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class EventBus:
    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._subscribers: list[Callable[[PipelineEvent], None]] = []

    def subscribe(self, callback: Callable[[PipelineEvent], None]) -> None:
        self._subscribers.append(callback)

    def emit(self, event: PipelineEvent) -> None:
        self._queue.put(event)

    def shutdown(self) -> None:
        self._queue.put(_SENTINEL)

    def run_consumer(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            for cb in self._subscribers:
                try:
                    cb(item)
                except Exception:
                    pass
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_pipeline_event.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add agentforge/pipeline.py tests/test_pipeline_event.py
git commit -m "feat: add PipelineEvent and EventBus for pipeline progress tracking"
```

---

## Task 4: SingleExperimentMonitor — 提取自 Monitor

**Files:**
- Modify: `agentforge/monitor.py:21-226` (提取 SingleExperimentMonitor)
- Test: `tests/test_single_monitor.py`

**Step 1: Write failing tests**

```python
# tests/test_single_monitor.py
import os
import time
from pathlib import Path
from unittest.mock import MagicMock
from agentforge.monitor import SingleExperimentMonitor


def test_check_nan_detects_nan(tmp_path):
    log_path = tmp_path / "exp.log"
    log_path.write_text("step 1 loss=0.5\nstep 2 loss=nan\n")

    mon = SingleExperimentMonitor(
        index=0, log_path=log_path, timeout=3600,
    )
    assert mon.check_nan() is True


def test_check_nan_no_nan(tmp_path):
    log_path = tmp_path / "exp.log"
    log_path.write_text("step 1 loss=0.5\nstep 2 loss=0.3\n")

    mon = SingleExperimentMonitor(
        index=0, log_path=log_path, timeout=3600,
    )
    assert mon.check_nan() is False


def test_is_timed_out():
    mon = SingleExperimentMonitor(
        index=0, log_path=Path("/dev/null"), timeout=1,
    )
    mon._start_time = time.time() - 10
    assert mon.is_timed_out() is True


def test_read_new_log_lines(tmp_path):
    log_path = tmp_path / "exp.log"
    log_path.write_text("line1\nline2\n")

    mon = SingleExperimentMonitor(
        index=0, log_path=log_path, timeout=3600,
    )
    lines = mon.read_new_lines()
    assert len(lines) == 2
    assert "line1" in lines[0]

    # Second read returns nothing (no new content)
    lines2 = mon.read_new_lines()
    assert len(lines2) == 0

    # Append new content
    with open(log_path, "a") as f:
        f.write("line3\n")
    lines3 = mon.read_new_lines()
    assert len(lines3) == 1
    assert "line3" in lines3[0]
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_single_monitor.py -v`
Expected: FAIL — `SingleExperimentMonitor` not found

**Step 3: Add SingleExperimentMonitor to monitor.py**

在 `agentforge/monitor.py` 文件末尾（`_add_event` 方法之后）添加：

```python
class SingleExperimentMonitor:
    """单个实验的监控器，供 PipelineWorker 使用。"""

    NAN_PATTERN = Monitor.NAN_PATTERN

    def __init__(self, index: int, log_path: Path, timeout: int):
        self.index = index
        self.log_path = log_path
        self.timeout = timeout
        self._start_time = time.time()
        self._last_pos = 0

    def is_timed_out(self) -> bool:
        return (time.time() - self._start_time) > self.timeout

    def check_nan(self) -> bool:
        if not self.log_path.exists():
            return False
        try:
            with open(self.log_path) as f:
                lines = f.readlines()[-100:]
            return any(self.NAN_PATTERN.search(line) for line in lines)
        except OSError:
            return False

    def read_new_lines(self) -> list[str]:
        if not self.log_path.exists():
            return []
        try:
            with open(self.log_path) as f:
                f.seek(self._last_pos)
                content = f.read()
                self._last_pos = f.tell()
            if not content.strip():
                return []
            return [l for l in content.strip().split("\n") if l.strip()]
        except OSError:
            return []

    def elapsed_seconds(self) -> float:
        return time.time() - self._start_time
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_single_monitor.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add agentforge/monitor.py tests/test_single_monitor.py
git commit -m "feat: extract SingleExperimentMonitor for per-worker monitoring"
```

---

## Task 5: PipelineWorker — 完整流水线线程

**Files:**
- Modify: `agentforge/pipeline.py` (添加 PipelineWorker 类)
- Test: `tests/test_pipeline_worker.py`

**Step 1: Write failing tests**

```python
# tests/test_pipeline_worker.py
import json
import os
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentforge.agent import Strategy, StrategySpec
from agentforge.pipeline import PipelineEvent, EventBus, PipelineWorker
from agentforge.state import HardwareInfo, StrategyResult


def _make_spec(name="test_strategy"):
    return StrategySpec(
        name=name, description="test", approach="do stuff",
        category="optim", risk="low", estimated_train_command="echo ok",
    )


def _make_hw():
    return HardwareInfo(
        device="cpu", gpu_model="", num_gpus=0,
        cpu_cores=4, ram_gb=16, disk_free_gb=100,
    )


def _make_strategy(name="test_strategy", branch="agentforge/iter-1/exp-0"):
    return Strategy(
        name=name, branch=branch, confidence=0.8,
        measured_vram_gb=0.0, measured_epoch_seconds=10.0,
        batch_size=1, resume_checkpoint=False,
        category="optim", risk="low", train_command="echo done",
    )


def test_worker_emits_events():
    """Worker should emit implementing → training → scoring → done events."""
    bus = EventBus()
    events = []
    bus.subscribe(lambda e: events.append(e))
    consumer = threading.Thread(target=bus.run_consumer, daemon=True)
    consumer.start()

    spec = _make_spec()
    hw = _make_hw()

    mock_strategy = _make_strategy()
    mock_config = MagicMock()
    mock_config.test_benchmark = "echo ok"
    mock_config.test_full = "echo ok"

    worker = PipelineWorker(
        index=0, spec=spec, config=mock_config,
        hw=hw, round_num=1, workdir=Path("/tmp/test"),
        event_bus=bus, timeout=60, config_context="test",
    )

    with patch.object(worker, "_implement", return_value=mock_strategy), \
         patch.object(worker, "_train", return_value=0), \
         patch.object(worker, "_score", return_value=0.85):
        worker.run()

    bus.shutdown()
    consumer.join(timeout=2)

    phases = [e.phase for e in events]
    assert "implementing" in phases
    assert "training" in phases
    assert "scoring" in phases
    assert "done" in phases


def test_worker_handles_implement_failure():
    """Worker should emit 'failed' when Codex implementation fails."""
    bus = EventBus()
    events = []
    bus.subscribe(lambda e: events.append(e))
    consumer = threading.Thread(target=bus.run_consumer, daemon=True)
    consumer.start()

    spec = _make_spec()
    hw = _make_hw()
    mock_config = MagicMock()

    worker = PipelineWorker(
        index=0, spec=spec, config=mock_config,
        hw=hw, round_num=1, workdir=Path("/tmp/test"),
        event_bus=bus, timeout=60, config_context="test",
    )

    with patch.object(worker, "_implement", side_effect=RuntimeError("Codex failed")):
        worker.run()

    bus.shutdown()
    consumer.join(timeout=2)

    phases = [e.phase for e in events]
    assert "failed" in phases
    failed_event = [e for e in events if e.phase == "failed"][0]
    assert "Codex failed" in failed_event.error


def test_worker_result_property():
    """Worker.result should return StrategyResult after completion."""
    bus = EventBus()
    consumer = threading.Thread(target=bus.run_consumer, daemon=True)
    consumer.start()

    spec = _make_spec()
    hw = _make_hw()
    mock_strategy = _make_strategy()
    mock_config = MagicMock()

    worker = PipelineWorker(
        index=0, spec=spec, config=mock_config,
        hw=hw, round_num=1, workdir=Path("/tmp/test"),
        event_bus=bus, timeout=60, config_context="test",
    )

    with patch.object(worker, "_implement", return_value=mock_strategy), \
         patch.object(worker, "_train", return_value=0), \
         patch.object(worker, "_score", return_value=0.85):
        worker.run()

    bus.shutdown()
    consumer.join(timeout=2)

    result = worker.result
    assert isinstance(result, StrategyResult)
    assert result.score == 0.85
    assert result.status == "ok"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_pipeline_worker.py -v`
Expected: FAIL — `PipelineWorker` not found

**Step 3: Implement PipelineWorker**

在 `agentforge/pipeline.py` 中，在 `EventBus` 类之后添加：

```python
import os
import shlex
import subprocess
from pathlib import Path

from agentforge.agent import CodexCLI, Strategy, StrategySpec
from agentforge.config import ChallengeConfig
from agentforge.experiment import ExperimentSetup, Experiment
from agentforge.monitor import SingleExperimentMonitor
from agentforge.scorer import Scorer
from agentforge.state import HardwareInfo, StrategyResult


class PipelineWorker:
    """一个策略的完整流水线：Codex 实现 → worktree → 训练 → 评分。"""

    CHECK_INTERVAL = 10

    def __init__(
        self,
        index: int,
        spec: StrategySpec,
        config: ChallengeConfig,
        hw: HardwareInfo,
        round_num: int,
        workdir: Path,
        event_bus: EventBus,
        timeout: int = 345600,
        config_context: str = "",
    ):
        self.index = index
        self.spec = spec
        self.config = config
        self.hw = hw
        self.round_num = round_num
        self.workdir = workdir
        self.event_bus = event_bus
        self.timeout = timeout
        self.config_context = config_context
        self._result: StrategyResult | None = None

    @property
    def result(self) -> StrategyResult | None:
        return self._result

    def _emit(self, phase: str, **kwargs) -> None:
        self.event_bus.emit(PipelineEvent(
            worker_index=self.index,
            strategy_name=self.spec.name,
            phase=phase,
            timestamp=time.time(),
            **kwargs,
        ))

    def run(self) -> None:
        try:
            # Phase: Implementing
            self._emit("implementing")
            strategy = self._implement()

            # Phase: Training
            self._emit("training")
            returncode = self._train(strategy)

            # Phase: Scoring
            self._emit("scoring")
            score = self._score(strategy, returncode)

            # Done
            self._result = StrategyResult(
                id=f"exp-{self.index}",
                strategy=strategy.name,
                branch=strategy.branch,
                score=score,
                status="ok" if returncode == 0 else "error",
                error=None if returncode == 0 else f"exit code {returncode}",
                actual_vram_gb=strategy.measured_vram_gb,
                actual_epoch_seconds=strategy.measured_epoch_seconds,
                actual_batch_size=strategy.batch_size,
            )
            self._emit("done", score=score)

        except Exception as e:
            self._result = StrategyResult(
                id=f"exp-{self.index}",
                strategy=self.spec.name,
                branch="",
                score=0.0,
                status="error",
                error=str(e),
                actual_vram_gb=0.0,
                actual_epoch_seconds=0.0,
                actual_batch_size=0,
            )
            self._emit("failed", error=str(e))

    def _implement(self) -> Strategy:
        return CodexCLI.implement_strategy(
            spec=self.spec,
            index=self.index,
            round_num=self.round_num,
            cwd=self.workdir,
            config_context=self.config_context,
            timeout=self.timeout // 2,
        )

    def _train(self, strategy: Strategy) -> int:
        runs_dir = self.workdir / ".agentforge" / "runs"
        experiment = ExperimentSetup.create(
            strategy=strategy,
            index=self.index,
            repo_path=self.workdir,
            workdir=runs_dir,
            hw=self.hw,
            train_command=shlex.split(self.config.test_benchmark),
        )
        self._experiment = experiment

        experiment.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(experiment.log_path, "w")
        try:
            proc = subprocess.Popen(
                experiment.train_command,
                cwd=str(experiment.workdir),
                env=experiment.env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp,
            )
            monitor = SingleExperimentMonitor(
                index=self.index,
                log_path=experiment.log_path,
                timeout=self.timeout,
            )
            while proc.poll() is None:
                if monitor.is_timed_out():
                    self._kill_proc(proc)
                    return -1
                if monitor.check_nan():
                    self._kill_proc(proc)
                    return -2
                new_lines = monitor.read_new_lines()
                if new_lines:
                    self._emit("training", log_tail="\n".join(new_lines[-3:]),
                               progress={"elapsed_s": int(monitor.elapsed_seconds())})
                time.sleep(self.CHECK_INTERVAL)
            return proc.returncode
        finally:
            log_file.close()

    def _score(self, strategy: Strategy, returncode: int) -> float:
        if returncode != 0:
            return 0.0
        if not hasattr(self, "_experiment"):
            return 0.0
        return Scorer.score(self._experiment, self.config, returncode, N=1)

    @staticmethod
    def _kill_proc(proc: subprocess.Popen) -> None:
        import signal
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            proc.wait(timeout=5)
        except (ProcessLookupError, PermissionError, OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_pipeline_worker.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add agentforge/pipeline.py tests/test_pipeline_worker.py
git commit -m "feat: add PipelineWorker with full implement→train→score lifecycle"
```

---

## Task 6: PipelineOrchestrator — 线程池管理

**Files:**
- Modify: `agentforge/pipeline.py` (添加 PipelineOrchestrator 类)
- Test: `tests/test_pipeline_orchestrator.py`

**Step 1: Write failing tests**

```python
# tests/test_pipeline_orchestrator.py
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentforge.agent import Strategy, StrategySpec
from agentforge.pipeline import (
    PipelineEvent, EventBus, PipelineWorker, PipelineOrchestrator,
)
from agentforge.state import HardwareInfo, StrategyResult


def _make_specs(n=2):
    return [
        StrategySpec(
            name=f"strategy_{i}", description=f"desc {i}",
            approach=f"approach {i}", category="optim",
            risk="low", estimated_train_command="echo ok",
        )
        for i in range(n)
    ]


def _make_hw():
    return HardwareInfo(
        device="cpu", gpu_model="", num_gpus=0,
        cpu_cores=4, ram_gb=16, disk_free_gb=100,
    )


def _make_result(index, score=0.85, status="ok"):
    return StrategyResult(
        id=f"exp-{index}", strategy=f"strategy_{index}",
        branch=f"agentforge/iter-1/exp-{index}",
        score=score, status=status, error=None,
        actual_vram_gb=0.0, actual_epoch_seconds=10.0,
        actual_batch_size=1,
    )


def test_orchestrator_runs_all_workers():
    specs = _make_specs(3)
    hw = _make_hw()
    mock_config = MagicMock()
    mock_config.test_benchmark = "echo ok"

    events = []

    def mock_worker_run(worker):
        worker._result = _make_result(worker.index)
        worker._emit("implementing")
        worker._emit("done", score=0.85)

    with patch.object(PipelineWorker, "run", autospec=True, side_effect=mock_worker_run):
        orch = PipelineOrchestrator(
            specs=specs, config=mock_config, hw=hw,
            round_num=1, workdir=Path("/tmp/test"),
            timeout=60, config_context="test",
        )
        orch.on_event(lambda e: events.append(e))
        results = orch.run()

    assert len(results) == 3
    assert all(isinstance(r, StrategyResult) for r in results)
    assert all(r.score == 0.85 for r in results)


def test_orchestrator_collects_events():
    specs = _make_specs(2)
    hw = _make_hw()
    mock_config = MagicMock()

    events = []

    def mock_worker_run(worker):
        worker._result = _make_result(worker.index)
        worker._emit("implementing")
        worker._emit("training")
        worker._emit("done", score=0.9)

    with patch.object(PipelineWorker, "run", autospec=True, side_effect=mock_worker_run):
        orch = PipelineOrchestrator(
            specs=specs, config=mock_config, hw=hw,
            round_num=1, workdir=Path("/tmp/test"),
            timeout=60, config_context="test",
        )
        orch.on_event(lambda e: events.append(e))
        orch.run()

    # 2 workers * 3 events each = 6 events
    assert len(events) == 6
    phases = [e.phase for e in events]
    assert phases.count("implementing") == 2
    assert phases.count("done") == 2
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_pipeline_orchestrator.py -v`
Expected: FAIL — `PipelineOrchestrator` not found

**Step 3: Implement PipelineOrchestrator**

在 `agentforge/pipeline.py` 中，在 `PipelineWorker` 类之后添加：

```python
class PipelineOrchestrator:
    """管理 N 个 PipelineWorker 线程，收集结果。"""

    def __init__(
        self,
        specs: list[StrategySpec],
        config: ChallengeConfig,
        hw: HardwareInfo,
        round_num: int,
        workdir: Path,
        timeout: int = 345600,
        config_context: str = "",
    ):
        self.specs = specs
        self.config = config
        self.hw = hw
        self.round_num = round_num
        self.workdir = workdir
        self.timeout = timeout
        self.config_context = config_context
        self._event_bus = EventBus()
        self._workers: list[PipelineWorker] = []

    def on_event(self, callback: Callable[[PipelineEvent], None]) -> None:
        self._event_bus.subscribe(callback)

    def run(self) -> list[StrategyResult]:
        # Start event consumer thread
        consumer = threading.Thread(
            target=self._event_bus.run_consumer, daemon=True,
        )
        consumer.start()

        # Create and start worker threads
        threads: list[threading.Thread] = []
        for i, spec in enumerate(self.specs):
            worker = PipelineWorker(
                index=i, spec=spec, config=self.config,
                hw=self.hw, round_num=self.round_num,
                workdir=self.workdir, event_bus=self._event_bus,
                timeout=self.timeout, config_context=self.config_context,
            )
            self._workers.append(worker)
            t = threading.Thread(target=worker.run, name=f"worker-{i}", daemon=True)
            threads.append(t)

        for t in threads:
            t.start()

        # Wait for all workers to complete (barrier)
        for t in threads:
            t.join()

        # Shutdown event bus
        self._event_bus.shutdown()
        consumer.join(timeout=5)

        # Collect results
        return [w.result for w in self._workers if w.result is not None]
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_pipeline_orchestrator.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add agentforge/pipeline.py tests/test_pipeline_orchestrator.py
git commit -m "feat: add PipelineOrchestrator for parallel worker management"
```

---

## Task 7: LiveProgressDisplay — Rich Live 表格

**Files:**
- Modify: `agentforge/display.py` (添加 LiveProgressDisplay 类)
- Test: `tests/test_live_display.py`

**Step 1: Write failing tests**

```python
# tests/test_live_display.py
import time
from agentforge.display import LiveProgressDisplay
from agentforge.pipeline import PipelineEvent


def test_live_display_update_state():
    display = LiveProgressDisplay(num_workers=3, use_rich=False)

    display.handle_event(PipelineEvent(
        worker_index=0, strategy_name="cosine_lr",
        phase="implementing", timestamp=time.time(),
    ))
    assert display._states[0]["phase"] == "implementing"
    assert display._states[0]["strategy"] == "cosine_lr"

    display.handle_event(PipelineEvent(
        worker_index=0, strategy_name="cosine_lr",
        phase="training", timestamp=time.time(),
        progress={"elapsed_s": 30},
    ))
    assert display._states[0]["phase"] == "training"

    display.handle_event(PipelineEvent(
        worker_index=0, strategy_name="cosine_lr",
        phase="done", timestamp=time.time(),
        score=0.85,
    ))
    assert display._states[0]["phase"] == "done"
    assert display._states[0]["score"] == 0.85


def test_live_display_plain_render():
    display = LiveProgressDisplay(num_workers=2, use_rich=False)

    display.handle_event(PipelineEvent(
        worker_index=0, strategy_name="strategy_a",
        phase="training", timestamp=time.time(),
        log_tail="step 5 loss=0.3",
    ))
    display.handle_event(PipelineEvent(
        worker_index=1, strategy_name="strategy_b",
        phase="done", timestamp=time.time(), score=0.9,
    ))

    text = display.render_plain()
    assert "strategy_a" in text
    assert "training" in text
    assert "strategy_b" in text
    assert "0.9" in text
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_live_display.py -v`
Expected: FAIL — `LiveProgressDisplay` not found

**Step 3: Implement LiveProgressDisplay**

在 `agentforge/display.py` 末尾添加：

```python
from agentforge.pipeline import PipelineEvent


class LiveProgressDisplay:
    """订阅 EventBus，实时展示各 Worker 的进度。"""

    def __init__(self, num_workers: int, use_rich: bool | None = None):
        if use_rich is None:
            self._use_rich = _HAS_RICH and _is_tty()
        else:
            self._use_rich = use_rich and _HAS_RICH
        self._num_workers = num_workers
        self._states: list[dict] = [
            {"phase": "pending", "strategy": "", "progress": "", "score": None}
            for _ in range(num_workers)
        ]
        self._live = None

    def handle_event(self, event: PipelineEvent) -> None:
        idx = event.worker_index
        if 0 <= idx < self._num_workers:
            self._states[idx]["phase"] = event.phase
            if event.strategy_name:
                self._states[idx]["strategy"] = event.strategy_name
            if event.score is not None:
                self._states[idx]["score"] = event.score
            if event.log_tail:
                self._states[idx]["progress"] = event.log_tail.split("\n")[-1][:40]
            elif event.progress:
                self._states[idx]["progress"] = str(event.progress)

        if self._use_rich:
            self._refresh_rich()
        else:
            self._print_plain()

    def start(self) -> None:
        if self._use_rich:
            from rich.live import Live
            self._live = Live(self._build_rich_table(), refresh_per_second=2)
            self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def render_plain(self) -> str:
        lines = []
        lines.append(f"  {'#':<4} {'Strategy':<24} {'Phase':<16} {'Progress':<20} {'Score':>8}")
        lines.append(f"  {'─'*4} {'─'*24} {'─'*16} {'─'*20} {'─'*8}")
        for i, s in enumerate(self._states):
            score_str = f"{s['score']:.4f}" if s['score'] is not None else "—"
            progress = s.get("progress", "")[:20]
            lines.append(
                f"  {i:<4} {s['strategy']:<24} {s['phase']:<16} {progress:<20} {score_str:>8}"
            )
        return "\n".join(lines)

    def _print_plain(self) -> None:
        import sys
        sys.stdout.write("\033[2J\033[H")  # clear screen
        sys.stdout.write(self.render_plain() + "\n")
        sys.stdout.flush()

    def _build_rich_table(self):
        if not _HAS_RICH:
            return ""
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=4)
        table.add_column("Strategy", min_width=20)
        table.add_column("Phase", width=16)
        table.add_column("Progress", width=20)
        table.add_column("Score", justify="right", width=10)
        for i, s in enumerate(self._states):
            phase_str = s["phase"]
            score_str = f"{s['score']:.4f}" if s['score'] is not None else "—"
            progress = s.get("progress", "")[:20]
            if s["phase"] == "done":
                phase_str = "[green]done[/green]"
            elif s["phase"] == "failed":
                phase_str = "[red]failed[/red]"
            table.add_row(str(i), s["strategy"], phase_str, progress, score_str)
        return table

    def _refresh_rich(self) -> None:
        if self._live:
            self._live.update(self._build_rich_table())
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_live_display.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add agentforge/display.py tests/test_live_display.py
git commit -m "feat: add LiveProgressDisplay for real-time pipeline progress"
```

---

## Task 8: Orchestrator 集成 — 重写 _run_round

**Files:**
- Modify: `agentforge/orchestrator.py:119-257` (重写 _run_round)
- Modify: `agentforge/agent.py:329-358` (AgentSession 新增 develop_specs)
- Test: `tests/test_orchestrator_pipeline.py`

**Step 1: Write failing tests**

```python
# tests/test_orchestrator_pipeline.py
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

from agentforge.agent import AgentSession, StrategySpec
from agentforge.pipeline import PipelineOrchestrator
from agentforge.state import StrategyResult


def test_agent_session_develop_specs():
    """AgentSession.develop_specs() should return StrategySpec list."""
    mock_config = MagicMock()
    mock_config.challenge_name = "Test"
    mock_config.challenge_description = "Test desc"
    mock_config.target_metric = "accuracy"
    mock_config.target_direction = "maximize"
    mock_config.target_value = 0.95
    mock_config.read_only = ["tests/"]

    mock_state = MagicMock()
    mock_state.N = 2
    mock_state.hardware = MagicMock()
    mock_state.hardware.device = "cpu"
    mock_state.hardware.num_gpus = 0
    mock_state.hardware.gpu_model = ""
    mock_state.hardware.cpu_cores = 4
    mock_state.hardware.ram_gb = 16
    mock_state.hardware.disk_free_gb = 100
    mock_state.best = MagicMock(score=0.5, round=1, experiment="exp-0")
    mock_state.score_trajectory = [0.5]
    mock_state.current_round = 1
    mock_state.rounds = []
    mock_state.strategies_tried = []
    mock_state.hints_pending = []

    session = AgentSession(mock_config, mock_state, Path("/tmp/test"))

    raw_output = (
        'AGENTFORGE_SUMMARY_BEGIN\n'
        '[{"name": "cosine_lr", "description": "d", "approach": "a", '
        '"category": "optim", "risk": "low", "estimated_train_command": "echo"}]\n'
        'AGENTFORGE_SUMMARY_END'
    )

    with patch("agentforge.agent.CodexCLI.run", return_value=raw_output):
        specs = session.develop_specs()

    assert len(specs) == 1
    assert isinstance(specs[0], StrategySpec)
    assert specs[0].name == "cosine_lr"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_orchestrator_pipeline.py -v`
Expected: FAIL — `develop_specs` not defined

**Step 3: Add AgentSession.develop_specs**

在 `agentforge/agent.py` 的 `AgentSession` 类中添加：

```python
    def develop_specs(self) -> list[StrategySpec]:
        prompt = PromptBuilder.build_spec_only(self.config, self.state)
        first_gpu = os.environ.get(
            "CUDA_VISIBLE_DEVICES", "0"
        ).split(",")[0].strip()
        try:
            raw_output = CodexCLI.run(
                prompt=prompt, cwd=self.workdir,
                timeout=1800,  # 30 minutes (spec-only is much faster)
                env={"CUDA_VISIBLE_DEVICES": first_gpu},
            )
            return OutputParser.parse_specs(raw_output)
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as e:
            print(f"[AgentForge] Spec generation failed: {e}")
            raise
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_orchestrator_pipeline.py -v`
Expected: 1 passed

**Step 5: Rewrite orchestrator._run_round to use PipelineOrchestrator**

在 `agentforge/orchestrator.py` 中，替换 `_run_round` 方法。顶部导入区新增：

```python
from agentforge.pipeline import PipelineOrchestrator
from agentforge.display import LiveProgressDisplay
```

替换 `_run_round` 方法为：

```python
    def _run_round(self, state: SessionState) -> SessionState:
        state.current_round += 1
        t0 = time.time()
        self.display.round_start(state.current_round)

        if AntiOscillation.check_plateau(state.score_trajectory):
            state.hints_pending.append(
                "WARNING: 3+ rounds without improvement. Try fundamentally different approaches."
            )
            self.display.warn("3+ rounds without improvement")

        # Sandbox: protect read-only files
        sandbox = Sandbox(self.workdir, self.config.read_only)
        sandbox.setup()

        # Phase 1: Generate strategy specs only (no code)
        self.display.phase1_start()
        agent = AgentSession(self.config, state, self.workdir)
        try:
            specs = agent.develop_specs()
        except (RuntimeError, ValueError) as e:
            self.display.warn(f"策略规格生成失败: {e}")
            sandbox.teardown()
            return state

        # Validate spec diversity
        from agentforge.agent import Strategy
        pseudo_strategies = [
            Strategy(
                name=s.name, branch="", confidence=0.5,
                measured_vram_gb=0, measured_epoch_seconds=0,
                batch_size=1, resume_checkpoint=False,
                category=s.category, risk=s.risk,
            )
            for s in specs
        ]
        warnings = StrategyValidator.validate(pseudo_strategies)
        if warnings:
            state.hints_pending.extend(
                f"Strategy validation: {w}" for w in warnings
            )
            for w in warnings:
                self.display.warn(w)

        sandbox.teardown()
        t1 = time.time()
        self.display.phase1_done((t1 - t0) / 60, len(specs))

        # Cleanup between phases
        Cleanup(self.workdir).between_phases()

        # Build config context for workers
        config_context = (
            f"Challenge: {self.config.challenge_name}\n"
            f"Description: {self.config.challenge_description}\n"
            f"Target: {self.config.target_metric} {self.config.target_direction} "
            f"{self.config.target_value}\n"
            f"Read-only: {', '.join(self.config.read_only)}"
        )

        # Pipeline Phase: parallel implement → train → score
        self.display.phase2_start(len(specs))
        pipeline = PipelineOrchestrator(
            specs=specs,
            config=self.config,
            hw=state.hardware,
            round_num=state.current_round,
            workdir=self.workdir,
            timeout=345600,
            config_context=config_context,
        )

        # Setup live progress display
        live_display = LiveProgressDisplay(num_workers=len(specs))
        pipeline.on_event(live_display.handle_event)
        live_display.start()

        try:
            results = pipeline.run()
        finally:
            live_display.stop()

        t2 = time.time()
        self.display.phase2_done((t2 - t1) / 60)

        # All-fail handling
        if SelfRepair.is_all_fail(results):
            diagnosis = SelfRepair.diagnose_all_fail(results)
            if diagnosis == "environmental":
                SelfRepair.rebuild_venv(self.workdir)

        # Select winners
        d = self.config.target_direction
        winners = self.select_winners(results, d)

        # Display results table
        self.display.round_results(
            results, winners, state.best.score, state.best.round,
        )

        # Update state
        round_result = RoundResult(
            round=state.current_round, experiments=results,
            winners=winners,
            phase1_minutes=(t1 - t0) / 60,
            phase2_minutes=(t2 - t1) / 60,
        )
        state.rounds.append(round_result)

        for r in results:
            if r.status == "ok" and r.score > 0 and is_better(r.score, state.best.score, d):
                commit = self._get_branch_commit(r.branch)
                state.best = BestResult(
                    score=r.score, round=state.current_round,
                    experiment=r.id, commit=commit, checkpoint="",
                )

        ok_scores = [r.score for r in results if r.status == "ok" and r.score > 0]
        if ok_scores:
            best_this_round = min(ok_scores) if d == "minimize" else max(ok_scores)
        else:
            best_this_round = best_initial_score(d)
        state.score_trajectory.append(best_this_round)

        for r in results:
            outcome = "ok" if r.status == "ok" else r.status
            state.strategies_tried.append(
                StrategyRecord(r.strategy, state.current_round, r.score, outcome)
            )

        state.budget.rounds_used += 1
        round_hours = (t2 - t0) / 3600 * max(state.hardware.num_gpus, 1)
        state.budget.gpu_hours_used += round_hours
        state.hints_pending.clear()

        return state
```

**Step 6: Run full test suite**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/ -v`
Expected: All tests pass

**Step 7: Commit**

```bash
git add agentforge/agent.py agentforge/orchestrator.py tests/test_orchestrator_pipeline.py
git commit -m "feat: integrate PipelineOrchestrator into orchestrator round loop"
```

---

## Task 9: 端到端集成验证

**Files:**
- Test: `tests/test_e2e_pipeline.py`

**Step 1: Write integration test**

```python
# tests/test_e2e_pipeline.py
"""端到端集成测试：验证完整的 spec → pipeline → results 流程。"""
import json
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentforge.agent import StrategySpec, Strategy
from agentforge.pipeline import PipelineOrchestrator, PipelineEvent
from agentforge.state import HardwareInfo, StrategyResult


def _make_hw():
    return HardwareInfo(
        device="cpu", gpu_model="", num_gpus=0,
        cpu_cores=4, ram_gb=16, disk_free_gb=100,
    )


def test_e2e_pipeline_all_succeed(tmp_path):
    """All workers succeed: 3 specs → 3 results with scores."""
    specs = [
        StrategySpec(name=f"s{i}", description=f"d{i}", approach=f"a{i}",
                     category="optim", risk="low",
                     estimated_train_command="echo ok")
        for i in range(3)
    ]

    config = MagicMock()
    config.test_benchmark = "echo ok"
    config.test_full = "echo ok"
    hw = _make_hw()
    events = []

    def mock_implement(spec, index, round_num, cwd, config_context, timeout):
        return Strategy(
            name=spec.name, branch=f"agentforge/iter-1/exp-{index}",
            confidence=0.8, measured_vram_gb=0, measured_epoch_seconds=5,
            batch_size=1, resume_checkpoint=False,
            category=spec.category, risk=spec.risk,
            train_command="echo done",
        )

    def mock_create(strategy, index, repo_path, workdir, hw, train_command):
        from agentforge.experiment import Experiment
        log_dir = workdir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return Experiment(
            index=index, strategy=strategy,
            workdir=tmp_path, log_path=log_dir / f"exp-{index}.log",
            env=dict(__import__("os").environ),
            train_command=["echo", "training_done"],
        )

    def mock_score(exp, config, returncode, N=1):
        return 0.85 + exp.index * 0.01

    with patch("agentforge.pipeline.CodexCLI.implement_strategy", side_effect=mock_implement), \
         patch("agentforge.pipeline.ExperimentSetup.create", side_effect=mock_create), \
         patch("agentforge.pipeline.Scorer.score", side_effect=mock_score):
        orch = PipelineOrchestrator(
            specs=specs, config=config, hw=hw,
            round_num=1, workdir=tmp_path,
            timeout=60, config_context="test",
        )
        orch.on_event(lambda e: events.append(e))
        results = orch.run()

    assert len(results) == 3
    scores = sorted([r.score for r in results])
    assert scores == [0.85, 0.86, 0.87]

    # Verify event flow: each worker should emit at least implementing + done
    for i in range(3):
        worker_events = [e for e in events if e.worker_index == i]
        phases = [e.phase for e in worker_events]
        assert "implementing" in phases
        assert "done" in phases or "training" in phases


def test_e2e_pipeline_one_fails(tmp_path):
    """One worker fails during implementation, others succeed."""
    specs = [
        StrategySpec(name=f"s{i}", description=f"d{i}", approach=f"a{i}",
                     category="optim", risk="low",
                     estimated_train_command="echo ok")
        for i in range(2)
    ]
    config = MagicMock()
    config.test_benchmark = "echo ok"
    hw = _make_hw()

    call_count = {"n": 0}

    def mock_implement(spec, index, round_num, cwd, config_context, timeout):
        if index == 0:
            raise RuntimeError("Codex crashed")
        return Strategy(
            name=spec.name, branch=f"agentforge/iter-1/exp-{index}",
            confidence=0.8, measured_vram_gb=0, measured_epoch_seconds=5,
            batch_size=1, resume_checkpoint=False,
            category=spec.category, risk=spec.risk,
            train_command="echo done",
        )

    def mock_create(strategy, index, repo_path, workdir, hw, train_command):
        from agentforge.experiment import Experiment
        log_dir = workdir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return Experiment(
            index=index, strategy=strategy,
            workdir=tmp_path, log_path=log_dir / f"exp-{index}.log",
            env=dict(__import__("os").environ),
            train_command=["echo", "done"],
        )

    with patch("agentforge.pipeline.CodexCLI.implement_strategy", side_effect=mock_implement), \
         patch("agentforge.pipeline.ExperimentSetup.create", side_effect=mock_create), \
         patch("agentforge.pipeline.Scorer.score", return_value=0.9):
        orch = PipelineOrchestrator(
            specs=specs, config=config, hw=hw,
            round_num=1, workdir=tmp_path,
            timeout=60, config_context="test",
        )
        results = orch.run()

    assert len(results) == 2
    failed = [r for r in results if r.status == "error"]
    ok = [r for r in results if r.status == "ok"]
    assert len(failed) == 1
    assert len(ok) == 1
    assert ok[0].score == 0.9
```

**Step 2: Run integration tests**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/test_e2e_pipeline.py -v`
Expected: 2 passed

**Step 3: Run full test suite**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/ -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/test_e2e_pipeline.py
git commit -m "test: add end-to-end integration tests for pipeline architecture"
```

---

## Task 10: 最终验证 + 清理

**Files:**
- Review all modified files
- Run full test suite

**Step 1: Run all tests**

Run: `cd /Users/shatianming/RRRR && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 2: Verify imports are clean**

Run: `cd /Users/shatianming/RRRR && python -c "from agentforge.pipeline import PipelineOrchestrator, PipelineWorker, EventBus, PipelineEvent; from agentforge.agent import StrategySpec; from agentforge.display import LiveProgressDisplay; from agentforge.monitor import SingleExperimentMonitor; print('All imports OK')"`
Expected: "All imports OK"

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: pipeline parallelization - complete implementation"
```
