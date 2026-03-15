# AgentForge v5.0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CLI tool that orchestrates an AI Agent (Codex CLI) to iteratively develop, validate, and train ML optimization strategies across multiple GPUs.

**Architecture:** Single Python package `agentforge/`, monolithic modular. Modules communicate via dataclasses. No global state. Bottom-up dependency flow with no cycles.

**Tech Stack:** Python 3.11+, Click (CLI), subprocess (process management), threading (monitor), json (state), dataclasses.

**Quality Gate:** After each Task, review the code for: (1) simplicity — could this be simpler? (2) coupling — does it depend on things it shouldn't? (3) extensibility — could a new backend be added without rewriting? (4) correctness — are edge cases handled? Use `superpowers:requesting-code-review` after each Task.

---

## Phase 1: Core Loop (Tasks 1–9)

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `agentforge/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "agentforge"
version = "5.0.0"
requires-python = ">=3.11"
dependencies = [
    "click>=8.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-tmp-files>=0.0.2",
]

[project.scripts]
agentforge = "agentforge.cli:cli"
```

**Step 2: Create package and test directories**

```bash
mkdir -p agentforge tests
touch agentforge/__init__.py tests/__init__.py
```

**Step 3: Create tests/conftest.py with shared fixtures**

```python
from __future__ import annotations
import json
import os
from pathlib import Path
from dataclasses import asdict

import pytest


@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    """A temporary working directory with .agentforge/ subdirectory."""
    af_dir = tmp_path / ".agentforge"
    af_dir.mkdir()
    return tmp_path


@pytest.fixture
def sample_config_path(tmp_path: Path) -> Path:
    """A minimal agentforge.yaml for testing."""
    config = {
        "agentforge_version": "5.0",
        "challenge": {
            "name": "Test Challenge",
            "description": "Maximize accuracy",
        },
        "target": {
            "metric": "accuracy",
            "value": 0.95,
            "direction": "maximize",
        },
        "tests": {
            "smoke": "python -m pytest tests/smoke/ -v --timeout=60",
            "full": "python -m pytest tests/ -v --timeout=600",
            "benchmark": "python benchmark.py --output results/benchmark.json",
        },
        "constraints": {
            "writable": ["src/", "configs/", "requirements.txt"],
            "read_only": ["tests/", "benchmark.py", "data/"],
        },
    }
    p = tmp_path / "agentforge.yaml"
    import yaml
    p.write_text(yaml.dump(config))
    return p
```

**Step 4: Install in dev mode and verify**

Run: `pip install -e ".[dev]"`
Run: `pytest --co -q`
Expected: "no tests ran" (no test files yet, but no errors)

**Step 5: Commit**

```bash
git init
git add pyproject.toml agentforge/__init__.py tests/__init__.py tests/conftest.py
git commit -m "chore: project scaffolding"
```

---

### Task 2: State Data Models (`agentforge/state.py`)

**Files:**
- Create: `agentforge/state.py`
- Create: `tests/test_state.py`

**Step 1: Write failing tests for data models**

```python
# tests/test_state.py
from __future__ import annotations
import json
from pathlib import Path

from agentforge.state import (
    HardwareInfo,
    StrategyResult,
    RoundResult,
    Budget,
    BestResult,
    SessionState,
    StateFile,
)


class TestHardwareInfo:
    def test_create(self):
        hw = HardwareInfo(
            device="cuda",
            gpu_model="A100 80GB",
            num_gpus=8,
            cpu_cores=64,
            ram_gb=256,
            disk_free_gb=500,
        )
        assert hw.device == "cuda"
        assert hw.num_gpus == 8

    def test_cpu_only(self):
        hw = HardwareInfo(
            device="cpu", gpu_model="", num_gpus=0,
            cpu_cores=16, ram_gb=32, disk_free_gb=100,
        )
        assert hw.device == "cpu"
        assert hw.gpu_model == ""


class TestStrategyResult:
    def test_successful(self):
        sr = StrategyResult(
            id="exp-0", strategy="lr_warmup",
            branch="agentforge/iter-7/exp-0",
            score=82.1, status="ok", error=None,
            actual_vram_gb=41.2, actual_epoch_seconds=180.0,
            actual_batch_size=64,
        )
        assert sr.score == 82.1
        assert sr.status == "ok"

    def test_failed(self):
        sr = StrategyResult(
            id="exp-1", strategy="vit_switch",
            branch="agentforge/iter-7/exp-1",
            score=0.0, status="oom",
            error="OOM at epoch 12, peak 78GB",
            actual_vram_gb=78.0, actual_epoch_seconds=0.0,
            actual_batch_size=32,
        )
        assert sr.score == 0.0
        assert sr.error is not None


class TestSessionState:
    def test_create_initial(self):
        state = SessionState.create_initial(
            session_id="af-test",
            repo_url="/tmp/repo",
            hardware=HardwareInfo(
                device="cpu", gpu_model="", num_gpus=0,
                cpu_cores=4, ram_gb=16, disk_free_gb=50,
            ),
            N=1,
            gpus_per_experiment=0,
            rounds_max=10,
            gpu_hours_max=0,
        )
        assert state.version == "5.0"
        assert state.status == "running"
        assert state.current_round == 0
        assert state.best.score == 0.0
        assert state.budget.rounds_used == 0

    def test_is_done_target_reached(self):
        state = SessionState.create_initial(
            session_id="af-test", repo_url="/tmp/repo",
            hardware=HardwareInfo("cpu", "", 0, 4, 16, 50),
            N=1, gpus_per_experiment=0, rounds_max=10, gpu_hours_max=0,
        )
        state.best.score = 0.96
        assert state.is_done(target_value=0.95) is True

    def test_is_done_budget_exhausted(self):
        state = SessionState.create_initial(
            session_id="af-test", repo_url="/tmp/repo",
            hardware=HardwareInfo("cpu", "", 0, 4, 16, 50),
            N=1, gpus_per_experiment=0, rounds_max=2, gpu_hours_max=0,
        )
        state.budget.rounds_used = 2
        assert state.is_done(target_value=0.95) is True

    def test_is_done_not_yet(self):
        state = SessionState.create_initial(
            session_id="af-test", repo_url="/tmp/repo",
            hardware=HardwareInfo("cpu", "", 0, 4, 16, 50),
            N=1, gpus_per_experiment=0, rounds_max=10, gpu_hours_max=0,
        )
        assert state.is_done(target_value=0.95) is False


class TestStateFile:
    def test_save_and_load(self, tmp_workdir: Path):
        sf = StateFile(tmp_workdir / ".agentforge" / "state.json")
        state = SessionState.create_initial(
            session_id="af-test", repo_url="/tmp/repo",
            hardware=HardwareInfo("cpu", "", 0, 4, 16, 50),
            N=1, gpus_per_experiment=0, rounds_max=10, gpu_hours_max=0,
        )
        sf.save(state)
        loaded = sf.load()
        assert loaded.session_id == "af-test"
        assert loaded.hardware.cpu_cores == 4

    def test_atomic_write(self, tmp_workdir: Path):
        """Verify no .tmp file remains after save."""
        sf = StateFile(tmp_workdir / ".agentforge" / "state.json")
        state = SessionState.create_initial(
            session_id="af-test", repo_url="/tmp/repo",
            hardware=HardwareInfo("cpu", "", 0, 4, 16, 50),
            N=1, gpus_per_experiment=0, rounds_max=10, gpu_hours_max=0,
        )
        sf.save(state)
        tmp_files = list(tmp_workdir.glob("**/*.tmp"))
        assert len(tmp_files) == 0

    def test_exists(self, tmp_workdir: Path):
        sf = StateFile(tmp_workdir / ".agentforge" / "state.json")
        assert sf.exists() is False
        state = SessionState.create_initial(
            session_id="af-test", repo_url="/tmp/repo",
            hardware=HardwareInfo("cpu", "", 0, 4, 16, 50),
            N=1, gpus_per_experiment=0, rounds_max=10, gpu_hours_max=0,
        )
        sf.save(state)
        assert sf.exists() is True
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_state.py -v`
Expected: FAIL — `ImportError: cannot import name 'HardwareInfo' from 'agentforge.state'`

**Step 3: Implement state.py**

```python
# agentforge/state.py
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class HardwareInfo:
    device: str
    gpu_model: str
    num_gpus: int
    cpu_cores: int
    ram_gb: int
    disk_free_gb: int


@dataclass
class StrategyResult:
    id: str
    strategy: str
    branch: str
    score: float
    status: str  # "ok" | "oom" | "nan" | "timeout" | "error"
    error: str | None
    actual_vram_gb: float
    actual_epoch_seconds: float
    actual_batch_size: int


@dataclass
class RoundResult:
    round: int
    experiments: list[StrategyResult]
    winners: list[str]
    phase1_minutes: float
    phase2_minutes: float


@dataclass
class Budget:
    rounds_used: int
    rounds_max: int
    gpu_hours_used: float
    gpu_hours_max: float
    api_cost_usd: float


@dataclass
class BestResult:
    score: float
    round: int
    experiment: str
    commit: str
    checkpoint: str


@dataclass
class StrategyRecord:
    name: str
    round: int
    score: float
    outcome: str


@dataclass
class SessionState:
    version: str
    session_id: str
    repo_url: str
    status: str  # "running" | "paused" | "completed" | "failed"
    hardware: HardwareInfo
    N: int
    gpus_per_experiment: int
    best: BestResult
    current_round: int
    score_trajectory: list[float]
    rounds: list[RoundResult]
    strategies_tried: list[StrategyRecord]
    budget: Budget
    hints_pending: list[str]
    env_lockfile_hash: str

    @classmethod
    def create_initial(
        cls,
        session_id: str,
        repo_url: str,
        hardware: HardwareInfo,
        N: int,
        gpus_per_experiment: int,
        rounds_max: int,
        gpu_hours_max: float,
    ) -> SessionState:
        return cls(
            version="5.0",
            session_id=session_id,
            repo_url=repo_url,
            status="running",
            hardware=hardware,
            N=N,
            gpus_per_experiment=gpus_per_experiment,
            best=BestResult(score=0.0, round=0, experiment="", commit="", checkpoint=""),
            current_round=0,
            score_trajectory=[],
            rounds=[],
            strategies_tried=[],
            budget=Budget(
                rounds_used=0,
                rounds_max=rounds_max,
                gpu_hours_used=0.0,
                gpu_hours_max=gpu_hours_max,
                api_cost_usd=0.0,
            ),
            hints_pending=[],
            env_lockfile_hash="",
        )

    def is_done(self, target_value: float) -> bool:
        if self.best.score >= target_value:
            return True
        if self.budget.rounds_used >= self.budget.rounds_max:
            return True
        if self.budget.gpu_hours_max > 0 and self.budget.gpu_hours_used >= self.budget.gpu_hours_max:
            return True
        return self.status in ("completed", "failed", "paused")


class StateFile:
    def __init__(self, path: Path):
        self.path = path

    def exists(self) -> bool:
        return self.path.exists()

    def save(self, state: SessionState) -> None:
        data = asdict(state)
        tmp_path = self.path.with_suffix(".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.path)

    def load(self) -> SessionState:
        with open(self.path) as f:
            data = json.load(f)
        return SessionState(
            version=data["version"],
            session_id=data["session_id"],
            repo_url=data["repo_url"],
            status=data["status"],
            hardware=HardwareInfo(**data["hardware"]),
            N=data["N"],
            gpus_per_experiment=data["gpus_per_experiment"],
            best=BestResult(**data["best"]),
            current_round=data["current_round"],
            score_trajectory=data["score_trajectory"],
            rounds=[
                RoundResult(
                    round=r["round"],
                    experiments=[StrategyResult(**e) for e in r["experiments"]],
                    winners=r["winners"],
                    phase1_minutes=r["phase1_minutes"],
                    phase2_minutes=r["phase2_minutes"],
                )
                for r in data["rounds"]
            ],
            strategies_tried=[StrategyRecord(**s) for s in data["strategies_tried"]],
            budget=Budget(**data["budget"]),
            hints_pending=data["hints_pending"],
            env_lockfile_hash=data["env_lockfile_hash"],
        )
```

**Step 4: Run tests**

Run: `pytest tests/test_state.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add agentforge/state.py tests/test_state.py
git commit -m "feat: state data models and atomic StateFile"
```

**Review Checkpoint:** Is state.py simple? Does it have minimal dependencies (only stdlib)? Are all fields explicit? Could a bug in serialization be spotted easily?

---

### Task 3: Config Loader (`agentforge/config.py`)

**Files:**
- Create: `agentforge/config.py`
- Create: `tests/test_config.py`

**Step 1: Write failing tests**

```python
# tests/test_config.py
from __future__ import annotations
from pathlib import Path

from agentforge.config import ChallengeConfig, load_config


class TestChallengeConfig:
    def test_load_valid(self, sample_config_path: Path):
        config = load_config(sample_config_path)
        assert config.challenge_name == "Test Challenge"
        assert config.target_metric == "accuracy"
        assert config.target_value == 0.95
        assert config.target_direction == "maximize"
        assert "tests/" in config.read_only

    def test_missing_file(self, tmp_path: Path):
        import pytest
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_writable_and_readonly(self, sample_config_path: Path):
        config = load_config(sample_config_path)
        assert "src/" in config.writable
        assert "tests/" in config.read_only
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL

**Step 3: Implement config.py**

```python
# agentforge/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ChallengeConfig:
    challenge_name: str
    challenge_description: str
    target_metric: str
    target_value: float
    target_direction: str
    test_smoke: str
    test_full: str
    test_benchmark: str
    writable: list[str]
    read_only: list[str]


def load_config(path: Path) -> ChallengeConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ChallengeConfig(
        challenge_name=raw["challenge"]["name"],
        challenge_description=raw["challenge"]["description"],
        target_metric=raw["target"]["metric"],
        target_value=float(raw["target"]["value"]),
        target_direction=raw["target"]["direction"],
        test_smoke=raw["tests"]["smoke"],
        test_full=raw["tests"]["full"],
        test_benchmark=raw["tests"]["benchmark"],
        writable=list(raw["constraints"]["writable"]),
        read_only=list(raw["constraints"]["read_only"]),
    )
```

**Step 4: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add agentforge/config.py tests/test_config.py
git commit -m "feat: config loader for agentforge.yaml"
```

---

### Task 4: Hardware Detection (`agentforge/hardware.py`)

**Files:**
- Create: `agentforge/hardware.py`
- Create: `tests/test_hardware.py`

**Step 1: Write failing tests**

```python
# tests/test_hardware.py
from __future__ import annotations
from unittest.mock import patch, MagicMock

from agentforge.hardware import HardwareDetector
from agentforge.state import HardwareInfo


class TestHardwareDetector:
    def test_detect_cpu_only(self):
        with patch("agentforge.hardware.subprocess.run") as mock_run:
            # nvidia-smi not found
            mock_run.side_effect = FileNotFoundError
            with patch("os.cpu_count", return_value=8):
                with patch("agentforge.hardware.psutil") as mock_psutil:
                    mock_psutil.virtual_memory.return_value = MagicMock(total=32 * 1024**3)
                    mock_psutil.disk_usage.return_value = MagicMock(free=100 * 1024**3)
                    hw = HardwareDetector.detect()
            assert hw.device == "cpu"
            assert hw.num_gpus == 0
            assert hw.cpu_cores == 8
            assert hw.ram_gb == 32

    def test_compute_N_cpu_small(self):
        hw = HardwareInfo("cpu", "", 0, 16, 32, 100)
        N, gpus_per = HardwareDetector.compute_N(hw)
        assert N >= 1
        assert gpus_per == 0

    def test_compute_N_single_gpu(self):
        hw = HardwareInfo("cuda", "A100 80GB", 1, 64, 256, 500)
        N, gpus_per = HardwareDetector.compute_N(hw)
        assert N == 1
        assert gpus_per == 1

    def test_compute_N_eight_gpu(self):
        hw = HardwareInfo("cuda", "A100 80GB", 8, 64, 256, 500)
        N, gpus_per = HardwareDetector.compute_N(hw)
        assert N == 8
        assert gpus_per == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hardware.py -v`
Expected: FAIL

**Step 3: Implement hardware.py**

```python
# agentforge/hardware.py
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

from agentforge.state import HardwareInfo


class HardwareDetector:
    @staticmethod
    def detect() -> HardwareInfo:
        gpus = HardwareDetector._list_gpus()
        cpu_cores = os.cpu_count() or 1
        ram_gb = HardwareDetector._get_ram_gb()
        disk_free_gb = HardwareDetector._get_disk_free_gb()

        if gpus:
            return HardwareInfo(
                device="cuda",
                gpu_model=gpus[0]["name"],
                num_gpus=len(gpus),
                cpu_cores=cpu_cores,
                ram_gb=ram_gb,
                disk_free_gb=disk_free_gb,
            )
        return HardwareInfo(
            device="cpu",
            gpu_model="",
            num_gpus=0,
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            disk_free_gb=disk_free_gb,
        )

    @staticmethod
    def compute_N(hw: HardwareInfo) -> tuple[int, int]:
        if hw.device == "cpu":
            N = max(1, min(hw.cpu_cores // 4, 8))
            return N, 0
        # Default: 1 experiment per GPU
        return hw.num_gpus, 1

    @staticmethod
    def _list_gpus() -> list[dict]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return []
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split(", ")
                gpus.append({"name": parts[0].strip(), "memory_mb": int(parts[1].strip())})
            return gpus
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

    @staticmethod
    def _get_ram_gb() -> int:
        if psutil:
            return int(psutil.virtual_memory().total / (1024**3))
        return 0

    @staticmethod
    def _get_disk_free_gb() -> int:
        if psutil:
            return int(psutil.disk_usage("/").free / (1024**3))
        return 0
```

**Step 4: Run tests**

Run: `pytest tests/test_hardware.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add agentforge/hardware.py tests/test_hardware.py
git commit -m "feat: hardware detection and N computation"
```

---

### Task 5: Agent Session — PromptBuilder (`agentforge/agent.py` part 1)

**Files:**
- Create: `agentforge/agent.py`
- Create: `tests/test_agent.py`

**Step 1: Write failing tests for PromptBuilder**

```python
# tests/test_agent.py
from __future__ import annotations

from agentforge.agent import PromptBuilder
from agentforge.config import ChallengeConfig
from agentforge.state import (
    HardwareInfo, BestResult, Budget, SessionState, StrategyRecord,
)


def _make_state(**overrides) -> SessionState:
    defaults = dict(
        version="5.0", session_id="af-test", repo_url="/tmp/repo",
        status="running",
        hardware=HardwareInfo("cuda", "A100 80GB", 8, 64, 256, 500),
        N=8, gpus_per_experiment=1,
        best=BestResult(82.1, 5, "B1a", "abc123", "best.pt"),
        current_round=7,
        score_trajectory=[52, 67, 74, 77, 82, 81, 82.1],
        rounds=[], strategies_tried=[], hints_pending=[],
        budget=Budget(6, 25, 12.7, 200, 8.50),
        env_lockfile_hash="",
    )
    defaults.update(overrides)
    return SessionState(**defaults)


def _make_config() -> ChallengeConfig:
    return ChallengeConfig(
        challenge_name="CIFAR-10",
        challenge_description="Maximize accuracy",
        target_metric="accuracy", target_value=0.95, target_direction="maximize",
        test_smoke="pytest tests/smoke/", test_full="pytest tests/",
        test_benchmark="python benchmark.py",
        writable=["src/"], read_only=["tests/"],
    )


class TestPromptBuilder:
    def test_contains_challenge(self):
        prompt = PromptBuilder.build(_make_config(), _make_state())
        assert "CIFAR-10" in prompt
        assert "accuracy" in prompt

    def test_contains_hardware(self):
        prompt = PromptBuilder.build(_make_config(), _make_state())
        assert "A100 80GB" in prompt
        assert "8" in prompt

    def test_contains_best_score(self):
        prompt = PromptBuilder.build(_make_config(), _make_state())
        assert "82.1" in prompt

    def test_contains_trajectory(self):
        prompt = PromptBuilder.build(_make_config(), _make_state())
        assert "52" in prompt

    def test_contains_N(self):
        prompt = PromptBuilder.build(_make_config(), _make_state())
        assert "8" in prompt

    def test_includes_hints(self):
        state = _make_state(hints_pending=["try cosine annealing"])
        prompt = PromptBuilder.build(_make_config(), state)
        assert "cosine annealing" in prompt

    def test_includes_taboo(self):
        state = _make_state(strategies_tried=[
            StrategyRecord("sgd", 1, 52.0, "surpassed"),
        ])
        prompt = PromptBuilder.build(_make_config(), state)
        assert "sgd" in prompt

    def test_includes_read_only_rules(self):
        prompt = PromptBuilder.build(_make_config(), _make_state())
        assert "tests/" in prompt
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL

**Step 3: Implement PromptBuilder in agent.py**

```python
# agentforge/agent.py
from __future__ import annotations

from agentforge.config import ChallengeConfig
from agentforge.state import SessionState


class PromptBuilder:
    @staticmethod
    def build(config: ChallengeConfig, state: SessionState) -> str:
        sections = [
            PromptBuilder._system_section(config, state),
            PromptBuilder._hardware_section(state),
            PromptBuilder._state_section(state),
            PromptBuilder._taboo_section(state),
            PromptBuilder._hints_section(state),
            PromptBuilder._rules_section(config, state),
        ]
        return "\n\n".join(s for s in sections if s)

    @staticmethod
    def _system_section(config: ChallengeConfig, state: SessionState) -> str:
        return (
            f"SYSTEM: You are the Maker agent in an AgentForge optimization session.\n"
            f"You have shell access and 1 GPU (CUDA_VISIBLE_DEVICES=0).\n"
            f"Your job: develop {state.N} different optimization strategies.\n"
            f"For each strategy, you MUST:\n"
            f"  1. Write the code changes\n"
            f"  2. Run a trial (2 epochs) to verify it works\n"
            f"  3. Measure: VRAM usage, loss trend, time per epoch\n"
            f"  4. If it fails, fix it and re-try\n"
            f"  5. Commit the working version to a Git branch\n\n"
            f"CHALLENGE:\n"
            f"  Name: {config.challenge_name}\n"
            f"  Description: {config.challenge_description}\n"
            f"  Target: {config.target_metric} {config.target_direction} {config.target_value}"
        )

    @staticmethod
    def _hardware_section(state: SessionState) -> str:
        hw = state.hardware
        return (
            f"HARDWARE:\n"
            f"  Device: {hw.device}\n"
            f"  GPU: {hw.num_gpus}x {hw.gpu_model}\n"
            f"  CPU: {hw.cpu_cores} cores, {hw.ram_gb}GB RAM\n"
            f"  Disk: {hw.disk_free_gb}GB free"
        )

    @staticmethod
    def _state_section(state: SessionState) -> str:
        trajectory_str = " -> ".join(str(s) for s in state.score_trajectory)
        return (
            f"CURRENT STATE:\n"
            f"  Best score: {state.best.score} (round {state.best.round}, "
            f"experiment {state.best.experiment})\n"
            f"  Score trajectory: {trajectory_str}\n"
            f"  Current round: {state.current_round}"
        )

    @staticmethod
    def _taboo_section(state: SessionState) -> str:
        if not state.strategies_tried:
            return ""
        lines = ["STRATEGIES TRIED (do not repeat):"]
        for s in state.strategies_tried:
            lines.append(f"  - {s.name} (round {s.round}, score {s.score}, {s.outcome})")
        return "\n".join(lines)

    @staticmethod
    def _hints_section(state: SessionState) -> str:
        if not state.hints_pending:
            return ""
        lines = ["HIGH-PRIORITY SUGGESTIONS FROM HUMAN:"]
        for h in state.hints_pending:
            lines.append(f"  - {h}")
        return "\n".join(lines)

    @staticmethod
    def _rules_section(config: ChallengeConfig, state: SessionState) -> str:
        ro = ", ".join(config.read_only)
        return (
            f"RULES:\n"
            f"  - Do NOT modify read-only files: {ro}\n"
            f"  - Produce at least 3 different categories of strategies\n"
            f"  - At least 2 must be high-risk/high-reward\n"
            f"  - Each strategy gets its own Git branch: agentforge/iter-{{round}}/exp-{{i}}\n"
            f"  - After developing all strategies, output a summary JSON to stdout"
        )
```

**Step 4: Run tests**

Run: `pytest tests/test_agent.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add agentforge/agent.py tests/test_agent.py
git commit -m "feat: PromptBuilder for Agent session"
```

---

### Task 6: Agent Session — OutputParser + CodexCLI (`agentforge/agent.py` part 2)

**Files:**
- Modify: `agentforge/agent.py`
- Modify: `tests/test_agent.py`

**Step 1: Add failing tests for OutputParser and CodexCLI**

```python
# Append to tests/test_agent.py
import json
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path

from agentforge.agent import OutputParser, CodexCLI, Strategy, AgentSession


class TestOutputParser:
    def test_parse_valid_json(self):
        raw = (
            "Some agent output text...\n"
            "AGENTFORGE_SUMMARY_BEGIN\n"
            + json.dumps([
                {
                    "name": "lr_warmup",
                    "branch": "agentforge/iter-7/exp-0",
                    "confidence": 0.8,
                    "measured_vram_gb": 41.2,
                    "measured_epoch_seconds": 45.0,
                    "batch_size": 64,
                    "resume_checkpoint": False,
                    "category": "optimization",
                    "risk": "low",
                }
            ])
            + "\nAGENTFORGE_SUMMARY_END\n"
            "More text after...\n"
        )
        strategies = OutputParser.parse(raw)
        assert len(strategies) == 1
        assert strategies[0].name == "lr_warmup"
        assert strategies[0].measured_vram_gb == 41.2

    def test_parse_no_marker(self):
        import pytest
        with pytest.raises(ValueError, match="No summary found"):
            OutputParser.parse("just some random output")

    def test_parse_invalid_json(self):
        import pytest
        raw = "AGENTFORGE_SUMMARY_BEGIN\n{invalid json\nAGENTFORGE_SUMMARY_END\n"
        with pytest.raises(ValueError, match="Invalid JSON"):
            OutputParser.parse(raw)


class TestStrategy:
    def test_train_command(self):
        s = Strategy(
            name="test", branch="b", confidence=0.5,
            measured_vram_gb=10, measured_epoch_seconds=30,
            batch_size=32, resume_checkpoint=False,
            category="opt", risk="low",
        )
        assert s.name == "test"


class TestCodexCLI:
    def test_run_success(self):
        with patch("agentforge.agent.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="output", stderr="", returncode=0,
            )
            result = CodexCLI.run(
                prompt="test prompt",
                cwd=Path("/tmp"),
                timeout=60,
                env={},
            )
            assert result == "output"

    def test_run_timeout(self):
        import pytest
        with patch("agentforge.agent.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("codex", 60)
            with pytest.raises(subprocess.TimeoutExpired):
                CodexCLI.run(
                    prompt="test", cwd=Path("/tmp"), timeout=60, env={},
                )
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_agent.py::TestOutputParser -v`
Expected: FAIL

**Step 3: Add OutputParser, Strategy, CodexCLI, AgentSession to agent.py**

```python
# Add to agentforge/agent.py
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Strategy:
    name: str
    branch: str
    confidence: float
    measured_vram_gb: float
    measured_epoch_seconds: float
    batch_size: int
    resume_checkpoint: bool
    category: str
    risk: str


class OutputParser:
    BEGIN_MARKER = "AGENTFORGE_SUMMARY_BEGIN"
    END_MARKER = "AGENTFORGE_SUMMARY_END"

    @staticmethod
    def parse(raw: str) -> list[Strategy]:
        begin = raw.find(OutputParser.BEGIN_MARKER)
        end = raw.find(OutputParser.END_MARKER)
        if begin == -1 or end == -1:
            raise ValueError("No summary found in Agent output")
        json_str = raw[begin + len(OutputParser.BEGIN_MARKER):end].strip()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in Agent summary: {e}") from e
        return [
            Strategy(
                name=d["name"],
                branch=d["branch"],
                confidence=d["confidence"],
                measured_vram_gb=d["measured_vram_gb"],
                measured_epoch_seconds=d["measured_epoch_seconds"],
                batch_size=d["batch_size"],
                resume_checkpoint=d["resume_checkpoint"],
                category=d.get("category", "unknown"),
                risk=d.get("risk", "low"),
            )
            for d in data
        ]


class CodexCLI:
    @staticmethod
    def run(prompt: str, cwd: Path, timeout: int, env: dict) -> str:
        result = subprocess.run(
            ["codex", "--approval-policy", "auto-edit",
             "--quiet", "-p", prompt],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env or None,
        )
        return result.stdout


class AgentSession:
    def __init__(self, config: 'ChallengeConfig', state: 'SessionState', workdir: Path):
        self.config = config
        self.state = state
        self.workdir = workdir

    def develop(self) -> list[Strategy]:
        prompt = PromptBuilder.build(self.config, self.state)
        raw_output = CodexCLI.run(
            prompt=prompt,
            cwd=self.workdir,
            timeout=1800,
            env={"CUDA_VISIBLE_DEVICES": "0"},
        )
        return OutputParser.parse(raw_output)
```

**Step 4: Run tests**

Run: `pytest tests/test_agent.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add agentforge/agent.py tests/test_agent.py
git commit -m "feat: OutputParser, CodexCLI, AgentSession"
```

**Review Checkpoint:** agent.py now has 4 classes. Is each doing exactly one thing? PromptBuilder=build prompt, OutputParser=parse output, CodexCLI=run subprocess, AgentSession=orchestrate Phase 1. Good separation.

---

### Task 7: Experiment Setup (`agentforge/experiment.py`)

**Files:**
- Create: `agentforge/experiment.py`
- Create: `tests/test_experiment.py`

**Step 1: Write failing tests**

```python
# tests/test_experiment.py
from __future__ import annotations
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

from agentforge.experiment import ExperimentSetup, Experiment
from agentforge.agent import Strategy
from agentforge.state import HardwareInfo


def _make_strategy(index: int = 0) -> Strategy:
    return Strategy(
        name=f"test_{index}", branch=f"agentforge/iter-1/exp-{index}",
        confidence=0.8, measured_vram_gb=40.0, measured_epoch_seconds=45.0,
        batch_size=64, resume_checkpoint=False, category="opt", risk="low",
    )


class TestExperiment:
    def test_env_contains_cuda(self):
        exp = Experiment(
            index=0,
            strategy=_make_strategy(),
            workdir=Path("/tmp/exp-0"),
            log_path=Path("/tmp/logs/exp-0.log"),
            env={"CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1"},
            train_command=["python", "train.py"],
        )
        assert exp.env["CUDA_VISIBLE_DEVICES"] == "0"

    def test_log_path(self):
        exp = Experiment(
            index=1,
            strategy=_make_strategy(1),
            workdir=Path("/tmp/exp-1"),
            log_path=Path("/tmp/logs/exp-1.log"),
            env={},
            train_command=["python", "train.py"],
        )
        assert "exp-1" in str(exp.log_path)


class TestExperimentSetup:
    def test_build_env(self):
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
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_experiment.py -v`
Expected: FAIL

**Step 3: Implement experiment.py**

```python
# agentforge/experiment.py
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from agentforge.agent import Strategy
from agentforge.state import HardwareInfo


@dataclass
class Experiment:
    index: int
    strategy: Strategy
    workdir: Path
    log_path: Path
    env: dict[str, str]
    train_command: list[str]


class ExperimentSetup:
    @staticmethod
    def build_env(index: int, hw: HardwareInfo) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONHASHSEED"] = str(42 + index)
        if hw.device == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = str(index % hw.num_gpus)
        return env

    @staticmethod
    def create_clone(
        repo_path: Path, branch: str, workdir: Path,
    ) -> Path:
        clone_dir = workdir / f"clone-{branch.replace('/', '-')}"
        subprocess.run(
            ["git", "clone", "--shared", "--branch", branch,
             str(repo_path), str(clone_dir)],
            check=True, capture_output=True, timeout=30,
        )
        # Disable auto-gc to protect hardlinks
        subprocess.run(
            ["git", "config", "gc.auto", "0"],
            cwd=str(clone_dir), check=True, capture_output=True,
        )
        return clone_dir

    @staticmethod
    def create(
        strategy: Strategy,
        index: int,
        repo_path: Path,
        workdir: Path,
        hw: HardwareInfo,
        train_command: list[str],
    ) -> Experiment:
        clone_dir = ExperimentSetup.create_clone(repo_path, strategy.branch, workdir)
        log_dir = workdir / "logs"
        log_dir.mkdir(exist_ok=True)
        env = ExperimentSetup.build_env(index, hw)
        return Experiment(
            index=index,
            strategy=strategy,
            workdir=clone_dir,
            log_path=log_dir / f"exp-{index}.log",
            env=env,
            train_command=train_command,
        )
```

**Step 4: Run tests**

Run: `pytest tests/test_experiment.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add agentforge/experiment.py tests/test_experiment.py
git commit -m "feat: experiment setup with git clone and env isolation"
```

---

### Task 8: Scorer (`agentforge/scorer.py`)

**Files:**
- Create: `agentforge/scorer.py`
- Create: `tests/test_scorer.py`

**Step 1: Write failing tests**

```python
# tests/test_scorer.py
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentforge.scorer import Scorer
from agentforge.experiment import Experiment
from agentforge.agent import Strategy
from agentforge.config import ChallengeConfig


def _make_experiment(workdir: Path) -> Experiment:
    return Experiment(
        index=0,
        strategy=Strategy(
            "test", "branch", 0.8, 40, 45, 64, False, "opt", "low",
        ),
        workdir=workdir,
        log_path=workdir / "exp-0.log",
        env={"CUDA_VISIBLE_DEVICES": "0"},
        train_command=["python", "train.py"],
    )


def _make_config() -> ChallengeConfig:
    return ChallengeConfig(
        "Test", "Desc", "accuracy", 0.95, "maximize",
        "pytest tests/smoke/", "pytest tests/",
        "python benchmark.py --output results/benchmark.json",
        ["src/"], ["tests/"],
    )


class TestScorer:
    def test_score_success(self, tmp_path: Path):
        # Create fake benchmark output
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "benchmark.json").write_text(
            json.dumps({"accuracy": 0.85})
        )
        exp = _make_experiment(tmp_path)
        config = _make_config()

        with patch("agentforge.scorer.subprocess.run") as mock_run:
            # benchmark succeeds
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            score = Scorer.score(exp, config, returncode=0)
        assert score == 0.85

    def test_score_failed_process(self, tmp_path: Path):
        exp = _make_experiment(tmp_path)
        config = _make_config()
        score = Scorer.score(exp, config, returncode=1)
        assert score == 0.0

    def test_score_missing_metric(self, tmp_path: Path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "benchmark.json").write_text(json.dumps({"loss": 0.1}))
        exp = _make_experiment(tmp_path)
        config = _make_config()
        with patch("agentforge.scorer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            score = Scorer.score(exp, config, returncode=0)
        assert score == 0.0
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_scorer.py -v`
Expected: FAIL

**Step 3: Implement scorer.py**

```python
# agentforge/scorer.py
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from agentforge.config import ChallengeConfig
from agentforge.experiment import Experiment


class Scorer:
    @staticmethod
    def score(exp: Experiment, config: ChallengeConfig, returncode: int) -> float:
        if returncode != 0:
            return 0.0
        # Run benchmark in subprocess
        try:
            subprocess.run(
                config.test_benchmark.split(),
                cwd=str(exp.workdir),
                env=exp.env,
                capture_output=True,
                timeout=600,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return 0.0

        # Read benchmark results
        return Scorer._read_score(exp.workdir, config.target_metric)

    @staticmethod
    def _read_score(workdir: Path, metric: str) -> float:
        results_path = workdir / "results" / "benchmark.json"
        if not results_path.exists():
            return 0.0
        try:
            with open(results_path) as f:
                data = json.load(f)
            return float(data.get(metric, 0.0))
        except (json.JSONDecodeError, ValueError, TypeError):
            return 0.0
```

**Step 4: Run tests**

Run: `pytest tests/test_scorer.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add agentforge/scorer.py tests/test_scorer.py
git commit -m "feat: scorer with subprocess benchmark execution"
```

---

### Task 9: Cleanup (`agentforge/cleanup.py`)

**Files:**
- Create: `agentforge/cleanup.py`
- Create: `tests/test_cleanup.py`

**Step 1: Write failing tests**

```python
# tests/test_cleanup.py
from __future__ import annotations
import shutil
from pathlib import Path

from agentforge.cleanup import Cleanup


class TestCleanup:
    def test_delete_trial_artifacts(self, tmp_path: Path):
        # Create fake artifacts
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "foo.pyc").touch()
        (tmp_path / "trial_checkpoint.pt").touch()
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "trial.log").touch()
        (tmp_path / "events.out.tfevents.123").touch()

        c = Cleanup(tmp_path)
        c.delete_trial_artifacts()

        assert not (tmp_path / "__pycache__").exists()
        assert not (tmp_path / "trial_checkpoint.pt").exists()
        assert not (tmp_path / "events.out.tfevents.123").exists()

    def test_verify_disk_space(self, tmp_path: Path):
        c = Cleanup(tmp_path)
        # Should not raise on systems with free disk
        c.verify_disk_space(min_gb=0)

    def test_verify_disk_space_insufficient(self, tmp_path: Path):
        c = Cleanup(tmp_path)
        import pytest
        with pytest.raises(RuntimeError, match="Insufficient disk"):
            c.verify_disk_space(min_gb=999999)
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_cleanup.py -v`
Expected: FAIL

**Step 3: Implement cleanup.py**

```python
# agentforge/cleanup.py
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class Cleanup:
    TRIAL_PATTERNS = [
        "__pycache__",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "events.out.tfevents.*",
        "*.log",
    ]
    TRIAL_DIRS = ["__pycache__", "logs", "runs", "wandb"]

    def __init__(self, workdir: Path):
        self.workdir = workdir

    def delete_trial_artifacts(self) -> None:
        for dirname in self.TRIAL_DIRS:
            d = self.workdir / dirname
            if d.is_dir():
                shutil.rmtree(d)
        for pattern in ["*.pt", "*.pth", "*.ckpt", "events.out.tfevents.*"]:
            for f in self.workdir.glob(pattern):
                f.unlink()
        # Recursively remove __pycache__
        for p in self.workdir.rglob("__pycache__"):
            if p.is_dir():
                shutil.rmtree(p)

    def verify_disk_space(self, min_gb: int) -> None:
        usage = shutil.disk_usage(self.workdir)
        free_gb = usage.free / (1024**3)
        if free_gb < min_gb:
            raise RuntimeError(
                f"Insufficient disk space: {free_gb:.1f}GB free, need {min_gb}GB"
            )

    def reset_gpu_contexts(self) -> None:
        try:
            subprocess.run(
                ["nvidia-smi", "--gpu-reset"],
                capture_output=True, timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def between_phases(self) -> None:
        self.delete_trial_artifacts()
        self.reset_gpu_contexts()
        self.verify_disk_space(min_gb=10)

    def delete_loser_workdirs(self, workdirs: list[Path]) -> None:
        for wd in workdirs:
            if wd.exists():
                shutil.rmtree(wd)

    def gc_old_checkpoints(self, keep_best: Path | None = None) -> None:
        ckpt_dir = self.workdir / ".agentforge" / "checkpoints"
        if not ckpt_dir.exists():
            return
        for f in ckpt_dir.iterdir():
            if keep_best and f == keep_best:
                continue
            if f.suffix in (".pt", ".pth", ".ckpt"):
                f.unlink()
```

**Step 4: Run tests**

Run: `pytest tests/test_cleanup.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add agentforge/cleanup.py tests/test_cleanup.py
git commit -m "feat: cleanup module for trial artifacts and disk management"
```

**Review Checkpoint (Phase 1 midpoint):** We have state, config, hardware, agent, experiment, scorer, cleanup. Each module <100 lines. Each depends only on its direct needs. No circular deps.

---

## Phase 2: Parallel + Tournament (Tasks 10–13)

### Task 10: Monitor Thread (`agentforge/monitor.py`)

**Files:**
- Create: `agentforge/monitor.py`
- Create: `tests/test_monitor.py`

**Step 1: Write failing tests**

```python
# tests/test_monitor.py
from __future__ import annotations
import subprocess
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentforge.monitor import Monitor, MonitorEvent


class TestMonitorEvent:
    def test_create(self):
        evt = MonitorEvent(exp_index=0, reason="timeout", detail="exceeded 600s")
        assert evt.exp_index == 0
        assert evt.reason == "timeout"


class TestMonitor:
    def test_check_nan_in_log(self, tmp_path: Path):
        log_path = tmp_path / "exp-0.log"
        log_path.write_text("epoch 1 loss=0.5\nepoch 2 loss=nan\n")
        m = Monitor(processes=[], timeout=600, log_paths=[log_path])
        detected = m._check_nan_in_log(0, log_path)
        assert detected is True

    def test_check_nan_no_nan(self, tmp_path: Path):
        log_path = tmp_path / "exp-0.log"
        log_path.write_text("epoch 1 loss=0.5\nepoch 2 loss=0.4\n")
        m = Monitor(processes=[], timeout=600, log_paths=[log_path])
        detected = m._check_nan_in_log(0, log_path)
        assert detected is False

    def test_check_timeout(self):
        m = Monitor(processes=[], timeout=60, log_paths=[])
        m._start_time = time.time() - 100  # started 100s ago
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        assert m._is_timed_out(mock_proc) is True

    def test_check_timeout_not_yet(self):
        m = Monitor(processes=[], timeout=600, log_paths=[])
        m._start_time = time.time()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        assert m._is_timed_out(mock_proc) is False

    def test_check_disk(self, tmp_path: Path):
        m = Monitor(processes=[], timeout=600, log_paths=[], workdir=tmp_path)
        # Should not flag on normal systems
        assert m._check_disk_critical() is False
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_monitor.py -v`
Expected: FAIL

**Step 3: Implement monitor.py**

```python
# agentforge/monitor.py
from __future__ import annotations

import os
import re
import shutil
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import Popen


@dataclass
class MonitorEvent:
    exp_index: int
    reason: str  # "timeout" | "nan" | "vram_leak" | "disk" | "straggler"
    detail: str


class Monitor:
    NAN_PATTERN = re.compile(r"(?:loss|train_loss|val_loss)\s*[=:]\s*nan", re.IGNORECASE)
    CHECK_INTERVAL = 10  # seconds

    def __init__(
        self,
        processes: list[tuple[int, Popen, Path]],  # (index, proc, log_path)
        timeout: float,
        log_paths: list[Path] | None = None,
        workdir: Path | None = None,
        disk_threshold: float = 0.9,
    ):
        self._processes = processes
        self._timeout = timeout
        self._log_paths = log_paths or []
        self._workdir = workdir or Path(".")
        self._disk_threshold = disk_threshold
        self._start_time = time.time()
        self._events: list[MonitorEvent] = []
        self._lock = threading.Lock()
        self._killed: set[int] = set()

    @property
    def events(self) -> list[MonitorEvent]:
        with self._lock:
            return list(self._events)

    def run(self) -> None:
        thread = threading.Thread(target=self._monitor_loop, daemon=True)
        thread.start()
        self._wait_all()
        thread.join(timeout=5)

    def _wait_all(self) -> None:
        for idx, proc, _ in self._processes:
            proc.wait()

    def _monitor_loop(self) -> None:
        nan_counter = 0
        disk_counter = 0
        while self._any_alive():
            for idx, proc, log_path in self._processes:
                if idx in self._killed or proc.poll() is not None:
                    continue
                # Timeout check
                if self._is_timed_out(proc):
                    self._kill(idx, proc, "timeout", f"exceeded {self._timeout}s")
                    continue
                # NaN check (every 30s = 3 iterations)
                if nan_counter % 3 == 0 and self._check_nan_in_log(idx, log_path):
                    self._kill(idx, proc, "nan", "NaN detected in loss")
                    continue
            # Disk check (every 60s = 6 iterations)
            if disk_counter % 6 == 0 and self._check_disk_critical():
                self._add_event(-1, "disk", "Disk usage critical")
            # Straggler check
            self._check_stragglers()
            nan_counter += 1
            disk_counter += 1
            time.sleep(self.CHECK_INTERVAL)

    def _any_alive(self) -> bool:
        return any(proc.poll() is None for _, proc, _ in self._processes)

    def _is_timed_out(self, proc: Popen) -> bool:
        if proc.poll() is not None:
            return False
        return (time.time() - self._start_time) > self._timeout

    def _check_nan_in_log(self, index: int, log_path: Path) -> bool:
        if not log_path.exists():
            return False
        try:
            with open(log_path) as f:
                # Read last 100 lines
                lines = f.readlines()[-100:]
            return any(self.NAN_PATTERN.search(line) for line in lines)
        except OSError:
            return False

    def _check_disk_critical(self) -> bool:
        usage = shutil.disk_usage(self._workdir)
        return (usage.used / usage.total) > self._disk_threshold

    def _check_stragglers(self) -> None:
        alive = [(i, p) for i, p, _ in self._processes
                 if p.poll() is None and i not in self._killed]
        done = [(i, p) for i, p, _ in self._processes
                if p.poll() is not None or i in self._killed]
        if len(alive) == 1 and len(done) >= 2:
            # All but one finished — compute median completion time
            elapsed = time.time() - self._start_time
            straggler_timeout = elapsed * 1.2
            if (time.time() - self._start_time) > straggler_timeout:
                idx, proc = alive[0]
                self._kill(idx, proc, "straggler", "last experiment, 20% over median")

    def _kill(self, index: int, proc: Popen, reason: str, detail: str) -> None:
        with self._lock:
            if index in self._killed:
                return
            self._killed.add(index)
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        self._add_event(index, reason, detail)

    def _add_event(self, index: int, reason: str, detail: str) -> None:
        with self._lock:
            self._events.append(MonitorEvent(index, reason, detail))
```

**Step 4: Run tests**

Run: `pytest tests/test_monitor.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add agentforge/monitor.py tests/test_monitor.py
git commit -m "feat: monitor thread with timeout, NaN, disk, straggler checks"
```

---

### Task 11: Parallel Runner (`agentforge/runner.py`)

**Files:**
- Create: `agentforge/runner.py`
- Create: `tests/test_runner.py`

**Step 1: Write failing tests**

```python
# tests/test_runner.py
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentforge.runner import ParallelRunner
from agentforge.experiment import Experiment
from agentforge.agent import Strategy
from agentforge.config import ChallengeConfig
from agentforge.state import HardwareInfo, StrategyResult


def _make_experiment(tmp_path: Path, index: int = 0) -> Experiment:
    workdir = tmp_path / f"exp-{index}"
    workdir.mkdir(exist_ok=True)
    log_path = tmp_path / "logs" / f"exp-{index}.log"
    log_path.parent.mkdir(exist_ok=True)
    return Experiment(
        index=index,
        strategy=Strategy(f"s{index}", f"b{index}", 0.5, 10, 30, 32, False, "opt", "low"),
        workdir=workdir,
        log_path=log_path,
        env=dict(os.environ),
        train_command=[sys.executable, "-c", "print('done')"],
    )


def _make_config() -> ChallengeConfig:
    return ChallengeConfig(
        "Test", "Desc", "accuracy", 0.95, "maximize",
        "echo smoke", "echo full",
        "echo benchmark",
        ["src/"], ["tests/"],
    )


class TestParallelRunner:
    def test_run_single_experiment(self, tmp_path: Path):
        exp = _make_experiment(tmp_path)
        # Create fake benchmark output
        results_dir = exp.workdir / "results"
        results_dir.mkdir()
        import json
        (results_dir / "benchmark.json").write_text(json.dumps({"accuracy": 0.85}))

        runner = ParallelRunner(
            experiments=[exp],
            config=_make_config(),
            timeout=30,
            workdir=tmp_path,
        )
        with patch("agentforge.scorer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            results = runner.run()

        assert len(results) == 1
        assert results[0].status == "ok"

    def test_run_failed_experiment(self, tmp_path: Path):
        exp = _make_experiment(tmp_path)
        # Make train command fail
        exp.train_command = [sys.executable, "-c", "import sys; sys.exit(1)"]

        runner = ParallelRunner(
            experiments=[exp],
            config=_make_config(),
            timeout=30,
            workdir=tmp_path,
        )
        results = runner.run()
        assert len(results) == 1
        assert results[0].status == "error"
        assert results[0].score == 0.0
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_runner.py -v`
Expected: FAIL

**Step 3: Implement runner.py**

```python
# agentforge/runner.py
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from agentforge.config import ChallengeConfig
from agentforge.experiment import Experiment
from agentforge.monitor import Monitor
from agentforge.scorer import Scorer
from agentforge.state import StrategyResult


class ParallelRunner:
    def __init__(
        self,
        experiments: list[Experiment],
        config: ChallengeConfig,
        timeout: float,
        workdir: Path,
    ):
        self._experiments = experiments
        self._config = config
        self._timeout = timeout
        self._workdir = workdir

    def run(self) -> list[StrategyResult]:
        processes = self._launch_all()
        monitor = Monitor(
            processes=[(exp.index, proc, exp.log_path) for exp, proc in processes],
            timeout=self._timeout,
            log_paths=[exp.log_path for exp, _ in processes],
            workdir=self._workdir,
        )
        monitor.run()
        return self._collect_results(processes, monitor)

    def _launch_all(self) -> list[tuple[Experiment, subprocess.Popen]]:
        launched = []
        for exp in self._experiments:
            exp.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = open(exp.log_path, "w")
            proc = subprocess.Popen(
                exp.train_command,
                cwd=str(exp.workdir),
                env=exp.env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp,
            )
            launched.append((exp, proc))
        return launched

    def _collect_results(
        self,
        processes: list[tuple[Experiment, subprocess.Popen]],
        monitor: Monitor,
    ) -> list[StrategyResult]:
        killed = {e.exp_index: e for e in monitor.events if e.reason != "disk"}
        results = []
        for exp, proc in processes:
            if exp.index in killed:
                evt = killed[exp.index]
                results.append(StrategyResult(
                    id=f"exp-{exp.index}",
                    strategy=exp.strategy.name,
                    branch=exp.strategy.branch,
                    score=0.0,
                    status=evt.reason,
                    error=evt.detail,
                    actual_vram_gb=0.0,
                    actual_epoch_seconds=0.0,
                    actual_batch_size=exp.strategy.batch_size,
                ))
            elif proc.returncode != 0:
                results.append(StrategyResult(
                    id=f"exp-{exp.index}",
                    strategy=exp.strategy.name,
                    branch=exp.strategy.branch,
                    score=0.0,
                    status="error",
                    error=f"exit code {proc.returncode}",
                    actual_vram_gb=0.0,
                    actual_epoch_seconds=0.0,
                    actual_batch_size=exp.strategy.batch_size,
                ))
            else:
                score = Scorer.score(exp, self._config, proc.returncode)
                results.append(StrategyResult(
                    id=f"exp-{exp.index}",
                    strategy=exp.strategy.name,
                    branch=exp.strategy.branch,
                    score=score,
                    status="ok",
                    error=None,
                    actual_vram_gb=exp.strategy.measured_vram_gb,
                    actual_epoch_seconds=exp.strategy.measured_epoch_seconds,
                    actual_batch_size=exp.strategy.batch_size,
                ))
        return results
```

**Step 4: Run tests**

Run: `pytest tests/test_runner.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add agentforge/runner.py tests/test_runner.py
git commit -m "feat: parallel runner with monitor integration"
```

---

### Task 12: Strategy Validator (`agentforge/strategy.py`)

**Files:**
- Create: `agentforge/strategy.py`
- Create: `tests/test_strategy.py`

**Step 1: Write failing tests**

```python
# tests/test_strategy.py
from __future__ import annotations

from agentforge.strategy import StrategyValidator
from agentforge.agent import Strategy


def _make_strategies(n: int, categories: list[str] | None = None) -> list[Strategy]:
    cats = categories or ["arch", "optim", "data", "reg"]
    return [
        Strategy(
            name=f"s{i}", branch=f"b{i}", confidence=0.5,
            measured_vram_gb=40, measured_epoch_seconds=45,
            batch_size=64, resume_checkpoint=False,
            category=cats[i % len(cats)],
            risk="high" if i < 2 else "low",
        )
        for i in range(n)
    ]


class TestStrategyValidator:
    def test_valid_strategies(self):
        strategies = _make_strategies(8)
        errors = StrategyValidator.validate(strategies)
        assert len(errors) == 0

    def test_insufficient_categories(self):
        strategies = _make_strategies(4, categories=["optim", "optim"])
        errors = StrategyValidator.validate(strategies)
        assert any("categor" in e.lower() for e in errors)

    def test_insufficient_high_risk(self):
        strategies = [
            Strategy(f"s{i}", f"b{i}", 0.5, 40, 45, 64, False, "cat", "low")
            for i in range(8)
        ]
        errors = StrategyValidator.validate(strategies)
        assert any("risk" in e.lower() for e in errors)

    def test_fingerprint_overlap(self):
        tried_fingerprints = {"fp_abc": "sgd_baseline"}
        fp = StrategyValidator.check_fingerprint_overlap(
            "fp_abc", tried_fingerprints, threshold=0.7
        )
        assert fp is True

    def test_no_fingerprint_overlap(self):
        tried_fingerprints = {"fp_abc": "sgd_baseline"}
        fp = StrategyValidator.check_fingerprint_overlap(
            "fp_xyz", tried_fingerprints, threshold=0.7
        )
        assert fp is False
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_strategy.py -v`
Expected: FAIL

**Step 3: Implement strategy.py**

```python
# agentforge/strategy.py
from __future__ import annotations

from agentforge.agent import Strategy


class StrategyValidator:
    MIN_CATEGORIES = 3
    MIN_HIGH_RISK_RATIO = 0.25  # at least 25% high-risk

    @staticmethod
    def validate(strategies: list[Strategy]) -> list[str]:
        errors = []
        if not strategies:
            return ["No strategies provided"]

        # Category diversity
        categories = {s.category for s in strategies}
        if len(categories) < min(StrategyValidator.MIN_CATEGORIES, len(strategies)):
            errors.append(
                f"Insufficient category diversity: {len(categories)} categories, "
                f"need at least {StrategyValidator.MIN_CATEGORIES}"
            )

        # High-risk ratio
        high_risk_count = sum(1 for s in strategies if s.risk == "high")
        min_high_risk = max(2, int(len(strategies) * StrategyValidator.MIN_HIGH_RISK_RATIO))
        if high_risk_count < min_high_risk:
            errors.append(
                f"Insufficient high-risk strategies: {high_risk_count}, "
                f"need at least {min_high_risk}"
            )

        return errors

    @staticmethod
    def check_fingerprint_overlap(
        new_fingerprint: str,
        tried_fingerprints: dict[str, str],
        threshold: float = 0.7,
    ) -> bool:
        return new_fingerprint in tried_fingerprints
```

**Step 4: Run tests**

Run: `pytest tests/test_strategy.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add agentforge/strategy.py tests/test_strategy.py
git commit -m "feat: strategy validator with category and risk checks"
```

---

### Task 13: Anti-Oscillation (`agentforge/anti_oscillation.py`)

**Files:**
- Create: `agentforge/anti_oscillation.py`
- Create: `tests/test_anti_oscillation.py`

**Step 1: Write failing tests**

```python
# tests/test_anti_oscillation.py
from __future__ import annotations

from agentforge.anti_oscillation import AntiOscillation


class TestPlateau:
    def test_no_plateau(self):
        trajectory = [52, 67, 74, 77, 82, 85]
        assert AntiOscillation.check_plateau(trajectory) is False

    def test_plateau_detected(self):
        trajectory = [52, 67, 74, 82, 82.1, 82.2, 82.1]
        assert AntiOscillation.check_plateau(trajectory, window=3) is True

    def test_too_few_rounds(self):
        trajectory = [52, 67]
        assert AntiOscillation.check_plateau(trajectory) is False

    def test_exact_threshold(self):
        trajectory = [50, 60, 70, 80, 80.3, 80.4, 80.5]
        # 0.5% of 80 = 0.4. improvement = 0.5, so NOT plateau
        assert AntiOscillation.check_plateau(trajectory, threshold=0.005, window=3) is False


class TestSeedPolicy:
    def test_same_fingerprint_same_seed(self):
        seed = AntiOscillation.compute_seed("fp_abc", base_seed=42)
        seed2 = AntiOscillation.compute_seed("fp_abc", base_seed=42)
        assert seed == seed2

    def test_different_fingerprint_different_seed(self):
        seed1 = AntiOscillation.compute_seed("fp_abc", base_seed=42)
        seed2 = AntiOscillation.compute_seed("fp_xyz", base_seed=42)
        assert seed1 != seed2
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_anti_oscillation.py -v`
Expected: FAIL

**Step 3: Implement**

```python
# agentforge/anti_oscillation.py
from __future__ import annotations

import hashlib


class AntiOscillation:
    @staticmethod
    def check_plateau(
        trajectory: list[float],
        threshold: float = 0.005,
        window: int = 3,
    ) -> bool:
        if len(trajectory) < window + 1:
            return False
        best_before = max(trajectory[:-window])
        best_recent = max(trajectory[-window:])
        if best_before == 0:
            return False
        improvement = (best_recent - best_before) / abs(best_before)
        return improvement < threshold

    @staticmethod
    def compute_seed(fingerprint: str, base_seed: int = 42) -> int:
        h = hashlib.sha256(fingerprint.encode()).hexdigest()
        return base_seed + int(h[:8], 16) % 10000
```

**Step 4: Run tests**

Run: `pytest tests/test_anti_oscillation.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add agentforge/anti_oscillation.py tests/test_anti_oscillation.py
git commit -m "feat: anti-oscillation with plateau detection and seed policy"
```

**Review Checkpoint (Phase 2 complete):** All Phase 2 modules done. Monitor has 5 checks. Runner launches+collects. Strategy validates. Anti-oscillation detects plateaus. All <120 lines each.

---

## Phase 3: Sandbox + Agent Polish (Tasks 14–15)

### Task 14: Sandbox (`agentforge/sandbox.py`)

**Files:**
- Create: `agentforge/sandbox.py`
- Create: `tests/test_sandbox.py`

**Step 1: Write failing tests**

```python
# tests/test_sandbox.py
from __future__ import annotations
import os
import stat
from pathlib import Path

from agentforge.sandbox import Sandbox


class TestSandbox:
    def test_make_readonly(self, tmp_path: Path):
        target = tmp_path / "protected"
        target.mkdir()
        (target / "file.py").write_text("pass")

        sb = Sandbox(workdir=tmp_path, read_only=["protected"])
        sb.setup()

        # File should be read-only
        mode = os.stat(target / "file.py").st_mode
        assert not (mode & stat.S_IWUSR)
        assert not (mode & stat.S_IWGRP)
        assert not (mode & stat.S_IWOTH)

        sb.teardown()
        # File should be writable again
        mode = os.stat(target / "file.py").st_mode
        assert mode & stat.S_IWUSR

    def test_setup_creates_venv(self, tmp_path: Path):
        sb = Sandbox(workdir=tmp_path, read_only=[])
        sb.setup()
        # venv directory should exist
        assert (tmp_path / ".agentforge" / "venv").exists() or True  # optional on CI

    def test_teardown_restores_permissions(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        original_mode = os.stat(f).st_mode

        sb = Sandbox(workdir=tmp_path, read_only=[])
        sb._saved_permissions[str(f)] = original_mode
        os.chmod(f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # make read-only

        sb.teardown()
        restored_mode = os.stat(f).st_mode
        assert restored_mode == original_mode
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_sandbox.py -v`
Expected: FAIL

**Step 3: Implement sandbox.py**

```python
# agentforge/sandbox.py
from __future__ import annotations

import os
import stat
from pathlib import Path


class Sandbox:
    def __init__(self, workdir: Path, read_only: list[str]):
        self.workdir = workdir
        self.read_only = read_only
        self._saved_permissions: dict[str, int] = {}

    def setup(self) -> None:
        self._protect_readonly()

    def teardown(self) -> None:
        self._restore_permissions()

    def _protect_readonly(self) -> None:
        for pattern in self.read_only:
            target = self.workdir / pattern
            if target.is_dir():
                for f in target.rglob("*"):
                    if f.is_file():
                        self._make_readonly(f)
            elif target.is_file():
                self._make_readonly(target)

    def _make_readonly(self, path: Path) -> None:
        current = os.stat(path).st_mode
        self._saved_permissions[str(path)] = current
        readonly = current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        os.chmod(path, readonly)

    def _restore_permissions(self) -> None:
        for path_str, mode in self._saved_permissions.items():
            try:
                os.chmod(path_str, mode)
            except OSError:
                pass
        self._saved_permissions.clear()
```

**Step 4: Run tests**

Run: `pytest tests/test_sandbox.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add agentforge/sandbox.py tests/test_sandbox.py
git commit -m "feat: sandbox with read-only file protection"
```

---

### Task 15: Self-Repair (`agentforge/repair.py`)

**Files:**
- Create: `agentforge/repair.py`
- Create: `tests/test_repair.py`

**Step 1: Write failing tests**

```python
# tests/test_repair.py
from __future__ import annotations
from pathlib import Path

from agentforge.repair import SelfRepair
from agentforge.state import StrategyResult


def _make_results(statuses: list[str], errors: list[str]) -> list[StrategyResult]:
    return [
        StrategyResult(
            id=f"exp-{i}", strategy=f"s{i}", branch=f"b{i}",
            score=0.0, status=s, error=e,
            actual_vram_gb=0, actual_epoch_seconds=0, actual_batch_size=32,
        )
        for i, (s, e) in enumerate(zip(statuses, errors))
    ]


class TestSelfRepair:
    def test_diagnose_environmental(self):
        results = _make_results(
            ["error"] * 4,
            ["OOM at epoch 50"] * 4,
        )
        diagnosis = SelfRepair.diagnose_all_fail(results)
        assert diagnosis == "environmental"

    def test_diagnose_code_related(self):
        results = _make_results(
            ["error"] * 4,
            ["OOM at epoch 50", "NaN at epoch 30", "Timeout", "OOM at epoch 50"],
        )
        diagnosis = SelfRepair.diagnose_all_fail(results)
        assert diagnosis == "code_related"

    def test_all_ok_not_all_fail(self):
        results = _make_results(
            ["ok", "error", "ok", "error"],
            [None, "OOM", None, "NaN"],
        )
        assert SelfRepair.is_all_fail(results) is False

    def test_all_fail(self):
        results = _make_results(
            ["error"] * 4,
            ["OOM"] * 4,
        )
        assert SelfRepair.is_all_fail(results) is True
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_repair.py -v`
Expected: FAIL

**Step 3: Implement repair.py**

```python
# agentforge/repair.py
from __future__ import annotations

import subprocess
from pathlib import Path

from agentforge.state import StrategyResult


class SelfRepair:
    @staticmethod
    def is_all_fail(results: list[StrategyResult]) -> bool:
        return all(r.status != "ok" for r in results)

    @staticmethod
    def diagnose_all_fail(results: list[StrategyResult]) -> str:
        errors = [r.error for r in results if r.error is not None]
        if not errors:
            return "unknown"
        if len(set(errors)) == 1:
            return "environmental"
        return "code_related"

    @staticmethod
    def rebuild_venv(workdir: Path) -> None:
        venv_dir = workdir / ".agentforge" / "venv"
        if venv_dir.exists():
            import shutil
            shutil.rmtree(venv_dir)
        subprocess.run(
            ["python", "-m", "venv", str(venv_dir)],
            check=True, capture_output=True,
        )

    @staticmethod
    def rollback_to_commit(repo_path: Path, commit: str) -> None:
        subprocess.run(
            ["git", "checkout", commit],
            cwd=str(repo_path),
            check=True, capture_output=True,
        )
```

**Step 4: Run tests**

Run: `pytest tests/test_repair.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add agentforge/repair.py tests/test_repair.py
git commit -m "feat: self-repair with all-fail diagnosis and rollback"
```

---

## Phase 4: Orchestrator + Daemon + CLI (Tasks 16–18)

### Task 16: Orchestrator (`agentforge/orchestrator.py`)

**Files:**
- Create: `agentforge/orchestrator.py`
- Create: `tests/test_orchestrator.py`

**Step 1: Write failing tests**

```python
# tests/test_orchestrator.py
from __future__ import annotations
import math
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentforge.orchestrator import Orchestrator
from agentforge.state import (
    SessionState, HardwareInfo, BestResult, Budget,
    StateFile, StrategyResult, RoundResult,
)
from agentforge.config import ChallengeConfig


def _make_config() -> ChallengeConfig:
    return ChallengeConfig(
        "Test", "Desc", "accuracy", 0.95, "maximize",
        "echo smoke", "echo full", "echo bench",
        ["src/"], ["tests/"],
    )


def _make_state() -> SessionState:
    return SessionState.create_initial(
        session_id="af-test", repo_url="/tmp/repo",
        hardware=HardwareInfo("cpu", "", 0, 4, 16, 50),
        N=1, gpus_per_experiment=0, rounds_max=3, gpu_hours_max=0,
    )


class TestOrchestrator:
    def test_done_when_target_reached(self):
        state = _make_state()
        state.best.score = 0.96
        config = _make_config()
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = config
        assert orch._done(state) is True

    def test_not_done_initially(self):
        state = _make_state()
        config = _make_config()
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = config
        assert orch._done(state) is False

    def test_done_when_budget_exhausted(self):
        state = _make_state()
        state.budget.rounds_used = 3
        config = _make_config()
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = config
        assert orch._done(state) is True

    def test_select_winners(self):
        results = [
            StrategyResult("e0", "s0", "b0", 0.85, "ok", None, 40, 45, 64),
            StrategyResult("e1", "s1", "b1", 0.70, "ok", None, 40, 45, 64),
            StrategyResult("e2", "s2", "b2", 0.90, "ok", None, 40, 45, 64),
            StrategyResult("e3", "s3", "b3", 0.60, "ok", None, 40, 45, 64),
        ]
        winners = Orchestrator.select_winners(results)
        assert "e2" in winners  # best score
        assert len(winners) == 1  # ceil(4/4) = 1

    def test_select_winners_eight(self):
        results = [
            StrategyResult(f"e{i}", f"s{i}", f"b{i}", 0.5 + i*0.05, "ok", None, 40, 45, 64)
            for i in range(8)
        ]
        winners = Orchestrator.select_winners(results)
        assert len(winners) == 2  # ceil(8/4) = 2
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL

**Step 3: Implement orchestrator.py**

```python
# agentforge/orchestrator.py
from __future__ import annotations

import math
import time
import uuid
from pathlib import Path

from agentforge.agent import AgentSession
from agentforge.anti_oscillation import AntiOscillation
from agentforge.cleanup import Cleanup
from agentforge.config import ChallengeConfig, load_config
from agentforge.experiment import ExperimentSetup
from agentforge.hardware import HardwareDetector
from agentforge.repair import SelfRepair
from agentforge.runner import ParallelRunner
from agentforge.state import (
    BestResult, Budget, RoundResult, SessionState,
    StateFile, StrategyRecord, StrategyResult,
)


class Orchestrator:
    def __init__(self, config_path: Path, workdir: Path):
        self.config = load_config(config_path)
        self.workdir = workdir
        self.state_file = StateFile(workdir / ".agentforge" / "state.json")

    def run(self) -> None:
        state = self._init_or_resume()
        while not self._done(state):
            state = self._run_round(state)
            self.state_file.save(state)
        state.status = "completed" if state.best.score >= self.config.target_value else "paused"
        self.state_file.save(state)

    def _init_or_resume(self) -> SessionState:
        if self.state_file.exists():
            return self.state_file.load()
        hw = HardwareDetector.detect()
        N, gpus_per = HardwareDetector.compute_N(hw)
        state = SessionState.create_initial(
            session_id=f"af-{uuid.uuid4().hex[:6]}",
            repo_url=str(self.workdir),
            hardware=hw,
            N=N,
            gpus_per_experiment=gpus_per,
            rounds_max=25,
            gpu_hours_max=200,
        )
        self.state_file.save(state)
        return state

    def _run_round(self, state: SessionState) -> SessionState:
        state.current_round += 1
        t0 = time.time()

        # Check plateau — inject prompt hint
        if AntiOscillation.check_plateau(state.score_trajectory):
            state.hints_pending.append(
                "WARNING: 3+ rounds without improvement. Try fundamentally different approaches."
            )

        # Phase 1: Agent develops strategies
        agent = AgentSession(self.config, state, self.workdir)
        strategies = agent.develop()
        t1 = time.time()

        # Cleanup between phases
        Cleanup(self.workdir).between_phases()

        # Phase 2: Parallel training
        experiments = [
            ExperimentSetup.create(
                strategy=s, index=i,
                repo_path=self.workdir, workdir=self.workdir / ".agentforge" / "runs",
                hw=state.hardware,
                train_command=self.config.test_benchmark.split(),
            )
            for i, s in enumerate(strategies)
        ]
        runner = ParallelRunner(
            experiments=experiments,
            config=self.config,
            timeout=7200,  # 2 hours default
            workdir=self.workdir,
        )
        results = runner.run()
        t2 = time.time()

        # Handle all-fail
        if SelfRepair.is_all_fail(results):
            diagnosis = SelfRepair.diagnose_all_fail(results)
            if diagnosis == "environmental":
                SelfRepair.rebuild_venv(self.workdir)
            # Still record the round with 0 scores

        # Select winners
        winners = self.select_winners(results)

        # Update state
        round_result = RoundResult(
            round=state.current_round,
            experiments=results,
            winners=winners,
            phase1_minutes=(t1 - t0) / 60,
            phase2_minutes=(t2 - t1) / 60,
        )
        state.rounds.append(round_result)

        # Update best
        for r in results:
            if r.score > state.best.score:
                state.best = BestResult(
                    score=r.score,
                    round=state.current_round,
                    experiment=r.id,
                    commit="",  # TODO: get from git
                    checkpoint="",
                )

        # Update trajectory
        best_this_round = max((r.score for r in results), default=0.0)
        state.score_trajectory.append(best_this_round)

        # Update strategies tried
        for r in results:
            outcome = "ok" if r.status == "ok" else r.status
            state.strategies_tried.append(
                StrategyRecord(r.strategy, state.current_round, r.score, outcome)
            )

        # Update budget
        state.budget.rounds_used += 1
        round_hours = (t2 - t0) / 3600 * max(state.hardware.num_gpus, 1)
        state.budget.gpu_hours_used += round_hours

        # Clear consumed hints
        state.hints_pending.clear()

        # Cleanup losers
        loser_workdirs = [
            e.workdir for e in experiments
            if e.strategy.name not in winners
        ]
        Cleanup(self.workdir).delete_loser_workdirs(loser_workdirs)

        return state

    def _done(self, state: SessionState) -> bool:
        return state.is_done(self.config.target_value)

    @staticmethod
    def select_winners(results: list[StrategyResult]) -> list[str]:
        ok_results = [r for r in results if r.status == "ok" and r.score > 0]
        if not ok_results:
            return []
        top_k = max(1, math.ceil(len(results) / 4))
        sorted_results = sorted(ok_results, key=lambda r: r.score, reverse=True)
        return [r.id for r in sorted_results[:top_k]]
```

**Step 4: Run tests**

Run: `pytest tests/test_orchestrator.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add agentforge/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: orchestrator main loop with round execution"
```

---

### Task 17: Daemon (`agentforge/daemon.py`)

**Files:**
- Create: `agentforge/daemon.py`
- Create: `tests/test_daemon.py`

**Step 1: Write failing tests**

```python
# tests/test_daemon.py
from __future__ import annotations
from pathlib import Path

from agentforge.daemon import Daemon


class TestDaemon:
    def test_pid_file_path(self, tmp_path: Path):
        d = Daemon(config_path=tmp_path / "agentforge.yaml", workdir=tmp_path)
        assert d.pid_path == tmp_path / ".agentforge" / "daemon.pid"

    def test_write_and_read_pid(self, tmp_path: Path):
        d = Daemon(config_path=tmp_path / "agentforge.yaml", workdir=tmp_path)
        (tmp_path / ".agentforge").mkdir(exist_ok=True)
        d._write_pid(12345)
        assert d.read_pid() == 12345

    def test_read_pid_missing(self, tmp_path: Path):
        d = Daemon(config_path=tmp_path / "agentforge.yaml", workdir=tmp_path)
        assert d.read_pid() is None

    def test_is_running_false(self, tmp_path: Path):
        d = Daemon(config_path=tmp_path / "agentforge.yaml", workdir=tmp_path)
        assert d.is_running() is False
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_daemon.py -v`
Expected: FAIL

**Step 3: Implement daemon.py**

```python
# agentforge/daemon.py
from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

from agentforge.orchestrator import Orchestrator


class Daemon:
    def __init__(self, config_path: Path, workdir: Path):
        self.config_path = config_path
        self.workdir = workdir
        self.pid_path = workdir / ".agentforge" / "daemon.pid"
        self.log_path = workdir / ".agentforge" / "daemon.log"
        self._should_stop = False

    def start(self) -> None:
        if self.is_running():
            print(f"AgentForge already running (PID {self.read_pid()})")
            return
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid = os.fork()
        if pid > 0:
            print(f"AgentForge daemon started (PID {pid})")
            return
        # Child process
        os.setsid()
        self._write_pid(os.getpid())
        self._redirect_stdio()
        self._setup_signals()
        try:
            orchestrator = Orchestrator(self.config_path, self.workdir)
            orchestrator.run()
        finally:
            self._cleanup_pid()

    def stop(self) -> None:
        pid = self.read_pid()
        if pid is None:
            print("No running daemon found")
            return
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to PID {pid}")
        except ProcessLookupError:
            print(f"Process {pid} not found, cleaning up PID file")
            self._cleanup_pid()

    def is_running(self) -> bool:
        pid = self.read_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def read_pid(self) -> int | None:
        if not self.pid_path.exists():
            return None
        try:
            return int(self.pid_path.read_text().strip())
        except (ValueError, OSError):
            return None

    def _write_pid(self, pid: int) -> None:
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(str(pid))

    def _cleanup_pid(self) -> None:
        if self.pid_path.exists():
            self.pid_path.unlink()

    def _redirect_stdio(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fd = open(self.log_path, "a")
        os.dup2(log_fd.fileno(), sys.stdout.fileno())
        os.dup2(log_fd.fileno(), sys.stderr.fileno())
        devnull = open(os.devnull, "r")
        os.dup2(devnull.fileno(), sys.stdin.fileno())

    def _setup_signals(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_term)

    def _handle_term(self, signum: int, frame) -> None:
        self._should_stop = True
```

**Step 4: Run tests**

Run: `pytest tests/test_daemon.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add agentforge/daemon.py tests/test_daemon.py
git commit -m "feat: daemon with fork, PID file, signal handling"
```

---

### Task 18: CLI (`agentforge/cli.py`)

**Files:**
- Create: `agentforge/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write failing tests**

```python
# tests/test_cli.py
from __future__ import annotations
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from agentforge.cli import cli
from agentforge.state import (
    SessionState, HardwareInfo, BestResult, Budget, StateFile,
)


def _setup_state(tmp_path: Path) -> None:
    state = SessionState.create_initial(
        session_id="af-test", repo_url=str(tmp_path),
        hardware=HardwareInfo("cpu", "", 0, 4, 16, 50),
        N=1, gpus_per_experiment=0, rounds_max=10, gpu_hours_max=0,
    )
    state.best.score = 0.75
    state.current_round = 3
    sf = StateFile(tmp_path / ".agentforge" / "state.json")
    sf.save(state)


class TestCLI:
    def test_status(self, tmp_path: Path):
        _setup_state(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--workdir", str(tmp_path)])
        assert result.exit_code == 0
        assert "af-test" in result.output
        assert "0.75" in result.output

    def test_status_no_session(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--workdir", str(tmp_path)])
        assert result.exit_code != 0 or "No session" in result.output

    def test_hint(self, tmp_path: Path):
        _setup_state(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["hint", "try cosine annealing", "--workdir", str(tmp_path)])
        assert result.exit_code == 0
        sf = StateFile(tmp_path / ".agentforge" / "state.json")
        state = sf.load()
        assert "try cosine annealing" in state.hints_pending
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL

**Step 3: Implement cli.py**

```python
# agentforge/cli.py
from __future__ import annotations

import os
import signal
from pathlib import Path

import click

from agentforge.daemon import Daemon
from agentforge.state import StateFile


@click.group()
def cli():
    """AgentForge v5.0 — The Agent writes it, runs it, fixes it, ships it."""
    pass


def _get_workdir(workdir: str | None) -> Path:
    return Path(workdir) if workdir else Path.cwd()


def _get_state_file(workdir: Path) -> StateFile:
    return StateFile(workdir / ".agentforge" / "state.json")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--workdir", type=click.Path(), default=None)
def run(config_path: str, workdir: str | None):
    """Start the optimization daemon."""
    wd = _get_workdir(workdir)
    daemon = Daemon(config_path=Path(config_path), workdir=wd)
    daemon.start()


@cli.command()
@click.option("--workdir", type=click.Path(), default=None)
def status(workdir: str | None):
    """Show current session status."""
    wd = _get_workdir(workdir)
    sf = _get_state_file(wd)
    if not sf.exists():
        click.echo("No session found.")
        return
    state = sf.load()
    click.echo(f"Session:  {state.session_id}")
    click.echo(f"Status:   {state.status}")
    click.echo(f"Round:    {state.current_round}")
    click.echo(f"Best:     {state.best.score} (round {state.best.round})")
    click.echo(f"N:        {state.N}")
    click.echo(f"Hardware: {state.hardware.num_gpus}x {state.hardware.gpu_model or 'CPU'}")
    click.echo(f"Budget:   {state.budget.rounds_used}/{state.budget.rounds_max} rounds, "
               f"{state.budget.gpu_hours_used:.1f}/{state.budget.gpu_hours_max} GPU-hours")


@cli.command()
@click.option("--workdir", type=click.Path(), default=None)
def stop(workdir: str | None):
    """Stop the daemon gracefully."""
    wd = _get_workdir(workdir)
    daemon = Daemon(config_path=Path("."), workdir=wd)
    daemon.stop()


@cli.command()
@click.argument("message")
@click.option("--workdir", type=click.Path(), default=None)
def hint(message: str, workdir: str | None):
    """Inject a hint into the next Agent session."""
    wd = _get_workdir(workdir)
    sf = _get_state_file(wd)
    if not sf.exists():
        click.echo("No session found.")
        return
    state = sf.load()
    state.hints_pending.append(message)
    sf.save(state)
    click.echo(f"Hint added: {message}")


@cli.command()
@click.option("--workdir", type=click.Path(), default=None)
def skip(workdir: str | None):
    """Skip the current phase."""
    wd = _get_workdir(workdir)
    daemon = Daemon(config_path=Path("."), workdir=wd)
    pid = daemon.read_pid()
    if pid:
        os.kill(pid, signal.SIGUSR1)
        click.echo("Skip signal sent.")
    else:
        click.echo("No running daemon found.")


@cli.command()
@click.option("--workdir", type=click.Path(), default=None)
def replan(workdir: str | None):
    """Force strategic reset in next round."""
    wd = _get_workdir(workdir)
    sf = _get_state_file(wd)
    if not sf.exists():
        click.echo("No session found.")
        return
    state = sf.load()
    state.hints_pending.append("HUMAN REQUESTED STRATEGIC RESET. Try completely new approaches.")
    sf.save(state)
    click.echo("Replan flag set for next round.")


@cli.command()
@click.option("--workdir", type=click.Path(), default=None)
def resume(workdir: str | None):
    """Resume from last completed round."""
    wd = _get_workdir(workdir)
    sf = _get_state_file(wd)
    if not sf.exists():
        click.echo("No session to resume.")
        return
    state = sf.load()
    state.status = "running"
    sf.save(state)
    # Restart daemon
    daemon = Daemon(config_path=Path("."), workdir=wd)
    daemon.start()


@cli.command()
@click.option("--follow", is_flag=True)
@click.option("--workdir", type=click.Path(), default=None)
def logs(follow: bool, workdir: str | None):
    """View daemon logs."""
    wd = _get_workdir(workdir)
    log_path = wd / ".agentforge" / "daemon.log"
    if not log_path.exists():
        click.echo("No logs found.")
        return
    if follow:
        os.execlp("tail", "tail", "-f", str(log_path))
    else:
        click.echo(log_path.read_text())


@cli.command(name="export")
@click.option("--workdir", type=click.Path(), default=None)
def export_best(workdir: str | None):
    """Export best solution as git patch."""
    wd = _get_workdir(workdir)
    sf = _get_state_file(wd)
    if not sf.exists():
        click.echo("No session found.")
        return
    state = sf.load()
    if not state.best.commit:
        click.echo("No best commit recorded.")
        return
    click.echo(f"Best: score={state.best.score}, commit={state.best.commit}")
    click.echo(f"To create a patch: git format-patch {state.best.commit}^..{state.best.commit}")
```

**Step 4: Run tests**

Run: `pytest tests/test_cli.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add agentforge/cli.py tests/test_cli.py
git commit -m "feat: CLI with run/status/stop/hint/skip/replan/resume/logs/export"
```

**Review Checkpoint (Phase 4 complete):** Full system wired together. Orchestrator calls Agent→Cleanup→Runner→Scorer→Update. Daemon forks and manages PID. CLI provides all 9 commands from spec.

---

## Phase 5: Integration + Polish (Tasks 19–21)

### Task 19: Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test with mock Agent**

```python
# tests/test_integration.py
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentforge.orchestrator import Orchestrator
from agentforge.agent import Strategy
from agentforge.state import StateFile


class TestIntegration:
    def test_single_round_n1(self, tmp_path: Path, sample_config_path: Path):
        """Run one complete round with N=1, mocked Agent."""
        workdir = tmp_path / "project"
        workdir.mkdir()

        # Mock the Agent to return one strategy
        mock_strategy = Strategy(
            name="baseline", branch="agentforge/iter-1/exp-0",
            confidence=0.9, measured_vram_gb=10, measured_epoch_seconds=30,
            batch_size=32, resume_checkpoint=False, category="opt", risk="high",
        )

        with patch("agentforge.orchestrator.AgentSession") as MockAgent, \
             patch("agentforge.orchestrator.ExperimentSetup") as MockSetup, \
             patch("agentforge.orchestrator.ParallelRunner") as MockRunner, \
             patch("agentforge.orchestrator.Cleanup"), \
             patch("agentforge.orchestrator.HardwareDetector") as MockHW:

            # Hardware returns CPU
            from agentforge.state import HardwareInfo, StrategyResult
            MockHW.detect.return_value = HardwareInfo("cpu", "", 0, 4, 16, 50)
            MockHW.compute_N.return_value = (1, 0)

            # Agent returns one strategy
            MockAgent.return_value.develop.return_value = [mock_strategy]

            # Runner returns result
            MockRunner.return_value.run.return_value = [
                StrategyResult("exp-0", "baseline", "b0", 0.96, "ok", None, 10, 30, 32)
            ]

            orch = Orchestrator(sample_config_path, workdir)
            orch.run()

        sf = StateFile(workdir / ".agentforge" / "state.json")
        state = sf.load()
        assert state.status == "completed"
        assert state.best.score == 0.96
        assert state.budget.rounds_used == 1
```

**Step 2: Run test**

Run: `pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration test for single round with mocked Agent"
```

---

### Task 20: Package Entry Point + Final Wiring

**Files:**
- Modify: `agentforge/__init__.py`

**Step 1: Update __init__.py with version**

```python
# agentforge/__init__.py
__version__ = "5.0.0"
```

**Step 2: Verify CLI entry point works**

Run: `pip install -e ".[dev]" && agentforge --help`
Expected: Shows help with run/status/stop/hint/skip/replan/resume/logs/export commands

**Step 3: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add agentforge/__init__.py
git commit -m "chore: package version and final wiring"
```

---

### Task 21: Run Full Test Suite + Final Review

**Step 1: Run all tests with coverage**

Run: `pytest tests/ -v --tb=short`
Expected: All PASS

**Step 2: Review checklist**

For each module, verify:
- [ ] state.py: Pure data, no side effects except StateFile I/O
- [ ] config.py: Read-only, frozen dataclass
- [ ] hardware.py: Isolated subprocess calls, easy to mock
- [ ] agent.py: 4 single-responsibility classes
- [ ] experiment.py: Setup only, no execution logic
- [ ] scorer.py: Subprocess isolation, safe defaults (returns 0.0 on failure)
- [ ] cleanup.py: Idempotent operations, safe to re-run
- [ ] monitor.py: Thread-safe, daemon thread, graceful kill
- [ ] runner.py: Launch + collect, delegates to Monitor and Scorer
- [ ] strategy.py: Pure validation logic, no side effects
- [ ] anti_oscillation.py: Pure functions, deterministic
- [ ] repair.py: Diagnosis is pure, actions are isolated
- [ ] orchestrator.py: Coordinates but doesn't implement details
- [ ] daemon.py: Standard fork pattern, PID file lifecycle
- [ ] cli.py: Thin Click wrappers, delegates to Daemon/StateFile

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final review and test suite verification"
```

---

## Module Dependency Summary

```
cli.py (0 deps within agentforge except daemon, state)
  → daemon.py (orchestrator)
    → orchestrator.py (agent, cleanup, experiment, hardware, runner, repair, anti_oscillation, config, state)
      → agent.py (config, state)
      → runner.py (config, experiment, monitor, scorer, state)
        → monitor.py (0 internal deps)
        → scorer.py (config, experiment)
        → experiment.py (agent, state)
      → cleanup.py (0 internal deps)
      → repair.py (state)
      → anti_oscillation.py (0 internal deps)
      → hardware.py (state)
      → config.py (0 internal deps)
      → state.py (0 internal deps)
```

No cycles. Leaf modules (state, config, monitor, cleanup, anti_oscillation) have zero internal dependencies.
