from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class HardwareInfo:
    device: str          # "cuda" | "cpu"
    gpu_model: str       # "A100 80GB" | ""
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
    status: str          # "ok" | "oom" | "nan" | "timeout" | "error"
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
            best=BestResult(0.0, 0, "", "", ""),
            current_round=0,
            score_trajectory=[],
            rounds=[],
            strategies_tried=[],
            budget=Budget(0, rounds_max, 0.0, gpu_hours_max, 0.0),
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
