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
