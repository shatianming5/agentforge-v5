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
    try:
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
    except KeyError as e:
        raise ValueError(f"Missing required config field: {e}") from e
