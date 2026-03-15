from __future__ import annotations
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    af_dir = tmp_path / ".agentforge"
    af_dir.mkdir()
    return tmp_path


@pytest.fixture
def sample_config_path(tmp_path: Path) -> Path:
    import yaml
    config = {
        "agentforge_version": "5.0",
        "challenge": {"name": "Test Challenge", "description": "Maximize accuracy"},
        "target": {"metric": "accuracy", "value": 0.95, "direction": "maximize"},
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
    p.write_text(yaml.dump(config))
    return p
