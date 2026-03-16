#!/usr/bin/env python3
"""
AgentForge v5.0 — Full Real CPU Demo

ALL components run for real, including Codex CLI:
  - HardwareDetector: real CPU/RAM/disk detection
  - Codex CLI: real AI agent develops optimization strategies
  - Git: real branch creation and cloning
  - ParallelRunner: real subprocess execution
  - Monitor: real process monitoring (NaN, timeout, disk, straggler)
  - Scorer: real test suite + benchmark evaluation
  - StateFile: real atomic JSON persistence
  - Sandbox: real file permission enforcement
  - Cleanup: real artifact cleanup
  - AntiOscillation: real plateau detection + deterministic seeds
  - StrategyValidator: real category/risk checks

Usage:
    .venv/bin/python demo_cpu.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentforge.orchestrator import Orchestrator
from agentforge.state import StateFile

PYTHON = sys.executable

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Demo project files
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRAIN_PY = r'''#!/usr/bin/env python3
"""
Gradient descent optimizer for f(x) = (x - 3)^2.
Goal: find x as close to 3.0 as possible.

Current hyperparameters are deliberately suboptimal.
An AI agent should improve LR, EPOCHS, and/or add momentum.
"""
import json
import os
import random
import sys
import time

random.seed(int(os.environ.get("PYTHONHASHSEED", "42")))

# ── Hyperparameters (THESE ARE SUBOPTIMAL — IMPROVE THEM) ──
LR = 0.001          # learning rate: too small, converges slowly
EPOCHS = 30         # iterations: too few to converge
MOMENTUM = 0.0      # no momentum
TARGET = 3.0

# Initialize from random starting point
x = random.uniform(-10, 10)
velocity = 0.0

print(f"[train] Start: x={x:.4f}, lr={LR}, epochs={EPOCHS}, momentum={MOMENTUM}")

for epoch in range(EPOCHS):
    loss = (x - TARGET) ** 2
    grad = 2.0 * (x - TARGET)
    velocity = MOMENTUM * velocity - LR * grad
    x += velocity
    if epoch % max(1, EPOCHS // 5) == 0 or epoch == EPOCHS - 1:
        print(f"[train] Epoch {epoch+1:4d}/{EPOCHS} | loss={loss:8.4f} | x={x:.6f}")
    time.sleep(0.005)

final_loss = (x - TARGET) ** 2
accuracy = 1.0 / (1.0 + abs(x - TARGET))

print(f"[train] Done: x={x:.6f}, loss={final_loss:.6f}, accuracy={accuracy:.6f}")

os.makedirs("model", exist_ok=True)
with open("model/state.json", "w") as f:
    json.dump({
        "x": x, "loss": final_loss, "accuracy": accuracy,
        "lr": LR, "epochs": EPOCHS, "momentum": MOMENTUM,
    }, f, indent=2)
'''

BENCHMARK_PY = r'''#!/usr/bin/env python3
"""Evaluate trained model. Writes results/benchmark.json with accuracy metric."""
import json
import os
import sys

if not os.path.exists("model/state.json"):
    print("[benchmark] ERROR: model/state.json not found")
    sys.exit(1)

with open("model/state.json") as f:
    state = json.load(f)

x = state["x"]
accuracy = 1.0 / (1.0 + abs(x - 3.0))

os.makedirs("results", exist_ok=True)
with open("results/benchmark.json", "w") as f:
    json.dump({"accuracy": round(accuracy, 6)}, f, indent=2)

print(f"[benchmark] accuracy={accuracy:.6f} (x={x:.6f}, optimal=3.0)")
'''

TEST_SUITE_PY = r'''#!/usr/bin/env python3
"""Basic sanity tests. Must pass for any valid solution."""
import sys

def test_imports():
    import json, os, random, time
    return True

def test_math():
    assert abs((2.0 - 3.0)**2 - 1.0) < 1e-10
    return True

def test_accuracy_formula():
    acc = 1.0 / (1.0 + abs(3.0 - 3.0))
    assert acc == 1.0
    return True

tests = [test_imports, test_math, test_accuracy_formula]
for t in tests:
    try:
        assert t()
    except Exception as e:
        print(f"FAIL: {t.__name__}: {e}")
        sys.exit(1)
print(f"All {len(tests)} tests passed")
'''


def create_project(base_dir: Path) -> tuple[Path, Path]:
    """Create a minimal ML project as a real git repo."""
    project = base_dir / "project"
    project.mkdir()

    # Write project files
    (project / "train.py").write_text(TRAIN_PY)
    (project / "benchmark.py").write_text(BENCHMARK_PY)
    (project / "test_suite.py").write_text(TEST_SUITE_PY)
    (project / "requirements.txt").write_text("")

    # Initialize git repo
    cmds = [
        ["git", "init", "-b", "main"],
        ["git", "config", "user.email", "demo@agentforge.dev"],
        ["git", "config", "user.name", "AgentForge Demo"],
        ["git", "add", "."],
        ["git", "commit", "-m", "init: baseline with suboptimal hyperparams"],
    ]
    for cmd in cmds:
        subprocess.run(cmd, cwd=str(project), check=True, capture_output=True)

    # Write challenge config
    config_path = base_dir / "challenge.yaml"
    config_path.write_text(
        f"challenge:\n"
        f"  name: 'CPU Demo - Gradient Descent Optimization'\n"
        f"  description: |\n"
        f"    Optimize f(x) = (x-3)^2 using gradient descent.\n"
        f"    The baseline in train.py uses lr=0.001 and only 30 epochs, which barely converges.\n"
        f"    Improve the hyperparameters (LR, EPOCHS, MOMENTUM) in train.py to achieve\n"
        f"    accuracy >= 0.90, where accuracy = 1/(1+|x-3|). Perfect score = 1.0 when x=3.\n"
        f"    You can modify train.py to change LR, EPOCHS, MOMENTUM, or add new techniques.\n"
        f"target:\n"
        f"  metric: accuracy\n"
        f"  value: 0.90\n"
        f"  direction: maximize\n"
        f"tests:\n"
        f"  smoke: '{PYTHON} test_suite.py'\n"
        f"  full: '{PYTHON} test_suite.py'\n"
        f"  benchmark: '{PYTHON} benchmark.py'\n"
        f"constraints:\n"
        f"  writable:\n"
        f"    - train.py\n"
        f"  read_only:\n"
        f"    - test_suite.py\n"
        f"    - benchmark.py\n"
    )

    return project, config_path


def show_baseline(project: Path):
    """Run the baseline to show its poor performance."""
    print("--- Baseline Performance ---")
    subprocess.run([PYTHON, "train.py"], cwd=str(project))
    subprocess.run([PYTHON, "benchmark.py"], cwd=str(project))

    results_file = project / "results" / "benchmark.json"
    if results_file.exists():
        with open(results_file) as f:
            baseline = json.load(f)
        print(f"Baseline accuracy: {baseline['accuracy']:.6f}")
        print(f"Target accuracy:   0.90")
        print(f"Gap:               {0.90 - baseline['accuracy']:.6f}")

    # Clean up baseline artifacts so Codex starts fresh
    shutil.rmtree(project / "model", ignore_errors=True)
    shutil.rmtree(project / "results", ignore_errors=True)


def show_results(project: Path, config_target: float, elapsed: float):
    """Load and display final results."""
    sf = StateFile(project / ".agentforge" / "state.json")
    if not sf.exists():
        print("ERROR: No state file found. Run may have failed.")
        return

    state = sf.load()

    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Status:           {state.status}")
    print(f"  Best score:       {state.best.score:.6f}")
    print(f"  Target:           {config_target}")
    hit = state.best.score >= config_target
    print(f"  Target reached:   {'YES' if hit else 'NO'}")
    print(f"  Rounds completed: {state.budget.rounds_used}")
    print(f"  Time elapsed:     {elapsed:.1f}s ({elapsed/60:.1f}m)")

    if state.rounds:
        r = state.rounds[-1]
        print(f"\n  Experiments (round {r.round}):")
        for e in r.experiments:
            status_icon = "OK" if e.status == "ok" else "FAIL"
            print(f"    [{status_icon}] {e.id} ({e.strategy}): "
                  f"score={e.score:.6f}, status={e.status}")
            if e.error:
                print(f"         error: {e.error}")
        print(f"    Winners: {r.winners}")
        print(f"    Phase 1 (Agent):    {r.phase1_minutes:.1f} min")
        print(f"    Phase 2 (Training): {r.phase2_minutes:.1f} min")

    # Show strategies tried
    if state.strategies_tried:
        print(f"\n  Strategies tried:")
        for s in state.strategies_tried:
            print(f"    - {s.name}: score={s.score:.4f} ({s.outcome})")

    # Show training logs
    log_dir = project / ".agentforge" / "runs" / "logs"
    if log_dir.exists():
        print(f"\n  Training Logs:")
        for log in sorted(log_dir.glob("*.log")):
            content = log.read_text().strip()
            lines = content.split("\n")
            print(f"\n    --- {log.name} ({len(lines)} lines) ---")
            # Show last 8 lines
            for line in lines[-8:]:
                print(f"    {line}")

    print()
    print("=" * 60)
    if hit:
        print("  SUCCESS: Target accuracy reached!")
    else:
        print(f"  PAUSED: Best {state.best.score:.4f} < target {config_target}")
    print("=" * 60)

    # Show state file path
    print(f"\n  State file: {sf.path}")


def main():
    print()
    print("=" * 60)
    print("  AgentForge v5.0 — Full Real CPU Demo")
    print("  ALL components real, including Codex CLI")
    print("=" * 60)
    print()

    # Create temp working directory
    base_dir = Path(tempfile.mkdtemp(prefix="agentforge-demo-"))
    print(f"Working directory: {base_dir}")

    try:
        # 1. Setup project
        project, config_path = create_project(base_dir)
        print(f"Project:           {project}")
        print(f"Config:            {config_path}")

        # Show git branches
        result = subprocess.run(
            ["git", "branch", "-a"], cwd=str(project),
            capture_output=True, text=True,
        )
        print(f"Git branches:      {result.stdout.strip()}")
        print()

        # 2. Show baseline performance
        show_baseline(project)
        print()

        # 3. Run AgentForge
        print("--- Starting AgentForge ---")
        orch = Orchestrator(config_path, project)

        # Initialize state
        state = orch._init_or_resume()
        state.budget.rounds_max = 1  # 1 round for demo
        orch.state_file.save(state)

        print(f"Hardware:  {state.hardware.device}, "
              f"{state.hardware.cpu_cores} cores, "
              f"{state.hardware.ram_gb}GB RAM, "
              f"{state.hardware.disk_free_gb}GB disk")
        print(f"N:         {state.N} (parallel experiments)")
        print(f"Target:    accuracy >= {orch.config.target_value}")
        print()

        print("Phase 1: Codex CLI developing strategies...")
        print("  (Codex will read train.py, create improved versions,")
        print("   commit to git branches, run trials to verify)")
        print()

        t0 = time.time()
        orch.run()
        elapsed = time.time() - t0

        # 4. Show results
        show_results(project, orch.config.target_value, elapsed)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

        # Show any partial state
        sf = StateFile(project / ".agentforge" / "state.json")
        if sf.exists():
            state = sf.load()
            print(f"\nPartial state: round={state.current_round}, "
                  f"status={state.status}")

        # Show Codex output if available
        output_file = project / ".agentforge" / "agent_output.json"
        if output_file.exists():
            print(f"\nAgent output file exists: {output_file}")
            print(output_file.read_text()[:500])
    finally:
        # Ask before cleanup
        print(f"\nDemo directory: {base_dir}")
        try:
            answer = input("Clean up demo directory? [Y/n] ").strip().lower()
            if answer in ("", "y", "yes"):
                shutil.rmtree(base_dir, ignore_errors=True)
                print("Cleaned up.")
            else:
                print(f"Kept at: {base_dir}")
        except (EOFError, KeyboardInterrupt):
            print(f"\nKept at: {base_dir}")


if __name__ == "__main__":
    main()
