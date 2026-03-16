#!/usr/bin/env python3
"""
AgentForge v5.0 — nanoGPT Real ML Demo

Runs a complete AgentForge optimization session on nanoGPT shakespeare_char.
ALL components run for real including Codex CLI.

Flow:
  1. Clone nanoGPT, prepare shakespeare_char dataset
  2. Run baseline training (poor hyperparams → high val_loss)
  3. AgentForge Phase 1: Codex CLI analyzes code, develops N improved strategies
  4. AgentForge Phase 2: Real parallel training of all strategies
  5. Scorer: evaluates val_loss for each strategy
  6. Report results

Usage:
    python3 demo_nanogpt.py
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

# Use system python3 which has torch
PYTHON = "/opt/homebrew/bin/python3"

# Ensure agentforge is importable
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# benchmark.py — evaluates the trained model's val_loss
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BENCHMARK_PY = r'''#!/usr/bin/env python3
"""
Evaluate a trained nanoGPT checkpoint.
Loads best checkpoint, computes val_loss, writes results/benchmark.json.

val_loss is cross-entropy; lower is better.
We convert to a 0-1 "accuracy" score: accuracy = max(0, 1 - val_loss / 3.0)
  - val_loss=3.0 (random) → accuracy=0.0
  - val_loss=1.5 → accuracy=0.5
  - val_loss=0.0 (perfect) → accuracy=1.0
"""
import json
import os
import sys
import glob

# Find the best checkpoint
ckpt_dirs = glob.glob("out-shakespeare-char*")
if not ckpt_dirs:
    ckpt_dirs = glob.glob("out*")
if not ckpt_dirs:
    print("[benchmark] ERROR: no output directory found")
    sys.exit(1)

best_val_loss = float("inf")
best_dir = None

for d in ckpt_dirs:
    ckpt_path = os.path.join(d, "ckpt.pt")
    if os.path.exists(ckpt_path):
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        vl = float(ckpt.get("best_val_loss", float("inf")))
        print(f"[benchmark] {d}: val_loss={vl:.4f}")
        if vl < best_val_loss:
            best_val_loss = vl
            best_dir = d

if best_dir is None:
    # Fallback: parse training log
    for d in ckpt_dirs:
        log_path = os.path.join(d, "train.log")
        if os.path.exists(log_path):
            with open(log_path) as f:
                for line in f:
                    if "val loss" in line.lower():
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if "val" in p.lower() and i + 2 < len(parts):
                                try:
                                    vl = float(parts[i + 2].rstrip(","))
                                    if vl < best_val_loss:
                                        best_val_loss = vl
                                        best_dir = d
                                except ValueError:
                                    pass

if best_val_loss == float("inf"):
    print("[benchmark] ERROR: could not find val_loss in any checkpoint")
    sys.exit(1)

# Convert val_loss to accuracy score (0-1, higher is better)
accuracy = max(0.0, 1.0 - best_val_loss / 3.0)

print(f"[benchmark] Best: {best_dir}, val_loss={best_val_loss:.4f}, accuracy={accuracy:.4f}")

os.makedirs("results", exist_ok=True)
with open("results/benchmark.json", "w") as f:
    json.dump({"accuracy": round(accuracy, 6), "val_loss": round(best_val_loss, 6)}, f, indent=2)
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# test_suite.py — basic sanity tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEST_SUITE_PY = r'''#!/usr/bin/env python3
"""Sanity tests for nanoGPT project."""
import sys
import os

def test_torch():
    import torch
    assert torch.__version__
    return True

def test_data_exists():
    assert os.path.exists("data/shakespeare_char/train.bin"), "train.bin missing"
    assert os.path.exists("data/shakespeare_char/val.bin"), "val.bin missing"
    return True

def test_model_importable():
    from model import GPT, GPTConfig
    cfg = GPTConfig(vocab_size=65, block_size=64, n_layer=2, n_head=2, n_embd=64)
    m = GPT(cfg)
    params = sum(p.numel() for p in m.parameters())
    assert params > 0
    return True

tests = [test_torch, test_data_exists, test_model_importable]
for t in tests:
    try:
        assert t()
    except Exception as e:
        print(f"FAIL: {t.__name__}: {e}")
        sys.exit(1)
print(f"All {len(tests)} tests passed")
'''


def setup_nanogpt(base_dir: Path) -> tuple[Path, Path]:
    """Clone nanoGPT, prepare data, create challenge config."""
    project = base_dir / "nanogpt"

    print("[Setup] Cloning nanoGPT...")
    subprocess.run(
        ["git", "clone", "--depth=1", "https://github.com/karpathy/nanoGPT.git",
         str(project)],
        check=True, capture_output=True, timeout=60,
    )

    # Configure git for AgentForge branch operations
    subprocess.run(["git", "config", "user.email", "demo@agentforge.dev"],
                   cwd=str(project), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "AgentForge Demo"],
                   cwd=str(project), check=True, capture_output=True)

    print("[Setup] Preparing shakespeare_char dataset...")
    subprocess.run(
        [PYTHON, "data/shakespeare_char/prepare.py"],
        cwd=str(project), check=True, timeout=60,
    )

    # Verify data files exist
    assert (project / "data" / "shakespeare_char" / "train.bin").exists()
    assert (project / "data" / "shakespeare_char" / "val.bin").exists()
    print("[Setup] Dataset ready (1.1M chars, 65-char vocab)")

    # Add benchmark.py and test_suite.py
    (project / "benchmark.py").write_text(BENCHMARK_PY)
    (project / "test_suite.py").write_text(TEST_SUITE_PY)
    subprocess.run(["git", "add", "benchmark.py", "test_suite.py"],
                   cwd=str(project), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add benchmark and test suite"],
                   cwd=str(project), check=True, capture_output=True)

    # Create challenge config
    config_path = base_dir / "challenge.yaml"
    config_path.write_text(
        f"challenge:\n"
        f"  name: 'nanoGPT Shakespeare char-level optimization'\n"
        f"  description: |\n"
        f"    Train a character-level GPT on Shakespeare text using nanoGPT.\n"
        f"    The baseline config uses suboptimal hyperparameters:\n"
        f"    n_layer=2, n_head=2, n_embd=64, batch_size=8, max_iters=500, lr=5e-4.\n"
        f"    This gives val_loss ~1.85 (accuracy ~0.38).\n"
        f"    \n"
        f"    Your goal: modify the training config/code to minimize val_loss.\n"
        f"    Lower val_loss = higher accuracy. Target: accuracy >= 0.50 (val_loss <= 1.50).\n"
        f"    \n"
        f"    accuracy = max(0, 1 - val_loss/3.0)\n"
        f"    Key files: train.py, model.py, config/train_shakespeare_char.py\n"
        f"    Training command: {PYTHON} train.py <your_config_overrides>\n"
        f"    IMPORTANT: Must use --device=cpu --compile=False\n"
        f"    \n"
        f"    TRIAL RUNS: For verification trials, use max_iters=50 --eval_interval=25\n"
        f"    --eval_iters=5 (takes ~15s on CPU). Do NOT run long trials.\n"
        f"    FULL TRAINING: Use max_iters=1000-2000 in the train_command field.\n"
        f"    Keep total trial time under 2 minutes per strategy.\n"
        f"target:\n"
        f"  metric: accuracy\n"
        f"  value: 0.50\n"
        f"  direction: maximize\n"
        f"tests:\n"
        f"  smoke: '{PYTHON} test_suite.py'\n"
        f"  full: '{PYTHON} test_suite.py'\n"
        f"  benchmark: '{PYTHON} benchmark.py'\n"
        f"constraints:\n"
        f"  writable:\n"
        f"    - train.py\n"
        f"    - model.py\n"
        f"    - config\n"
        f"  read_only:\n"
        f"    - data\n"
        f"    - test_suite.py\n"
        f"    - benchmark.py\n"
    )

    return project, config_path


def run_baseline(project: Path):
    """Run baseline nanoGPT training with poor hyperparameters."""
    print("\n--- Baseline Training (deliberately suboptimal) ---")
    print("  Config: n_layer=2, n_head=2, n_embd=64, batch_size=8, max_iters=500")
    print("  (CPU training, ~6 seconds)")

    baseline_cmd = [
        PYTHON, "train.py",
        "config/train_shakespeare_char.py",
        "--device=cpu",
        "--compile=False",
        "--n_layer=2",
        "--n_head=2",
        "--n_embd=64",
        "--batch_size=8",
        "--block_size=64",
        "--max_iters=500",
        "--lr_decay_iters=500",
        "--learning_rate=5e-4",
        "--eval_interval=100",
        "--eval_iters=20",
        "--log_interval=50",
        "--out_dir=out-shakespeare-char",
    ]

    t0 = time.time()
    result = subprocess.run(
        baseline_cmd, cwd=str(project),
        capture_output=True, text=True, timeout=300,
    )
    elapsed = time.time() - t0

    # Show training output
    for line in (result.stdout or "").split("\n"):
        if "val loss" in line.lower() or "step" in line.lower() or "iter" in line.lower():
            print(f"  {line.strip()}")

    # Run benchmark
    bm_result = subprocess.run(
        [PYTHON, "benchmark.py"], cwd=str(project),
        capture_output=True, text=True, timeout=30,
    )
    print(bm_result.stdout.strip() if bm_result.stdout else "")

    results_file = project / "results" / "benchmark.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        print(f"\n  Baseline: val_loss={data.get('val_loss', '?')}, "
              f"accuracy={data.get('accuracy', '?')}")
        print(f"  Training time: {elapsed:.0f}s")
        print(f"  Target: accuracy >= 0.50 (val_loss <= 1.50)")
    else:
        print(f"  WARNING: benchmark.py failed: {bm_result.stderr[:200] if bm_result.stderr else 'no output'}")

    # Clean baseline artifacts but keep data
    shutil.rmtree(project / "out-shakespeare-char", ignore_errors=True)
    shutil.rmtree(project / "results", ignore_errors=True)
    # Reset any modified files
    subprocess.run(["git", "checkout", "."], cwd=str(project), capture_output=True)


def show_results(project: Path, config_target: float, elapsed: float):
    """Load and display final AgentForge results."""
    from agentforge.state import StateFile

    sf = StateFile(project / ".agentforge" / "state.json")
    if not sf.exists():
        print("ERROR: No state file found.")
        return

    state = sf.load()

    print()
    print("=" * 64)
    print("  RESULTS")
    print("=" * 64)
    print(f"  Status:           {state.status}")
    print(f"  Best score:       {state.best.score:.6f}")
    print(f"  Target:           {config_target}")
    hit = state.best.score >= config_target
    print(f"  Target reached:   {'YES' if hit else 'NO'}")
    print(f"  Rounds completed: {state.budget.rounds_used}")
    print(f"  Time elapsed:     {elapsed:.0f}s ({elapsed/60:.1f}m)")

    if state.rounds:
        r = state.rounds[-1]
        print(f"\n  Experiments (round {r.round}):")
        for e in r.experiments:
            icon = "OK" if e.status == "ok" else "FAIL"
            print(f"    [{icon}] {e.id} ({e.strategy}): "
                  f"score={e.score:.4f}, status={e.status}")
            if e.error:
                print(f"         error: {e.error}")
        print(f"    Winners: {r.winners}")
        print(f"    Phase 1 (Codex):    {r.phase1_minutes:.1f}m")
        print(f"    Phase 2 (Training): {r.phase2_minutes:.1f}m")

    if state.strategies_tried:
        print(f"\n  Strategies:")
        for s in state.strategies_tried:
            print(f"    - {s.name}: score={s.score:.4f} ({s.outcome})")

    # Show training logs
    log_dir = project / ".agentforge" / "runs" / "logs"
    if log_dir.exists():
        for log in sorted(log_dir.glob("*.log")):
            content = log.read_text().strip()
            lines = content.split("\n")
            # Find val loss lines
            val_lines = [l for l in lines if "val loss" in l.lower() or "val_loss" in l.lower()]
            other_lines = lines[-3:] if not val_lines else []
            print(f"\n  Log {log.name} ({len(lines)} lines):")
            for line in (val_lines[-5:] + other_lines):
                print(f"    {line.strip()}")

    print()
    print("=" * 64)
    if hit:
        print("  SUCCESS: nanoGPT optimization target reached!")
    else:
        print(f"  PAUSED: best {state.best.score:.4f} < target {config_target}")
    print("=" * 64)


def main():
    print()
    print("=" * 64)
    print("  AgentForge v5.0 — nanoGPT Shakespeare Demo")
    print("  Real ML training, real Codex CLI, all components live")
    print("=" * 64)
    print()

    base_dir = Path(tempfile.mkdtemp(prefix="agentforge-nanogpt-"))
    print(f"Working directory: {base_dir}")

    try:
        # 1. Setup
        project, config_path = setup_nanogpt(base_dir)

        # 2. Baseline
        run_baseline(project)

        # 3. AgentForge
        print("\n--- Starting AgentForge ---")

        from agentforge.orchestrator import Orchestrator
        orch = Orchestrator(config_path, project)

        state = orch._init_or_resume()
        state.budget.rounds_max = 1
        orch.state_file.save(state)

        print(f"Hardware: {state.hardware.device}, {state.hardware.cpu_cores} cores")
        print(f"N:        {state.N} (parallel experiments)")
        print(f"Target:   accuracy >= {orch.config.target_value}")
        print()
        print("Phase 1: Codex CLI analyzing nanoGPT, developing strategies...")
        print("Phase 2: Real parallel nanoGPT training on CPU...")
        print()

        t0 = time.time()
        orch.run()
        elapsed = time.time() - t0

        # 4. Results
        show_results(project, orch.config.target_value, elapsed)

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\nDemo directory: {base_dir}")
        try:
            answer = input("Clean up? [Y/n] ").strip().lower()
            if answer in ("", "y", "yes"):
                shutil.rmtree(base_dir, ignore_errors=True)
                print("Cleaned up.")
            else:
                print(f"Kept at: {base_dir}")
        except (EOFError, KeyboardInterrupt):
            print(f"\nKept at: {base_dir}")


if __name__ == "__main__":
    main()
