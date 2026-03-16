#!/usr/bin/env python3
"""
AgentForge v5.0 — Auto-Setup Demo

完整流程：
  Phase 0: Codex 分析 nanoGPT repo → 自动生成 challenge.yaml/benchmark.py/test_suite.py
  Phase 1: Codex 开发优化策略
  Phase 2: 并行训练 + 评分

与 demo_nanogpt.py 的区别：
  - 不手动写 challenge.yaml/benchmark.py/test_suite.py
  - 让 auto-setup 自动分析项目并生成这些文件
  - 用户确认后才开始优化

Usage:
    python3 demo_auto_setup.py
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

PYTHON = "/opt/homebrew/bin/python3"
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def setup_nanogpt(base_dir: Path) -> Path:
    """Clone nanoGPT, prepare data. 不创建任何配置文件。"""
    project = base_dir / "nanogpt"

    print("[Setup] Cloning nanoGPT...")
    subprocess.run(
        ["git", "clone", "--depth=1", "https://github.com/karpathy/nanoGPT.git",
         str(project)],
        check=True, capture_output=True, timeout=60,
    )

    # Configure git
    subprocess.run(["git", "config", "user.email", "demo@agentforge.dev"],
                   cwd=str(project), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "AgentForge Demo"],
                   cwd=str(project), check=True, capture_output=True)

    print("[Setup] Preparing shakespeare_char dataset...")
    subprocess.run(
        [PYTHON, "data/shakespeare_char/prepare.py"],
        cwd=str(project), check=True, timeout=60,
    )

    assert (project / "data" / "shakespeare_char" / "train.bin").exists()
    assert (project / "data" / "shakespeare_char" / "val.bin").exists()
    print("[Setup] Dataset ready")

    return project


def run_baseline(project: Path) -> dict:
    """Run baseline training, return benchmark results."""
    print("\n--- Baseline Training (deliberately suboptimal) ---")

    baseline_cmd = [
        PYTHON, "train.py",
        "config/train_shakespeare_char.py",
        "--device=cpu", "--compile=False",
        "--n_layer=2", "--n_head=2", "--n_embd=64",
        "--batch_size=8", "--block_size=64",
        "--max_iters=500", "--lr_decay_iters=500",
        "--learning_rate=5e-4",
        "--eval_interval=100", "--eval_iters=20",
        "--log_interval=50",
        "--out_dir=out-shakespeare-char",
    ]

    t0 = time.time()
    result = subprocess.run(
        baseline_cmd, cwd=str(project),
        capture_output=True, text=True, timeout=300,
    )
    elapsed = time.time() - t0

    for line in (result.stdout or "").split("\n"):
        if "val loss" in line.lower() or "iter" in line.lower():
            print(f"  {line.strip()}")

    print(f"  Baseline training: {elapsed:.0f}s")

    # Clean baseline artifacts
    shutil.rmtree(project / "out-shakespeare-char", ignore_errors=True)
    subprocess.run(["git", "checkout", "."], cwd=str(project), capture_output=True)
    return {}


def show_results(project: Path, elapsed: float):
    """Display final results."""
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

    print()
    print("=" * 64)

    # Show generated files
    print("\n  Auto-generated files:")
    for fname in ["challenge.yaml", "benchmark.py", "test_suite.py"]:
        path = project / fname
        if path.exists():
            lines = path.read_text().strip().split("\n")
            print(f"\n  --- {fname} ({len(lines)} lines) ---")
            for line in lines[:8]:
                print(f"    {line}")
            if len(lines) > 8:
                print(f"    ... ({len(lines) - 8} more lines)")


def main():
    print()
    print("=" * 64)
    print("  AgentForge v5.0 — Auto-Setup Demo")
    print("  Phase 0: Codex 自动分析 repo → 生成配置")
    print("  Phase 1: Codex 开发优化策略")
    print("  Phase 2: 并行训练 + 评分")
    print("=" * 64)
    print()

    base_dir = Path(tempfile.mkdtemp(prefix="agentforge-auto-"))
    print(f"Working directory: {base_dir}")

    try:
        # 1. Setup nanoGPT (只 clone + 准备数据，不创建任何配置)
        project = setup_nanogpt(base_dir)

        # 2. Baseline (可选，展示初始性能)
        run_baseline(project)

        # 3. AgentForge — 从 auto-setup 开始
        print("\n" + "=" * 64)
        print("  Phase 0: Auto-Setup (Codex 分析项目)")
        print("=" * 64)
        print()
        print("  项目中没有 challenge.yaml，AgentForge 将自动:")
        print("  1. 调用 Codex 分析项目结构、入口、指标")
        print("  2. 生成 challenge.yaml + benchmark.py + test_suite.py")
        print("  3. 让你确认每个文件")
        print()

        from agentforge.orchestrator import Orchestrator

        # config_path=None → 触发 auto-setup
        t0 = time.time()
        orch = Orchestrator(config_path=None, workdir=project)

        # Verify auto-setup generated files
        for fname in ["challenge.yaml", "benchmark.py", "test_suite.py"]:
            if (project / fname).exists():
                print(f"  [OK] {fname} generated")
            else:
                print(f"  [SKIP] {fname} not generated")

        print(f"\n  Config loaded: target={orch.config.target_metric} "
              f"{orch.config.target_direction} {orch.config.target_value}")

        # 4. Run optimization (1 round)
        print("\n" + "=" * 64)
        print("  Phase 1 + 2: Optimization")
        print("=" * 64)

        state = orch._init_or_resume()
        state.budget.rounds_max = 1
        orch.state_file.save(state)

        print(f"  Hardware: {state.hardware.device}, {state.hardware.cpu_cores} cores")
        print(f"  N:        {state.N} (parallel experiments)")
        print()

        orch.run()
        elapsed = time.time() - t0

        # 5. Results
        show_results(project, elapsed)

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
