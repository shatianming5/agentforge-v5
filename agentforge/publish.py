from __future__ import annotations

import subprocess
from pathlib import Path

from agentforge.validate import validate_challenge


def run_publish(workdir: Path) -> bool:
    """发布 challenge：validate → 确认 → 提示 push。返回 True 表示成功。"""
    # Step 1: 校验
    errors = validate_challenge(workdir)
    if errors:
        print("[publish] 校验未通过，无法发布:")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}")
        return False

    # Step 2: 展示摘要
    from agentforge.config import load_config
    config = load_config(workdir / "challenge.yaml")
    print("\n[publish] Challenge 摘要:")
    print(f"  名称:    {config.challenge_name}")
    print(f"  指标:    {config.target_metric} ({config.target_direction})")
    print(f"  目标值:  {config.target_value}")
    print(f"  可写文件: {', '.join(config.writable[:5])}")

    # Step 3: 检查 git 状态
    dirty = _git_has_uncommitted(workdir)
    if dirty:
        print("\n[publish] 有未提交的更改，请先 commit:")
        print(f"  {dirty}")
        return False

    # Step 4: 检查是否有 remote
    remote = _git_remote_url(workdir)
    if not remote:
        print("\n[publish] 未设置 git remote，请先添加:")
        print("  git remote add origin <url>")
        return False

    # Step 5: 提示 push
    branch = _git_current_branch(workdir)
    print(f"\n[publish] 校验通过。请执行以下命令发布:")
    print(f"  git push origin {branch}")
    return True


def _git_has_uncommitted(workdir: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(workdir), capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _git_remote_url(workdir: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "config", "remote.origin.url"],
            cwd=str(workdir), capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _git_current_branch(workdir: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(workdir), capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "main"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "main"
