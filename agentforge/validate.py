from __future__ import annotations

from pathlib import Path

from agentforge.config import ChallengeConfig, load_config


def validate_challenge(workdir: Path) -> list[str]:
    """校验 challenge 目录的完整性，返回错误列表（空 = 通过）。"""
    errors: list[str] = []

    # 1. challenge.yaml 存在且可解析
    config_path = workdir / "challenge.yaml"
    if not config_path.exists():
        errors.append("challenge.yaml 不存在")
        return errors

    try:
        config = load_config(config_path)
    except (ValueError, FileNotFoundError) as e:
        errors.append(f"challenge.yaml 解析失败: {e}")
        return errors

    # 2. 必要字段非空
    if not config.challenge_name.strip():
        errors.append("challenge.name 为空")
    if not config.target_metric.strip():
        errors.append("target.metric 为空")
    if config.target_direction not in ("minimize", "maximize"):
        errors.append(f"target.direction 无效: {config.target_direction!r}（应为 minimize 或 maximize）")

    # 3. benchmark.py 存在
    benchmark = workdir / "benchmark.py"
    if not benchmark.exists():
        errors.append("benchmark.py 不存在")

    # 4. test_suite.py 存在
    test_suite = workdir / "test_suite.py"
    if not test_suite.exists():
        errors.append("test_suite.py 不存在")

    # 5. writable 文件至少有一个存在
    writable_exists = any((workdir / w).exists() for w in config.writable)
    if not writable_exists:
        errors.append("writable 列表中没有任何文件存在")

    # 6. benchmark.py 输出格式检查（静态：检查是否写 benchmark.json）
    if benchmark.exists():
        content = benchmark.read_text()
        if "benchmark.json" not in content:
            errors.append("benchmark.py 中未发现 'benchmark.json' 输出路径")
        if config.target_metric not in content:
            errors.append(f"benchmark.py 中未发现 metric key '{config.target_metric}'")

    # 7. git repo 检查
    git_dir = workdir / ".git"
    if not git_dir.exists():
        errors.append("不是 git 仓库（缺少 .git）")

    return errors


def run_validate(workdir: Path) -> bool:
    """CLI 入口：运行校验并打印结果。返回 True 表示通过。"""
    errors = validate_challenge(workdir)
    if not errors:
        print("[validate] 全部通过 ✓")
        return True
    print(f"[validate] 发现 {len(errors)} 个问题:")
    for i, e in enumerate(errors, 1):
        print(f"  {i}. {e}")
    return False
