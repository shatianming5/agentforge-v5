from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path


def prepare_data(workdir: Path, data_config: dict | None) -> None:
    """优化循环前调用一次。检查、下载、校验、锁定数据。"""
    if not data_config:
        return

    data_path = workdir / data_config.get("path", "data")

    # 1. 检查必要环境变量
    for env_var in data_config.get("requires_env", []):
        if not os.environ.get(env_var):
            raise RuntimeError(f"缺少环境变量: {env_var}")

    # 2. 检查磁盘空间
    expected_mb = data_config.get("expected_size_mb", 0)
    if expected_mb > 0:
        free = _disk_free_mb(workdir)
        if free < expected_mb * 2.5:
            raise RuntimeError(
                f"磁盘不足: 需要 {expected_mb * 2.5:.0f}MB, 可用 {free:.0f}MB"
            )

    # 3. 数据已存在则跳过下载
    if data_path.exists() and any(data_path.iterdir()):
        print(f"[data] 数据已存在: {data_path}")
        _lock_readonly(data_path)
        return

    # 4. 下载数据
    source = data_config.get("source", "auto")
    if source == "script":
        cmd = data_config["command"]
        print(f"[data] 下载数据: {cmd}")
        r = subprocess.run(cmd, shell=True, cwd=str(workdir), timeout=7200)
        if r.returncode != 0:
            raise RuntimeError(f"数据下载失败: exit {r.returncode}")
    elif source == "auto":
        _auto_download(workdir, data_path)

    # 5. 校验 checksum
    checksum = data_config.get("checksum")
    if checksum and data_path.exists():
        actual = _dir_checksum(data_path)
        if actual != checksum:
            raise RuntimeError(
                f"数据校验失败: 期望 {checksum[:24]}..., 实际 {actual[:24]}..."
            )

    if data_path.exists():
        _lock_readonly(data_path)
        size = _dir_size_mb(data_path)
        print(f"[data] 数据就绪: {data_path} ({size:.0f}MB, 已锁定只读)")


def _auto_download(workdir: Path, data_path: Path) -> None:
    """自动检测并运行数据准备脚本。"""
    candidates = [
        data_path / "prepare.py",
        data_path / "download.py",
        workdir / "data" / "prepare.py",
        workdir / "data" / "download.py",
    ]
    for script in candidates:
        if script.exists():
            print(f"[data] 运行数据脚本: {script}")
            subprocess.run(
                ["python3", str(script)],
                cwd=str(workdir), timeout=7200, check=True,
            )
            return


def _lock_readonly(path: Path) -> None:
    subprocess.run(["chmod", "-R", "a-w", str(path)], check=False)


def _disk_free_mb(path: Path) -> float:
    st = os.statvfs(str(path))
    return (st.f_bavail * st.f_frsize) / (1024 * 1024)


def _dir_size_mb(path: Path) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)


def _dir_checksum(path: Path) -> str:
    h = hashlib.sha256()
    for dirpath, _, filenames in os.walk(path):
        for f in sorted(filenames):
            with open(os.path.join(dirpath, f), "rb") as fh:
                while chunk := fh.read(65536):
                    h.update(chunk)
    return f"sha256:{h.hexdigest()}"
