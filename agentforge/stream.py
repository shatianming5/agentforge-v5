"""共享的流式 subprocess 执行工具。所有需要实时输出的 subprocess 调用都用这里的函数。"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def stream_run(
    cmd: list[str] | str,
    *,
    cwd: str | Path | None = None,
    env: dict | None = None,
    timeout: int = 3600,
    prefix: str = "",
    check: bool = False,
    shell: bool = False,
) -> subprocess.CompletedProcess[str]:
    """subprocess.run 的流式替代：实时逐行打印 stdout+stderr。

    返回与 subprocess.CompletedProcess 兼容的对象（stdout 为完整输出）。
    """
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=shell,
    )
    stdout_lines: list[str] = []
    deadline = time.monotonic() + timeout
    tag = f"  [{prefix}] " if prefix else "  "
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            stdout_lines.append(line)
            sys.stdout.write(f"{tag}{line}")
            sys.stdout.flush()
            if time.monotonic() > deadline:
                proc.kill()
                proc.wait()
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)
        proc.wait()
    except subprocess.TimeoutExpired:
        raise
    except Exception:
        proc.kill()
        proc.wait()
        raise

    stdout = "".join(stdout_lines)
    result = subprocess.CompletedProcess(
        args=cmd, returncode=proc.returncode, stdout=stdout, stderr="",
    )
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=stdout, stderr="",
        )
    return result
