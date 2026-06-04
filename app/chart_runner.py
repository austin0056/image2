"""在隔离子进程里执行 LLM 生成的 matplotlib 代码，产出 PNG。

约定：LLM 代码用 matplotlib（plt）/numpy（np）把图画在当前 figure 上；
不要 show/savefig。harness 负责把首个 figure 存成 out.png。

安全说明同 cad_runner：子进程 + 超时 + 清空环境变量 + 临时目录 + 危险调用静态拦截，
是务实隔离而非完美沙箱。
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile

from .cad_runner import _FORBIDDEN_RE, _safe_env  # 复用同一套拦截规则与白名单环境
from .config import settings

log = logging.getLogger("image2.chart_runner")


class ChartRunError(RuntimeError):
    pass


_MAX_PNG_BYTES = 20 * 1024 * 1024

_HARNESS = '''\
import sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.autolayout"] = True

# === LLM CODE BEGIN ===
{code}
# === LLM CODE END ===

_nums = plt.get_fignums()
if not _nums:
    print("ERROR: 代码没有创建任何 matplotlib 图形", file=sys.stderr)
    sys.exit(3)
plt.figure(_nums[0]).savefig("out.png", dpi=160, bbox_inches="tight")
print("OK")
'''


def _guard(code: str) -> None:
    for rx in _FORBIDDEN_RE:
        m = rx.search(code)
        if m:
            raise ChartRunError(f"代码包含被禁止的调用：{m.group(0)!r}")


def _run_sync(code: str) -> bytes:
    _guard(code)
    timeout = max(5, settings.chart_exec_timeout)
    tmp = tempfile.mkdtemp(prefix="chartgen_")
    try:
        harness_path = os.path.join(tmp, "harness.py")
        with open(harness_path, "w", encoding="utf-8") as f:
            f.write(_HARNESS.format(code=code))

        env = _safe_env()
        env["MPLCONFIGDIR"] = tmp  # matplotlib 字体缓存/配置目录（环境被精简，需指定可写目录）
        env["MPLBACKEND"] = "Agg"
        popen_kw: dict = dict(cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout)
        try:
            proc = subprocess.run([sys.executable, harness_path], **popen_kw)
        except subprocess.TimeoutExpired:
            raise ChartRunError(f"执行超时（> {timeout}s）")

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "未知错误").strip()
            raise ChartRunError(err[-1500:])

        out = os.path.join(tmp, "out.png")
        if not os.path.exists(out):
            raise ChartRunError("未生成 out.png")
        size = os.path.getsize(out)
        if size == 0:
            raise ChartRunError("out.png 为空")
        if size > _MAX_PNG_BYTES:
            raise ChartRunError(f"图片过大（{size} 字节）")
        with open(out, "rb") as fp:
            return fp.read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def run_matplotlib_script(code: str) -> bytes:
    """异步入口：线程里跑阻塞子进程，返回 PNG 字节。"""
    return await asyncio.to_thread(_run_sync, code)
