"""在隔离子进程里执行 LLM 生成的 build123d 代码，导出 STEP/STL/GLB。

约定：LLM 生成的代码必须把最终模型赋值给全局变量 `result`
（build123d 的 Part/Solid/Compound/Shape）。

安全说明（务实基线，非完美沙箱）：
  - 代码在独立子进程里跑，cwd 为一次性临时目录，跑完即删。
  - 子进程环境变量只保留运行所需的白名单，DB/S3/Claude 等密钥一律不传入。
  - 硬超时 + （Linux 下）CPU/内存/文件大小 rlimit 限制。
  - 执行前对代码做轻量静态拦截，命中危险调用直接拒绝。
这些手段能显著降低风险，但**不能根除**任意代码执行的隐患。生产级硬化的
理想方案是独立容器 / gVisor 沙箱，列为后续 TODO，不在本期范围。
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile

from .config import settings

log = logging.getLogger("image2.cad_runner")


class CadRunError(RuntimeError):
    """build123d 代码执行/导出失败。message 里带截断后的 stderr 供重修。"""


# 执行前静态拦截：命中即拒绝（best-effort，非安全边界）。
_FORBIDDEN = [
    r"\bimport\s+os\b",
    r"\bimport\s+sys\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+socket\b",
    r"\bimport\s+shutil\b",
    r"\bimport\s+requests\b",
    r"\bimport\s+urllib\b",
    r"\bimport\s+pathlib\b",
    r"\bfrom\s+os\b",
    r"\bfrom\s+subprocess\b",
    r"\b__import__\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    r"\bopen\s*\(",
    r"\bos\.",
    r"\bsys\.",
    r"\bsubprocess\.",
    r"\bpopen\b",
    r"\bsystem\s*\(",
    r"\bgetattr\s*\(",
    r"\bsetattr\s*\(",
]
_FORBIDDEN_RE = [re.compile(p, re.IGNORECASE) for p in _FORBIDDEN]

# 临时目录写满 / 巨型导出保护：单个产物上限。
_MAX_ARTIFACT_BYTES = 60 * 1024 * 1024  # 60MB

_HARNESS = '''\
import sys
from build123d import *
from build123d import export_step, export_stl, export_gltf

# === LLM CODE BEGIN ===
{code}
# === LLM CODE END ===

try:
    _shape = result
except NameError:
    print("ERROR: 代码必须把最终模型赋值给名为 `result` 的变量", file=sys.stderr)
    sys.exit(3)

export_step(_shape, "out.step")
export_stl(_shape, "out.stl")
export_gltf(_shape, "out.glb", binary=True)
print("OK")
'''


def _guard(code: str) -> None:
    for rx in _FORBIDDEN_RE:
        m = rx.search(code)
        if m:
            raise CadRunError(f"代码包含被禁止的调用：{m.group(0)!r}")


def _safe_env() -> dict[str, str]:
    """只保留运行 Python + 加载原生库所需的白名单环境变量。"""
    keep = (
        "PATH", "PYTHONHOME", "PYTHONPATH",
        "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",  # Linux / macOS 原生库
        "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR",  # Windows / 临时目录
        "HOME", "USERPROFILE", "LANG", "LC_ALL",
    )
    return {k: os.environ[k] for k in keep if k in os.environ}


def _preexec_limits():  # pragma: no cover - 仅 POSIX
    """Linux/macOS 下给子进程加资源上限。"""
    import resource

    cpu = max(5, settings.cad_exec_timeout)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 2))
    # 地址空间 2GB；OCP 网格化可能较吃内存，留足余量
    mem = 2 * 1024 * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    except (ValueError, OSError):
        pass
    resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_ARTIFACT_BYTES, _MAX_ARTIFACT_BYTES))


def _run_sync(code: str) -> dict[str, bytes]:
    _guard(code)
    timeout = max(5, settings.cad_exec_timeout)
    tmp = tempfile.mkdtemp(prefix="cadgen_")
    try:
        harness_path = os.path.join(tmp, "harness.py")
        with open(harness_path, "w", encoding="utf-8") as f:
            f.write(_HARNESS.format(code=code))

        popen_kw: dict = dict(
            cwd=tmp,
            env=_safe_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if sys.platform != "win32":
            popen_kw["preexec_fn"] = _preexec_limits

        try:
            proc = subprocess.run([sys.executable, harness_path], **popen_kw)
        except subprocess.TimeoutExpired:
            raise CadRunError(f"执行超时（> {timeout}s）")

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "未知错误").strip()
            # 只带回最后部分，足够 LLM 定位又不至于过长
            raise CadRunError(err[-1500:])

        out: dict[str, bytes] = {}
        for fmt, name in (("step", "out.step"), ("stl", "out.stl"), ("glb", "out.glb")):
            path = os.path.join(tmp, name)
            if not os.path.exists(path):
                raise CadRunError(f"未生成 {name}")
            size = os.path.getsize(path)
            if size == 0:
                raise CadRunError(f"{name} 为空")
            if size > _MAX_ARTIFACT_BYTES:
                raise CadRunError(f"{name} 过大（{size} 字节）")
            with open(path, "rb") as fp:
                out[fmt] = fp.read()
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def run_build123d_script(code: str) -> dict[str, bytes]:
    """异步入口：在线程里跑阻塞子进程，返回 {"step":bytes,"stl":bytes,"glb":bytes}。"""
    return await asyncio.to_thread(_run_sync, code)
