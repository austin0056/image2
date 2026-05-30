"""文字转 CAD 上游：让 Claude 中转生成 build123d 代码，再用 cad_runner 子进程
执行导出 STEP/STL/GLB。执行失败时把报错喂回 LLM 重修（对应 CAD Skill 的
"repair and rerun" 思路）。

复用 upstream_claude 的 CLAUDE_BASE/KEY/MODEL 与 httpx 调用骨架，区别仅在于
产物是"可执行的 build123d 代码"而非 SVG。
"""
from __future__ import annotations

import logging
import os
import re

import httpx

from . import cad_runner

log = logging.getLogger("image2.cad")

CLAUDE_BASE = os.getenv("CLAUDE_BASE", "https://claude.moon9.cloud").rstrip("/")
CLAUDE_KEY = os.getenv("CLAUDE_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "kiro-opus-4.7")

# 注意：整条管线（LLM 调用 + 子进程执行，含重修）的最坏耗时必须留在后台清扫
# 任务的 pending 超时窗口（tasks.GEN_TIMEOUT_SECONDS=300s）以内，否则会与 reaper
# 的自动退款撞车导致重复退款。默认 read=90s、max_repairs=1、CAD_EXEC_TIMEOUT=60s：
# 最坏 2*90 + 2*60 = 300s。若调大 CAD_EXEC_TIMEOUT 或 max_repairs，请同步评估。
_TIMEOUT = httpx.Timeout(connect=15.0, read=90.0, write=60.0, pool=15.0)
_HTTP_RETRIES = 1  # 502/503/504/连接错误的网络重试（压低以控制总耗时）


class CadError(RuntimeError):
    pass


_SYSTEM_PROMPT = """你是一名精通 build123d 的参数化 CAD 工程师。根据用户的自然语言需求，
编写一段 build123d Python 代码来构建模型。

# 硬性约定
- 只用 build123d 标准 API（`from build123d import *` 已在运行环境中导入，你无需再写 import）。
- 必须把最终模型赋值给名为 `result` 的变量（Part / Solid / Compound / Shape 均可）。
- 单位默认毫米(mm)；XY 为基面，+Z 向上；产出封闭实体（watertight solid）。
- 用户没给的尺寸用工程上合理的默认值，并保证各特征不互相穿插出错。
- 禁止任何文件 / 网络 / 系统调用（os、sys、subprocess、open、eval、exec 等一律不准），只做几何建模。
- 不要打印、不要写文件、不要 if __name__ 守卫，代码会被直接内联执行。

# 输出格式（严格）
[PLAN] 一句话说明你的建模思路（不超过 40 字）。
```python
# 这里是完整的 build123d 代码，最后 result = ...
```

# 风格提示
- 优先用 Builder 模式（with BuildPart() as ...）或直接的代数建模（Box/Cylinder/...），怎么稳怎么来。
- 复杂特征（孔、圆角、倒角）确认引用的面/边存在再操作，避免空选择集报错。
"""


def _build_repair_prompt(prev_code: str, error: str) -> str:
    return (
        "上一版代码执行失败，请修复后重新输出完整代码。\n\n"
        f"# 出错代码\n```python\n{prev_code}\n```\n\n"
        f"# 运行报错（截断）\n{error[-1200:]}\n\n"
        "请定位根因并修正，仍按 [PLAN] + ```python``` 代码块格式输出完整可执行代码。"
    )


def extract_code(text: str) -> str:
    """从模型回复里提取 python 代码块。"""
    if not text:
        raise CadError("返回内容为空")
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        code = m.group(1).strip()
    else:
        # 没有围栏：退而求其次，从第一处出现 result= / build123d 关键字起截取
        code = text.strip()
    if "result" not in code:
        raise CadError(f"代码中未见 `result` 变量赋值。返回片段：{text[:300]}")
    return code


def _extract_plan(text: str) -> str:
    m = re.search(r"\[PLAN\]\s*(.+)", text)
    return m.group(1).strip()[:120] if m else ""


async def _call_claude(messages: list[dict]) -> str:
    """调一次 Claude /v1/messages，返回文本内容。骨架照搬 upstream_claude。"""
    url = f"{CLAUDE_BASE}/v1/messages"
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 4096,
        "system": _SYSTEM_PROMPT,
        "messages": messages,
    }
    headers = {
        "x-api-key": CLAUDE_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    last_err = "未知错误"
    data = None
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for attempt in range(_HTTP_RETRIES + 1):
            try:
                resp = await client.post(url, json=body, headers=headers)
            except httpx.HTTPError as e:
                last_err = f"连接失败: {e}"
                log.warning("cad claude attempt %d 连接失败: %s", attempt + 1, e)
                if attempt < _HTTP_RETRIES:
                    continue
                raise CadError(last_err)
            if resp.status_code in (502, 503, 504):
                last_err = f"messages {resp.status_code}: {resp.text[:300]}"
                log.warning("cad claude attempt %d %s", attempt + 1, last_err)
                if attempt < _HTTP_RETRIES:
                    continue
                raise CadError(last_err)
            if resp.status_code >= 400:
                log.error("cad claude %s: %s", resp.status_code, resp.text[:1000])
                try:
                    j = resp.json()
                    msg = j.get("error", {}).get("message") or j.get("message") or resp.text[:300]
                except Exception:
                    msg = resp.text[:300]
                raise CadError(f"messages {resp.status_code}: {msg}")
            try:
                data = resp.json()
                break
            except Exception:
                log.error("cad claude 返回非 JSON: %s", resp.text[:500])
                raise CadError("上游返回非 JSON")
        else:
            raise CadError(last_err)

    if data is None:
        raise CadError(last_err)

    content = data.get("content")
    text = ""
    if isinstance(content, list):
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                text += blk.get("text", "")
    if not text:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            text = choices[0].get("message", {}).get("content", "")
    if not text:
        log.error("cad claude 返回不含文本: %s", str(data)[:500])
        raise CadError("上游返回不含文本内容")
    return text


async def generate_cad(prompt: str, *, max_repairs: int = 1) -> tuple[dict[str, bytes], dict]:
    """生成 CAD 文件。

    返回 (artifacts, meta)，artifacts = {"step":bytes,"stl":bytes,"glb":bytes}，
    meta = {"repairs": n, "plan": str}。全部尝试失败抛 CadError。
    """
    if not CLAUDE_KEY:
        raise CadError("CLAUDE_KEY 未配置")

    messages: list[dict] = [{"role": "user", "content": f"需求：{prompt.strip()}"}]
    last_err = "未知错误"
    plan = ""

    for attempt in range(max_repairs + 1):
        text = await _call_claude(messages)
        if not plan:
            plan = _extract_plan(text)
        try:
            code = extract_code(text)
        except CadError as e:
            last_err = str(e)
            log.warning("cad 提取代码失败 attempt=%d: %s", attempt + 1, last_err)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "没有找到可执行代码块，请重新按格式输出完整 build123d 代码。"})
            continue

        try:
            artifacts = await cad_runner.run_build123d_script(code)
            log.info("cad 生成成功 attempt=%d plan=%s", attempt + 1, plan)
            return artifacts, {"repairs": attempt, "plan": plan}
        except cad_runner.CadRunError as e:
            last_err = str(e)
            log.warning("cad 执行失败 attempt=%d: %s", attempt + 1, last_err)
            if attempt < max_repairs:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": _build_repair_prompt(code, last_err)})
                continue

    raise CadError(f"多次尝试仍失败：{last_err}")
