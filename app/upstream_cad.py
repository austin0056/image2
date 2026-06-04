"""文字转 CAD 上游：让 Claude 中转生成 build123d 代码，再用 cad_runner 子进程
执行导出 STEP/STL/GLB。执行失败时把报错喂回 LLM 重修（对应 CAD Skill 的
"repair and rerun" 思路）。

复用 upstream_claude 的 CLAUDE_BASE/KEY/MODEL 与 httpx 调用骨架，区别仅在于
产物是"可执行的 build123d 代码"而非 SVG。
"""
from __future__ import annotations

import logging
import re

import httpx

from . import cad_runner, upstream_claude

log = logging.getLogger("image2.cad")

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

# build123d API 正确用法（高频错误，务必遵守）
- 选圆形边：用 `obj.edges().filter_by(GeomType.CIRCLE)`。
  绝不要写 `e.geom_type()`——geom_type 是**属性不是方法**（调用它会报 'GeomType' object is not callable）；
  也绝不要 `isinstance(e.geom_type, Circle)`——Circle 是 2D 草图类、不是边的类型。
  确需判断时用比较：`e.geom_type == GeomType.CIRCLE`（同理 LINE/ELLIPSE/BSPLINE 等）。
- 选面：按方向用 `obj.faces().filter_by(Plane.XY)`，或取顶面/底面 `obj.faces().sort_by(Axis.Z)[-1]` / `[0]`。
- 选边/面还可用 `.filter_by(Axis.Z)`、`.group_by(Axis.Z)`、`.sort_by(...)`；用 `[...]` 取子集前先确认非空。
- 圆角/倒角：`fillet(edges, radius=r)`、`chamfer(edges, length=c)`，edges 传选择集（可空判断后再用）。
- 打孔优先布尔减：`result = base - Pos(x, y, z) * Cylinder(radius=r, height=h)`；
  代数建模里平移用 `Pos(x,y,z) * 形体`、旋转用 `Rot(rx,ry,rz) * 形体`，不要臆造 `.translate()/.move()` 之类方法名。
- 这些名字已随 `from build123d import *` 导入，直接用、别重新定义：
  GeomType、Axis、Plane、Align、Mode、Pos、Rot、Box、Cylinder、Sphere、Cone、fillet、chamfer 等。

# 风格提示
- 优先用 Builder 模式（with BuildPart() as ...）或直接的代数建模（Box/Cylinder/...），怎么稳怎么来。
- 复杂特征（孔、圆角、倒角）确认引用的面/边存在再操作，避免空选择集报错。
- 能用布尔运算（+ - &）和 filter_by 解决的，就别手写几何类型判断，更稳。
"""


def _build_repair_prompt(prev_code: str, error: str) -> str:
    return (
        "上一版代码执行失败，请修复后重新输出完整代码。\n\n"
        f"# 出错代码\n```python\n{prev_code}\n```\n\n"
        f"# 运行报错（截断）\n{error[-1200:]}\n\n"
        "请定位根因并修正。注意 build123d 高频错误：geom_type 是属性不是方法"
        "（用 `e.geom_type == GeomType.CIRCLE` 或 `edges().filter_by(GeomType.CIRCLE)`，"
        "不要 `e.geom_type()`，也不要对 Circle 用 isinstance）；面/边选择集取下标前先确认非空。"
        "仍按 [PLAN] + ```python``` 代码块格式输出完整可执行代码。"
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
    """调一次 Claude 中转，返回文本内容。

    通过 upstream_claude.complete_text 统一处理协议：自适配 Anthropic Messages /
    OpenAI Chat / Responses，并在 404/405 时自动回退——解决“messages 404: Not Found”
    （中转只提供 OpenAI /chat/completions 时）。配置取自管理面板 provider="claude"。
    """
    return await upstream_claude.complete_text(
        _SYSTEM_PROMPT, messages,
        max_tokens=4096, timeout=_TIMEOUT, retries=_HTTP_RETRIES, err_cls=CadError,
    )


async def generate_cad(prompt: str, *, max_repairs: int = 1) -> tuple[dict[str, bytes], dict]:
    """生成 CAD 文件。

    返回 (artifacts, meta)，artifacts = {"step":bytes,"stl":bytes,"glb":bytes}，
    meta = {"repairs": n, "plan": str}。全部尝试失败抛 CadError。
    """
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
