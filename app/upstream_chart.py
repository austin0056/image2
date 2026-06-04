"""公式/理工科图表上游：让 gpt-5.5（OpenAI 兼容 chat 接口）生成 matplotlib 代码，
再用 chart_runner 子进程执行产出 PNG。执行失败把报错喂回模型重修。

配置（base/key/model）取自管理面板 provider="chart"，不依赖环境变量。
"""
from __future__ import annotations

import logging
import re

import httpx

from . import chart_runner, db

log = logging.getLogger("image2.chart")

# 已异步化（后台任务 + 轮询），不受 Cloudflare 100s 限制，读超时给足
_TIMEOUT = httpx.Timeout(connect=15.0, read=180.0, write=60.0, pool=15.0)
_HTTP_RETRIES = 1


class ChartError(RuntimeError):
    pass


_SYSTEM_PROMPT = """你是科学绘图与数学公式排版专家。根据用户需求，用 Python + matplotlib 绘图。

# 硬性约定
- 环境已提供 `plt`(matplotlib.pyplot) 和 `np`(numpy)，也可自行 import；不要用其它第三方库。
- 数学公式用 matplotlib 的 mathtext：在文本里写 `$...$`，例如
  `plt.text(0.5, 0.5, r"$\\hat{n} = f/\\|f\\|$", fontsize=28, ha="center", va="center"); plt.axis("off")`。
  不要启用 usetex（系统无 LaTeX）。
- 图表（折线/柱状/散点/函数曲线/等高线等）要有坐标轴、标题、必要的图例；理工科风格、清晰。
- 只画一张图（单个 figure）。不要 plt.show()、不要保存文件、不要任何文件/网络/系统调用。
- 中文标签可能缺字体，优先用英文标注，或对纯公式用 mathtext。

# 输出格式（严格）
[PLAN] 一句话说明要画什么（≤30 字）。
```python
# 完整 matplotlib 代码
```
"""


def _build_repair_prompt(prev_code: str, error: str) -> str:
    return (
        "上一版代码执行失败，请修复后重新输出完整代码。\n\n"
        f"# 出错代码\n```python\n{prev_code}\n```\n\n"
        f"# 运行报错（截断）\n{error[-1200:]}\n\n"
        "请定位根因并修正，仍按 [PLAN] + ```python``` 代码块输出完整可执行代码。"
    )


def extract_code(text: str) -> str:
    if not text:
        raise ChartError("返回内容为空")
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    code = m.group(1).strip() if m else text.strip()
    if "plt" not in code and "pyplot" not in code:
        raise ChartError(f"代码里没看到 matplotlib 绘图。返回片段：{text[:300]}")
    return code


def _extract_plan(text: str) -> str:
    m = re.search(r"\[PLAN\]\s*(.+)", text)
    return m.group(1).strip()[:120] if m else ""


async def _call_chat(messages: list[dict]) -> str:
    cfg = await db.get_provider("chart")
    if not cfg["base"] or not cfg["key"]:
        raise ChartError("公式/图表模型未配置（管理面板 → AI 提供商 → 公式图表）")
    url = f"{cfg['base']}/chat/completions"
    body = {
        "model": cfg["model"],
        "messages": [{"role": "system", "content": _SYSTEM_PROMPT}] + messages,
        "max_tokens": 4096,
    }
    headers = {"Authorization": f"Bearer {cfg['key']}", "Content-Type": "application/json"}
    last_err = "未知错误"
    data = None
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for attempt in range(_HTTP_RETRIES + 1):
            try:
                resp = await client.post(url, json=body, headers=headers)
            except httpx.HTTPError as e:
                last_err = f"连接失败: {e}"
                if attempt < _HTTP_RETRIES:
                    continue
                raise ChartError(last_err)
            if resp.status_code in (502, 503, 504):
                last_err = f"chat {resp.status_code}: {resp.text[:300]}"
                if attempt < _HTTP_RETRIES:
                    continue
                raise ChartError(last_err)
            if resp.status_code >= 400:
                try:
                    j = resp.json()
                    msg = j.get("error", {}).get("message") or j.get("message") or resp.text[:300]
                except Exception:
                    msg = resp.text[:300]
                raise ChartError(f"chat {resp.status_code}: {msg}")
            try:
                data = resp.json()
                break
            except Exception:
                raise ChartError("上游返回非 JSON")
        else:
            raise ChartError(last_err)

    if data is None:
        raise ChartError(last_err)
    choices = data.get("choices") or []
    if not choices:
        raise ChartError("上游无 choices 返回")
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):  # 某些实现返回分段
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    if not content:
        raise ChartError("上游返回不含文本内容")
    return content


async def generate_chart(prompt: str, *, max_repairs: int = 1) -> tuple[bytes, dict]:
    """返回 (png_bytes, meta)。全部尝试失败抛 ChartError。"""
    messages: list[dict] = [{"role": "user", "content": f"需求：{prompt.strip()}"}]
    last_err = "未知错误"
    plan = ""
    for attempt in range(max_repairs + 1):
        text = await _call_chat(messages)
        if not plan:
            plan = _extract_plan(text)
        try:
            code = extract_code(text)
        except ChartError as e:
            last_err = str(e)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "没有找到可执行的 matplotlib 代码块，请重新按格式输出完整代码。"})
            continue
        try:
            png = await chart_runner.run_matplotlib_script(code)
            log.info("chart 生成成功 attempt=%d plan=%s", attempt + 1, plan)
            return png, {"repairs": attempt, "plan": plan}
        except chart_runner.ChartRunError as e:
            last_err = str(e)
            log.warning("chart 执行失败 attempt=%d: %s", attempt + 1, last_err)
            if attempt < max_repairs:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": _build_repair_prompt(code, last_err)})
                continue
    raise ChartError(f"多次尝试仍失败：{last_err}")
