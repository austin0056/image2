"""公式/理工科图表上游：让 gpt-5.5 生成 matplotlib 代码，再用 chart_runner 子进程执行
产出 PNG。执行失败把报错喂回模型重修。

兼容两种 OpenAI 文本接口：Chat Completions(/chat/completions) 与 Responses(/responses)。
由管理面板 provider="chart" 的 api_style 决定（auto/chat/responses）：auto 会按模型名
自动判别——gpt-5.x 与 o 系列推理模型走 Responses API，其余走 Chat Completions。

配置（base/key/model/api_style）取自管理面板 provider="chart"，不依赖环境变量。
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


def _is_responses_model(model: str) -> bool:
    """按模型名判断是否应走 OpenAI Responses API。gpt-5.x 与 o 系列推理模型默认走 Responses。"""
    m = (model or "").lower().strip()
    if "responses" in m:
        return True
    if m.startswith(("gpt-5", "gpt5")):
        return True
    if re.match(r"^o[1-9]", m):  # o1 / o3 / o4-mini 等
        return True
    return False


def _resolve_style(cfg: dict) -> str:
    style = (cfg.get("api_style") or "auto").lower()
    if style in ("chat", "responses"):
        return style
    return "responses" if _is_responses_model(cfg.get("model", "")) else "chat"


def _extract_responses_text(data: dict) -> str:
    """解析 OpenAI Responses API 返回的文本（兼容多种形状）。"""
    ot = data.get("output_text")  # SDK 便捷字段，部分中转也会带
    if isinstance(ot, str) and ot.strip():
        return ot
    text = ""
    out = data.get("output")
    if isinstance(out, list):
        for item in out:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue  # 跳过 reasoning 等其它条目
            content = item.get("content")
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") in ("output_text", "text"):
                        text += blk.get("text", "")
            elif isinstance(content, str):
                text += content
    if text.strip():
        return text
    # 个别中转把 responses 也包成 chat 形状，兜底再试一次
    return _extract_chat_text(data, _strict=False)


def _extract_chat_text(data: dict, *, _strict: bool = True) -> str:
    """解析 OpenAI Chat Completions 返回的文本。_strict=False 时失败返回空串而非抛错。"""
    choices = data.get("choices") or []
    if not choices:
        if not _strict:
            return ""
        # 兜底：也许其实是 responses 形状
        t = _extract_responses_text(data)
        if t:
            return t
        raise ChartError("上游无 choices 返回")
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):  # 某些实现返回分段
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content or ""


async def _post_json(url: str, body: dict, headers: dict, *, label: str) -> dict:
    """统一的带重试 POST，返回解析后的 JSON；失败抛 ChartError。"""
    last_err = "未知错误"
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
                last_err = f"{label} {resp.status_code}: {resp.text[:300]}"
                if attempt < _HTTP_RETRIES:
                    continue
                raise ChartError(last_err)
            if resp.status_code >= 400:
                log.error("chart %s %s: %s", label, resp.status_code, resp.text[:800])
                try:
                    j = resp.json()
                    msg = j.get("error", {}).get("message") or j.get("message") or resp.text[:300]
                except Exception:
                    msg = resp.text[:300]
                raise ChartError(f"{label} {resp.status_code}: {msg}")
            try:
                return resp.json()
            except Exception:
                raise ChartError("上游返回非 JSON")
    raise ChartError(last_err)


async def _call_llm(messages: list[dict]) -> str:
    """调一次公式/图表模型，返回文本。按 api_style 选择 Chat Completions 或 Responses API。"""
    cfg = await db.get_provider("chart")
    if not cfg["base"] or not cfg["key"]:
        raise ChartError("公式/图表模型未配置（管理面板 → AI 提供商 → 公式图表）")
    headers = {"Authorization": f"Bearer {cfg['key']}", "Content-Type": "application/json"}
    style = _resolve_style(cfg)

    if style == "responses":
        # OpenAI Responses API：system 用 instructions，对话用 input；输出在 output[].content[].text
        url = f"{cfg['base']}/responses"
        body = {
            "model": cfg["model"],
            "instructions": _SYSTEM_PROMPT,
            "input": [{"role": m["role"], "content": m["content"]} for m in messages],
            "max_output_tokens": 8192,  # 推理模型会先耗 reasoning token，给足避免截断
        }
        data = await _post_json(url, body, headers, label="responses")
        content = _extract_responses_text(data)
    else:
        url = f"{cfg['base']}/chat/completions"
        body = {
            "model": cfg["model"],
            "messages": [{"role": "system", "content": _SYSTEM_PROMPT}] + messages,
            "max_tokens": 4096,
        }
        data = await _post_json(url, body, headers, label="chat")
        content = _extract_chat_text(data)

    if not content:
        log.error("chart 上游返回不含文本 style=%s data=%s", style, str(data)[:500])
        raise ChartError("上游返回不含文本内容")
    return content


async def generate_chart(prompt: str, *, max_repairs: int = 1) -> tuple[bytes, dict]:
    """返回 (png_bytes, meta)。全部尝试失败抛 ChartError。"""
    messages: list[dict] = [{"role": "user", "content": f"需求：{prompt.strip()}"}]
    last_err = "未知错误"
    plan = ""
    for attempt in range(max_repairs + 1):
        text = await _call_llm(messages)
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
