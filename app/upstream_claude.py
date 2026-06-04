"""Claude 中转上游封装：Anthropic 原生 /v1/messages 协议。
图标生成核心：先检索真实图标库的 top-k 样本作为 few-shot，
让 LLM 学习设计师精调过的视觉语言再创作，质量远高于纯凭审美 checklist。"""
from __future__ import annotations

import logging
import re
from xml.etree import ElementTree as ET

import httpx

from app import db, icon_search

log = logging.getLogger("image2.claude")

# 上游 nginx 大约 120s 切断；这里设 110s 给一点余量
_TIMEOUT = httpx.Timeout(connect=15.0, read=110.0, write=60.0, pool=15.0)
_MAX_RETRIES = 2  # 503/504/连接错误的重试次数

# 每次注入多少个样本(太多会让 prompt 过长,3-5 个最合适)
_FEW_SHOT_K = 5


class ClaudeError(RuntimeError):
    pass


# ============================================================
# 统一文本补全：自适配 Anthropic Messages / OpenAI Chat / Responses
# CAD 与图标都走 provider="claude"，但中转协议各家不同（很多中转只提供
# OpenAI /chat/completions，没有 Anthropic /v1/messages → 404）。
# 这里按 api_style + 模型名选协议，并在 404/405 时自动回退，彻底解决
# “messages 404: Not Found”。
# ============================================================
def _claude_attempt_order(api_style: str, model: str) -> list[str]:
    style = (api_style or "auto").lower()
    if style in ("messages", "chat", "responses"):
        return [style]
    m = (model or "").lower()
    if m.startswith(("gpt-5", "gpt5")) or re.match(r"^o[1-9]", m):
        return ["responses", "chat", "messages"]
    if any(k in m for k in ("gpt", "deepseek", "qwen", "glm", "moonshot", "kimi", "grok", "llama", "mistral", "doubao", "ernie")):
        return ["chat", "messages", "responses"]
    # 形似 Claude（claude/opus/sonnet/haiku/kiro）或未知：先 Anthropic，再回退 OpenAI
    return ["messages", "chat", "responses"]


def _extract_text_any(data: dict) -> str:
    """从三种协议的返回里抽取助手文本。"""
    content = data.get("content")  # Anthropic messages
    if isinstance(content, list):
        t = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        if t.strip():
            return t
    ot = data.get("output_text")  # OpenAI responses 便捷字段
    if isinstance(ot, str) and ot.strip():
        return ot
    out = data.get("output")  # OpenAI responses 标准结构
    if isinstance(out, list):
        t = ""
        for item in out:
            if isinstance(item, dict) and item.get("type") == "message":
                c = item.get("content")
                if isinstance(c, list):
                    for blk in c:
                        if isinstance(blk, dict) and blk.get("type") in ("output_text", "text"):
                            t += blk.get("text", "")
                elif isinstance(c, str):
                    t += c
        if t.strip():
            return t
    choices = data.get("choices")  # OpenAI chat
    if isinstance(choices, list) and choices:
        c = choices[0].get("message", {}).get("content", "")
        if isinstance(c, list):
            c = "".join(b.get("text", "") for b in c if isinstance(b, dict))
        if c:
            return c
    return ""


def _build_request(fmt: str, base: str, model: str, key: str, system: str, messages: list[dict], max_tokens: int):
    if fmt == "messages":
        return (f"{base}/v1/messages",
                {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                {"model": model, "max_tokens": max_tokens, "system": system, "messages": messages})
    if fmt == "responses":
        return (f"{base}/responses",
                {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                {"model": model, "instructions": system, "input": messages, "max_output_tokens": max(max_tokens, 4096)})
    return (f"{base}/chat/completions",
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            {"model": model, "max_tokens": max_tokens, "messages": [{"role": "system", "content": system}] + messages})


async def complete_text(system: str, messages: list[dict], *, max_tokens: int = 4096,
                        timeout=None, retries: int = 1, err_cls: type = ClaudeError) -> str:
    """统一的 Claude 中转文本补全。配置取自 provider="claude"。

    协议选择与回退（重要：必须有界，避免拖到被清扫任务超时退款）：
    - api_style=auto 时按模型名选协议顺序。
    - 只有 404/405（路径不存在）才**快速**换下一种协议——这是发现正确协议的唯一信号。
    - 读超时：请求已发出、只是慢；换协议只会再等一遍，故**直接失败**（不跨协议放大耗时）。
    - 连接失败：同一 host 换协议也连不上，重试一次后失败。
    - 其它 4xx（401/400/模型不存在等）：换协议无意义，带上游原文直接报错。
    - 5xx：瞬时错误，重试一次后再换协议。
    """
    cfg = await db.get_provider("claude")
    if not cfg["key"]:
        raise err_cls("Claude 中转未配置（管理面板 → AI 提供商）")
    base, model, key = cfg["base"], cfg["model"], cfg["key"]
    order = _claude_attempt_order(cfg.get("api_style", "auto"), model)
    last_err = "未知错误"

    async with httpx.AsyncClient(timeout=timeout or _TIMEOUT) as client:
        for fmt in order:
            url, headers, body = _build_request(fmt, base, model, key, system, messages, max_tokens)
            for attempt in range(retries + 1):
                try:
                    resp = await client.post(url, json=body, headers=headers)
                except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                    last_err = f"连接失败（请检查 Base URL/网络）: {e}"
                    if attempt < retries:
                        continue
                    raise err_cls(last_err)
                except httpx.TimeoutException as e:
                    # 读/写/连接池超时：请求已发出但慢，换协议或重试都只会再等一遍 → 快速失败
                    raise err_cls(f"上游响应超时（{type(e).__name__}，模型太慢或中转排队）；"
                                  f"可在管理面板把该上游模型换快一点，或指定 API 风格。")
                except httpx.HTTPError as e:
                    last_err = f"网络错误: {e}"
                    if attempt < retries:
                        continue
                    raise err_cls(last_err)
                if resp.status_code in (404, 405):
                    last_err = f"{fmt} {resp.status_code}: 该中转无此协议路径"
                    log.warning("claude %s 返回 %s，自动尝试下一种协议", url, resp.status_code)
                    break  # 快速换下一种协议
                if resp.status_code in (502, 503, 504):
                    last_err = f"{fmt} {resp.status_code}: {resp.text[:200]}"
                    if attempt < retries:
                        continue
                    break  # 5xx 用尽 → 换下一种协议
                if resp.status_code >= 400:
                    log.error("claude %s %s: %s", fmt, resp.status_code, resp.text[:800])
                    try:
                        j = resp.json()
                        msg = j.get("error", {}).get("message") or j.get("message") or resp.text[:300]
                    except Exception:
                        msg = resp.text[:300]
                    raise err_cls(f"{fmt} {resp.status_code}: {msg}")
                try:
                    data = resp.json()
                except Exception:
                    raise err_cls("上游返回非 JSON")
                text = _extract_text_any(data)
                if text:
                    return text
                last_err = f"{fmt} 返回不含文本内容"
                break  # 换下一种协议
    raise err_cls(last_err)


# 各图标库的简短风格定位(配合样本一起注入)
LIBRARY_NOTES: dict[str, str] = {
    "lucide": "Lucide：viewBox=24，纯线性，stroke-width=2，圆角端点。",
    "heroicons-outline": "Heroicons Outline：viewBox=24，stroke-width=1.5，曲线优雅。",
    "heroicons-solid": "Heroicons Solid：viewBox=24，纯填充无 stroke。",
    "phosphor": "Phosphor：viewBox=256，stroke-width=16，圆润友好。",
    "tabler": "Tabler：viewBox=24，stroke-width=2，像素严格对齐。",
    "feather": "Feather：viewBox=24，stroke-width=2，极简风格。",
    "material": "Material：viewBox=24，纯填充，4px 网格。",
    "duotone": "Duotone：viewBox=24，填充层 + 轮廓层叠加。",
    "auto": "由 AI 自由发挥，建议线性 24x24。",
}


def _build_system_prompt(library: str, examples: list[dict], dual_color: bool) -> str:
    note = LIBRARY_NOTES.get(library, "")
    color_section = (
        "颜色规则：把 stroke/主轮廓换成 PRIMARY；fill/强调换成 SECONDARY；"
        "任一为空则该位置使用 currentColor。"
        if dual_color
        else "颜色规则：用户给主色时替换 currentColor；为空保持 currentColor。"
    )

    examples_block = ""
    if examples:
        parts = ["# 学习样本（请仔细观察以下图标的 viewBox/stroke/几何构造/留白节奏）"]
        for i, ex in enumerate(examples, 1):
            parts.append(f"\n## 样本 {i}: {ex['name']}\n{ex['svg']}")
        parts.append(
            "\n请观察样本的：\n"
            "- viewBox 范围和坐标使用习惯\n"
            "- stroke-width / linecap / linejoin 配置\n"
            "- 元素数量级（通常 1-5 个 path/circle/rect）\n"
            "- 留白比例和对称性\n"
            "**生成时严格匹配上述样本的视觉语言，但不要复制任何样本的内容。要原创。**"
        )
        examples_block = "\n".join(parts)

    body = [
        "你是一名顶尖的图标设计师。请严格学习下面真实图标库的样本风格,然后为新主题创作。",
        "",
        f"# 当前库: {library}",
        note,
        "",
        examples_block if examples_block else "(本次未提供样本,请按库默认风格创作)",
        "",
        "# 颜色",
        color_section,
        "",
        "# 输出格式（严格）",
        "[ANALYSIS] 一句话拆解主题。",
        "[CONCEPT] 一句话说你借鉴了哪些样本元素以及如何重组。",
        "[SVG]",
        "<svg ...>...</svg>",
        "",
        "# 硬要求",
        "- 前两段思考合计不超 60 字。",
        "- [SVG] 后是完整 SVG，不要 markdown 围栏。",
        "- viewBox 必须与样本一致。",
        "- stroke 属性配置必须与样本一致。",
        "- 禁止 <script>/<foreignObject>/远程 xlink:href。",
        "- 元素数量与样本同级（≤ 5 个核心形状）。",
        "",
        "重点:学样本的设计语言,不学样本的具体造型。原创但风格一致。",
    ]
    return "\n".join(body)


def _build_user_prompt(
    prompt: str,
    library: str,
    color_primary: str,
    color_secondary: str,
    stroke_width: float | None,
    examples: list[dict],
) -> str:
    parts = [f"主题：{prompt.strip()}"]
    if examples:
        sample_names = ", ".join(e["name"] for e in examples)
        parts.append(f"参考的样本：{sample_names}")
    if color_primary:
        parts.append(f"主色 PRIMARY = {color_primary}")
    if color_secondary:
        parts.append(f"辅色 SECONDARY = {color_secondary}")
    if not color_primary and not color_secondary:
        parts.append("未指定颜色，使用 currentColor。")
    if stroke_width:
        parts.append(f"笔画粗细：{stroke_width}（覆盖默认）")
    parts.append("严格按 [ANALYSIS]→[CONCEPT]→[SVG] 输出，思考段每段一句话。")
    return "\n".join(parts)


def _extract_svg(text: str) -> str:
    if not text:
        raise ClaudeError("返回内容为空")
    svg_seg = text
    m_seg = re.search(r"\[SVG\]\s*", text, flags=re.IGNORECASE)
    if m_seg:
        svg_seg = text[m_seg.end():]
    svg_seg = re.sub(r"^```(?:svg|xml|html)?\s*", "", svg_seg.strip(), flags=re.IGNORECASE)
    svg_seg = re.sub(r"\s*```$", "", svg_seg.strip())
    m = re.search(r"<svg[\s\S]*?</svg>", svg_seg, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"<svg[\s\S]*?</svg>", text, flags=re.IGNORECASE)
        if not m:
            raise ClaudeError(f"未找到 SVG 标签。返回片段: {text[:300]}")
    svg = m.group(0)
    try:
        ET.fromstring(svg)
    except ET.ParseError as e:
        raise ClaudeError(f"SVG 解析失败: {e}")
    if re.search(r"<script\b", svg, flags=re.IGNORECASE):
        raise ClaudeError("SVG 含 <script>，已拒绝")
    if "xmlns" not in svg[:200]:
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    return svg


def _audit_svg(svg: str) -> list[str]:
    warnings: list[str] = []
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return ["SVG 解析失败"]
    descendants = list(root.iter())
    n_paths = sum(1 for el in descendants if el.tag.endswith("path"))
    n_shapes = sum(
        1 for el in descendants
        if any(el.tag.endswith(t) for t in ("circle", "rect", "line", "polygon", "polyline", "ellipse", "path"))
    )
    if n_shapes > 8:
        warnings.append(f"形状数量较多 ({n_shapes})，建议 ≤ 5")
    if n_paths > 6:
        warnings.append(f"path 数量较多 ({n_paths})")
    strokes = set()
    for el in descendants:
        sw = el.get("stroke-width")
        if sw:
            try:
                strokes.add(float(sw))
            except ValueError:
                pass
    if len(strokes) > 1:
        warnings.append(f"笔画粗细不一致：{sorted(strokes)}")
    return warnings


async def generate_icon_svg(
    prompt: str,
    *,
    library: str = "lucide",
    color_primary: str = "",
    color_secondary: str = "",
    stroke_width: float | None = None,
) -> tuple[str, list[str], list[str]]:
    """生成图标 SVG。
    返回 (svg, quality_warnings, sample_names)
    """
    cfg = await db.get_provider("claude")
    if not cfg["key"]:
        raise ClaudeError("Claude 中转未配置（管理面板 → AI 提供商）")

    # 1. 检索 few-shot 样本
    try:
        examples = icon_search.search_examples(prompt, library, top_k=_FEW_SHOT_K)
    except Exception as e:
        log.warning("icon_search 异常,跳过样本: %s", e)
        examples = []

    sample_names = [ex["name"] for ex in examples]

    # 2. 构造 prompt
    dual = bool(color_primary or color_secondary) or library == "duotone"
    log.info("claude icon model=%s library=%s dual=%s samples=%s", cfg["model"], library, dual, sample_names)
    text = await complete_text(
        system=_build_system_prompt(library, examples, dual_color=dual),
        messages=[{
            "role": "user",
            "content": _build_user_prompt(prompt, library, color_primary, color_secondary, stroke_width, examples),
        }],
        max_tokens=3072,
    )

    svg = _extract_svg(text)
    warnings = _audit_svg(svg)
    if warnings:
        log.info("svg quality warnings: %s", warnings)
    return svg, warnings, sample_names
