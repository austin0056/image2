"""Claude 中转上游封装：Anthropic 原生 /v1/messages 协议。
图标生成核心：先检索真实图标库的 top-k 样本作为 few-shot，
让 LLM 学习设计师精调过的视觉语言再创作，质量远高于纯凭审美 checklist。"""
from __future__ import annotations

import logging
import os
import re
from xml.etree import ElementTree as ET

import httpx

from app import icon_search

log = logging.getLogger("image2.claude")

CLAUDE_BASE = os.getenv("CLAUDE_BASE", "https://claude.moon9.cloud").rstrip("/")
CLAUDE_KEY = os.getenv("CLAUDE_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "kiro-opus-4.7")

# 上游 nginx 大约 120s 切断；这里设 110s 给一点余量
_TIMEOUT = httpx.Timeout(connect=15.0, read=110.0, write=60.0, pool=15.0)
_MAX_RETRIES = 2  # 503/504/连接错误的重试次数

# 每次注入多少个样本(太多会让 prompt 过长,3-5 个最合适)
_FEW_SHOT_K = 5


class ClaudeError(RuntimeError):
    pass


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
    if not CLAUDE_KEY:
        raise ClaudeError("CLAUDE_KEY 未配置")

    # 1. 检索 few-shot 样本
    try:
        examples = icon_search.search_examples(prompt, library, top_k=_FEW_SHOT_K)
    except Exception as e:
        log.warning("icon_search 异常,跳过样本: %s", e)
        examples = []

    sample_names = [ex["name"] for ex in examples]

    # 2. 构造 prompt
    dual = bool(color_primary or color_secondary) or library == "duotone"
    url = f"{CLAUDE_BASE}/v1/messages"
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 3072,
        "system": _build_system_prompt(library, examples, dual_color=dual),
        "messages": [
            {
                "role": "user",
                "content": _build_user_prompt(
                    prompt, library, color_primary, color_secondary, stroke_width, examples,
                ),
            },
        ],
    }
    headers = {
        "x-api-key": CLAUDE_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    log.info(
        "claude POST %s model=%s library=%s dual=%s samples=%s",
        url, CLAUDE_MODEL, library, dual, sample_names,
    )

    last_err = "未知错误"
    data = None
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await client.post(url, json=body, headers=headers)
            except httpx.HTTPError as e:
                last_err = f"连接失败: {e}"
                log.warning("claude attempt %d 连接失败: %s", attempt + 1, e)
                if attempt < _MAX_RETRIES:
                    continue
                raise ClaudeError(last_err)

            if resp.status_code in (502, 503, 504):
                snippet = resp.text[:300]
                last_err = f"messages {resp.status_code}: {snippet}"
                log.warning("claude attempt %d %s", attempt + 1, last_err)
                if attempt < _MAX_RETRIES:
                    continue
                raise ClaudeError(last_err)

            if resp.status_code >= 400:
                log.error("claude %s: %s", resp.status_code, resp.text[:1000])
                try:
                    j = resp.json()
                    msg = j.get("error", {}).get("message") or j.get("message") or resp.text[:300]
                except Exception:
                    msg = resp.text[:300]
                raise ClaudeError(f"messages {resp.status_code}: {msg}")
            try:
                data = resp.json()
                break
            except Exception:
                log.error("claude 返回非 JSON: %s", resp.text[:500])
                raise ClaudeError("上游返回非 JSON")
        else:
            raise ClaudeError(last_err)

    if data is None:
        raise ClaudeError(last_err)

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
        log.error("claude 返回不含文本: %s", str(data)[:500])
        raise ClaudeError("上游返回不含文本内容")

    svg = _extract_svg(text)
    warnings = _audit_svg(svg)
    if warnings:
        log.info("svg quality warnings: %s", warnings)
    return svg, warnings, sample_names
