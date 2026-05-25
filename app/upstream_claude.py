"""Claude 中转上游封装：Anthropic 原生 /v1/messages 协议。
专注用例：高质量 SVG 图标，模仿主流图标库，带 CoT 推理与审美约束。"""
from __future__ import annotations

import logging
import os
import re
from xml.etree import ElementTree as ET

import httpx

log = logging.getLogger("image2.claude")

CLAUDE_BASE = os.getenv("CLAUDE_BASE", "https://claude.moon9.cloud").rstrip("/")
CLAUDE_KEY = os.getenv("CLAUDE_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "kiro-opus-4.7")

# 上游 nginx 大约 120s 切断；这里设 110s 给一点余量
_TIMEOUT = httpx.Timeout(connect=15.0, read=110.0, write=60.0, pool=15.0)
_MAX_RETRIES = 2  # 503/504/连接错误的重试次数


class ClaudeError(RuntimeError):
    pass


# ---------- 各图标库的设计语言 ----------
LIBRARY_GUIDES: dict[str, str] = {
    "lucide": (
        'Lucide：viewBox="0 0 24 24"；fill="none" stroke="currentColor" '
        'stroke-width="2"；linecap/linejoin="round"；2px 边距，几何对称。'
    ),
    "heroicons-outline": (
        'Heroicons Outline：viewBox="0 0 24 24"；fill="none" stroke="currentColor" '
        'stroke-width="1.5"；linecap/linejoin="round"；1.5px 边距，曲线优雅。'
    ),
    "heroicons-solid": (
        'Heroicons Solid：viewBox="0 0 24 24"；纯填充 fill="currentColor"，无 stroke；'
        'fill-rule="evenodd" clip-rule="evenodd"。'
    ),
    "phosphor": (
        'Phosphor：viewBox="0 0 256 256"；fill="none" stroke="currentColor" '
        'stroke-width="16"；linecap/linejoin="round"；圆润，主元素 24-232。'
    ),
    "tabler": (
        'Tabler：viewBox="0 0 24 24"；fill="none" stroke="currentColor" '
        'stroke-width="2"；linecap/linejoin="round"；像素严格对齐。'
    ),
    "feather": (
        'Feather：viewBox="0 0 24 24"；fill="none" stroke="currentColor" '
        'stroke-width="2"；极简，更多负空间。'
    ),
    "material": (
        'Material Symbols：viewBox="0 0 24 24"；纯填充 fill="currentColor"；'
        '4px 网格对齐，无 stroke。'
    ),
    "duotone": (
        'Duotone：viewBox="0 0 24 24"；先画填充层 fill="#SECONDARY" opacity="0.2"，'
        '再画轮廓层 stroke="#PRIMARY" stroke-width="2" fill="none"。'
    ),
}


def _build_system_prompt(library: str, dual_color: bool) -> str:
    guide = LIBRARY_GUIDES.get(library, "")
    color_section = (
        "用户给主色 PRIMARY 和辅色 SECONDARY 时：stroke/主轮廓用 PRIMARY，"
        "fill/强调用 SECONDARY；任一为空则该位置使用 currentColor。"
        if dual_color
        else "用户给主色时替换 currentColor；为空则保持 currentColor。"
    )
    parts = [
        "你是一名顶尖的图标设计师，擅长仿照主流开源图标库。",
        "请先简要思考再输出 SVG。",
        "",
        "# 输出格式（严格）",
        "[ANALYSIS] 一句话拆解主题。",
        "[CONCEPT] 一句话说你选什么构图。",
        "[GEOMETRY] 一句话说元素位置。",
        "[PALETTE] 一句话说颜色如何分配。",
        "[SVG]",
        "<svg ...>...</svg>",
        "",
        "# 硬要求",
        "- 前 4 段合计不超 80 字。",
        "- [SVG] 后面是完整 SVG，不要 markdown 围栏。",
        "- 禁止 <script>/<foreignObject>/远程 xlink:href。",
        "",
        "# 库规范",
        guide or "默认 24x24 线性。",
        "",
        "# 颜色",
        color_section,
        "",
        "# 关键原则",
        "- 同图 stroke-width 一致，端点落整数坐标。",
        "- 边距 ≥ 8% viewBox，核心形状 ≤ 5。",
        "- 能用 circle/rect/line 就不用 path；圆角 rx 用 2/4/6。",
        "- 默认对称；16px 仍需可识别。",
        "",
        "思考要短，重点是 SVG 本身质量。",
    ]
    return "\n".join(parts)


def _build_user_prompt(
    prompt: str,
    library: str,
    color_primary: str,
    color_secondary: str,
    stroke_width: float | None,
) -> str:
    parts = [f"主题：{prompt.strip()}"]
    if library and library != "auto":
        parts.append(f"参考图标库：{library}")
    if color_primary:
        parts.append(f"主色 PRIMARY = {color_primary}")
    if color_secondary:
        parts.append(f"辅色 SECONDARY = {color_secondary}")
    if not color_primary and not color_secondary:
        parts.append("未指定颜色，请使用 currentColor。")
    if stroke_width:
        parts.append(f"笔画粗细：{stroke_width}（覆盖默认）")
    parts.append("严格按 [ANALYSIS]→[CONCEPT]→[GEOMETRY]→[PALETTE]→[SVG] 输出，思考段每段一句话。")
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
) -> tuple[str, list[str]]:
    if not CLAUDE_KEY:
        raise ClaudeError("CLAUDE_KEY 未配置")

    dual = bool(color_primary or color_secondary) or library == "duotone"
    url = f"{CLAUDE_BASE}/v1/messages"
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 3072,
        "system": _build_system_prompt(library, dual_color=dual),
        "messages": [
            {
                "role": "user",
                "content": _build_user_prompt(
                    prompt, library, color_primary, color_secondary, stroke_width
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
        "claude POST %s model=%s library=%s dual=%s",
        url, CLAUDE_MODEL, library, dual,
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

            # 上游网关错误：可重试
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
    return svg, warnings
