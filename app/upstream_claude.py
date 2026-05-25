"""Claude 中转上游封装：Anthropic 原生 /v1/messages 协议。
专注用例：生成单个高质量 SVG 图标，模仿主流图标库的设计语言。"""
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

_TIMEOUT = httpx.Timeout(connect=15.0, read=120.0, write=60.0, pool=15.0)


class ClaudeError(RuntimeError):
    pass


# ---------- 各图标库的设计语言 ----------
# 这些规则参考自各库的官方贡献指南/设计文档
LIBRARY_GUIDES: dict[str, str] = {
    "lucide": (
        "模仿 Lucide (lucide.dev) 的设计语言：\n"
        "- viewBox=\"0 0 24 24\"，画布严格 24x24\n"
        "- 纯线性：fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\"\n"
        "- stroke-linecap=\"round\" stroke-linejoin=\"round\"\n"
        "- 留 2px 安全边距，主元素在 2,2 到 22,22 区域内\n"
        "- 几何对称、最少元素，避免装饰性细节\n"
        "- 圆角统一，斜线 45°/30°/60° 优先\n"
    ),
    "heroicons-outline": (
        "模仿 Heroicons Outline (heroicons.com)：\n"
        "- viewBox=\"0 0 24 24\"\n"
        "- fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\"\n"
        "- stroke-linecap=\"round\" stroke-linejoin=\"round\"\n"
        "- 比 Lucide 更细，留 1.5px 边距\n"
        "- 优雅的弧线和有机曲线\n"
    ),
    "heroicons-solid": (
        "模仿 Heroicons Solid (heroicons.com)：\n"
        "- viewBox=\"0 0 24 24\"\n"
        "- 纯填充：fill=\"currentColor\"，无 stroke\n"
        "- 用 fill-rule=\"evenodd\" clip-rule=\"evenodd\" 处理孔洞\n"
        "- 实心、扁平、无渐变\n"
        "- 留 1.5px 边距\n"
    ),
    "phosphor": (
        "模仿 Phosphor regular (phosphoricons.com)：\n"
        "- viewBox=\"0 0 256 256\"\n"
        "- 纯线性：fill=\"none\" stroke=\"currentColor\" stroke-width=\"16\"\n"
        "- stroke-linecap=\"round\" stroke-linejoin=\"round\"\n"
        "- 大画布精致细节，主元素在 24,24 到 232,232\n"
        "- 友好的圆润感、稍微夸张的圆角\n"
    ),
    "tabler": (
        "模仿 Tabler Icons (tabler.io/icons)：\n"
        "- viewBox=\"0 0 24 24\"\n"
        "- fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\"\n"
        "- stroke-linecap=\"round\" stroke-linejoin=\"round\"\n"
        "- 类似 Lucide 但允许更复杂结构\n"
        "- 严格像素对齐，边距 2px\n"
    ),
    "feather": (
        "模仿 Feather Icons (feathericons.com)：\n"
        "- viewBox=\"0 0 24 24\"\n"
        "- fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\"\n"
        "- stroke-linecap=\"round\" stroke-linejoin=\"round\"\n"
        "- 极简主义，更少元素，留更多负空间\n"
    ),
    "material": (
        "模仿 Material Symbols (fonts.google.com/icons) 实心风格：\n"
        "- viewBox=\"0 0 24 24\"\n"
        "- 纯填充：fill=\"currentColor\"\n"
        "- 几何块面，对齐 4px 网格\n"
        "- 无 stroke，元素融为一体\n"
    ),
    "duotone": (
        "双色图标风格：\n"
        "- viewBox=\"0 0 24 24\"\n"
        "- 主轮廓 stroke=\"currentColor\" stroke-width=\"2\" fill=\"none\"\n"
        "- 同时有一个填充层 fill=\"currentColor\" opacity=\"0.2\" 表示背景色块\n"
        "- 填充层放在 stroke 层之前（先绘制）\n"
        "- 整体保留线性骨架\n"
    ),
}


def _system_prompt(library: str) -> str:
    """根据选择的图标库构造 system prompt。"""
    guide = LIBRARY_GUIDES.get(library, "")

    base = """你是一名专业的图标设计师，擅长仿照知名开源图标库的视觉语言。
根据用户描述生成一个 SVG 图标。

# 严格输出要求
1. 只返回一段完整的 SVG 代码，从 <svg 开头到 </svg> 结尾。
2. 不要任何解释、markdown 围栏（```）、前后注释。
3. 必须包含 viewBox。
4. 禁止 <script>、<foreignObject>、远程引用（xlink:href、外部 url）。
5. 颜色优先用 currentColor；如果用户指定主色，则使用十六进制色值。

# 设计原则
- 单一图标主题，几何对称，避免堆砌元素。
- 严格对齐到画布的整数像素或半像素。
- 留出安全边距，不要顶到边缘。
- 圆角、笔画端点统一。
- 关键形状用最少的路径表达；能用 <circle>/<rect>/<line> 就不用 <path>。
- 避免渐变、阴影、滤镜（除非用户明确要求）。
"""
    if guide:
        base += "\n# 当前要模仿的视觉风格\n" + guide
    base += "\n用户的颜色偏好优先级最高，可以覆盖默认的 currentColor。"
    return base


def _build_user_prompt(
    prompt: str,
    library: str,
    color: str,
    stroke_width: float | None,
) -> str:
    parts = [f"主题：{prompt.strip()}"]
    if library and library != "auto":
        parts.append(f"参考图标库：{library}")
    if color:
        parts.append(f"主色：{color}（替换 currentColor）")
    if stroke_width:
        parts.append(f"笔画粗细：{stroke_width}（覆盖默认）")
    parts.append("请直接输出 SVG，从 <svg 开始，</svg> 结束，不要任何其他文字。")
    return "\n".join(parts)


def _extract_svg(text: str) -> str:
    """从 Claude 返回的内容里抽出第一段 <svg>...</svg>。"""
    if not text:
        raise ClaudeError("返回内容为空")
    text = re.sub(r"^```(?:svg|xml|html)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    m = re.search(r"<svg[\s\S]*?</svg>", text, flags=re.IGNORECASE)
    if not m:
        raise ClaudeError(f"未找到 SVG 标签。返回片段: {text[:200]}")
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


async def generate_icon_svg(
    prompt: str,
    *,
    library: str = "lucide",
    color: str = "",
    stroke_width: float | None = None,
) -> str:
    if not CLAUDE_KEY:
        raise ClaudeError("CLAUDE_KEY 未配置")

    url = f"{CLAUDE_BASE}/v1/messages"
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 4096,
        "system": _system_prompt(library),
        "messages": [
            {
                "role": "user",
                "content": _build_user_prompt(prompt, library, color, stroke_width),
            },
        ],
    }
    headers = {
        "x-api-key": CLAUDE_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    log.info("claude POST %s model=%s library=%s", url, CLAUDE_MODEL, library)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            resp = await client.post(url, json=body, headers=headers)
        except httpx.HTTPError as e:
            log.error("claude 连接失败: %s", e)
            raise ClaudeError(f"连接失败: {e}")
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
        except Exception:
            log.error("claude 返回非 JSON: %s", resp.text[:500])
            raise ClaudeError("上游返回非 JSON")

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

    return _extract_svg(text)
