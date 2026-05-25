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

_TIMEOUT = httpx.Timeout(connect=15.0, read=180.0, write=60.0, pool=15.0)


class ClaudeError(RuntimeError):
    pass


# ---------- 各图标库的设计语言 ----------
LIBRARY_GUIDES: dict[str, str] = {
    "lucide": (
        "Lucide (lucide.dev)：viewBox=\"0 0 24 24\"；纯线性 fill=\"none\" "
        "stroke=\"currentColor\" stroke-width=\"2\"；stroke-linecap/linejoin=\"round\"；"
        "主元素留 2px 边距；几何对称、最少元素。"
    ),
    "heroicons-outline": (
        "Heroicons Outline：viewBox=\"0 0 24 24\"；fill=\"none\" stroke=\"currentColor\" "
        "stroke-width=\"1.5\"；linecap/linejoin=\"round\"；优雅曲线，1.5px 边距。"
    ),
    "heroicons-solid": (
        "Heroicons Solid：viewBox=\"0 0 24 24\"；纯填充 fill=\"currentColor\" 无 stroke；"
        "fill-rule=\"evenodd\" clip-rule=\"evenodd\" 处理孔洞；扁平实心，1.5px 边距。"
    ),
    "phosphor": (
        "Phosphor regular：viewBox=\"0 0 256 256\"；fill=\"none\" stroke=\"currentColor\" "
        "stroke-width=\"16\"；linecap/linejoin=\"round\"；圆润友好，主元素 24-232 范围。"
    ),
    "tabler": (
        "Tabler：viewBox=\"0 0 24 24\"；fill=\"none\" stroke=\"currentColor\" "
        "stroke-width=\"2\"；linecap/linejoin=\"round\"；像素严格对齐，2px 边距。"
    ),
    "feather": (
        "Feather：viewBox=\"0 0 24 24\"；fill=\"none\" stroke=\"currentColor\" "
        "stroke-width=\"2\"；linecap/linejoin=\"round\"；极简，更多负空间。"
    ),
    "material": (
        "Material Symbols：viewBox=\"0 0 24 24\"；纯填充 fill=\"currentColor\"；"
        "几何块面，对齐 4px 网格，无 stroke。"
    ),
    "duotone": (
        "Duotone：viewBox=\"0 0 24 24\"；先画填充层 fill=\"#SECONDARY\" opacity=\"0.2\"，"
        "再画轮廓层 stroke=\"#PRIMARY\" stroke-width=\"2\" fill=\"none\"；保留线性骨架。"
    ),
}


# ---------- 审美 Checklist（写进 system prompt）----------
AESTHETIC_CHECKLIST = """
设计审美 Checklist（必须全部满足）：
1. 视觉重心：复杂或上重下轻形状，向下偏移 0.5-1 单位（光学对齐而非几何对齐）。
2. 像素对齐：stroke=2 时端点落在整数坐标，避免抗锯齿模糊。
3. 笔画一致：同图标内 stroke-width 必须完全相同，不混用粗细。
4. 圆角阶梯：rx/ry 用 {2, 4, 6, 8} 之一，整图统一一种值。
5. 安全留白：最外层留 ≥ viewBox 8% 的边距，绝不顶到边。
6. 元素精简：核心形状 ≤ 5 个，避免堆砌细节。
7. 几何优先：能用 <circle>/<rect>/<line>/<polygon> 就不用 <path>。
8. 路径简洁：单条 path 命令 ≤ 12 个，不允许冗余 M/L。
9. 对称性：默认轴对称或旋转对称，主题强需求才打破。
10. 16px 可读：关键特征在 16px 显示尺寸下仍 ≥ 4px 可识别。
"""


def _build_system_prompt(library: str, dual_color: bool) -> str:
    """构造带 CoT 推理 + 审美约束的 system prompt。"""
    guide = LIBRARY_GUIDES.get(library, "")
    color_section = (
        "颜色规则：用户给出主色 PRIMARY 和辅色 SECONDARY 时，把 stroke/主轮廓用 PRIMARY，"
        "fill/强调元素用 SECONDARY；任一为空则该位置使用 currentColor。"
        if dual_color
        else "颜色规则：用户给主色时替换 currentColor；为空则保持 currentColor。"
    )

    base = f"""你是一名顶尖的图标设计师，擅长仿照主流开源图标库的视觉语言。
请按下面的"思考-审查-输出"流程生成一个 SVG 图标。

# 必须按此结构输出（严格顺序）

[ANALYSIS]
拆解主题：核心隐喻是什么？2-3 个最强代表符号？

[CONCEPT]
列 2-3 个候选构图，给每个打分（清晰度/辨识度/原创性 各 1-10）。选最高分。

[GEOMETRY]
在指定 viewBox 上规划：主元素位置、视觉重心坐标、留白分布。

[PALETTE]
说明主色辅色如何分配到不同元素层。

[SVG]
<svg ...>
  ...完整 SVG ...
</svg>

# 严格输出要求
- 整段必须包含上述 5 个标记，且 [SVG] 只能出现一次。
- [SVG] 之后必须立刻是 <svg 开头到 </svg> 结尾，不要 markdown 围栏。
- 禁止 <script>、<foreignObject>、远程引用 (xlink:href、外部 url)。

# 当前要模仿的图标库
{guide or "由 AI 自由发挥（建议 24x24，线性优先）。"}

# 颜色
{color_section}

{AESTHETIC_CHECKLIST}

记住：先严肃思考前 4 段，再输出 [SVG]。粗暴跳过推理段会被视为失败。"""
    return base


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
    parts.append("请严格按 [ANALYSIS]→[CONCEPT]→[GEOMETRY]→[PALETTE]→[SVG] 顺序输出。")
    return "\n".join(parts)


def _extract_svg(text: str) -> str:
    """从 Claude 返回内容里抽出 [SVG] 段后的 <svg>...</svg>。
    兼容老格式（无 [SVG] 标记，直接 <svg>）。"""
    if not text:
        raise ClaudeError("返回内容为空")

    # 优先取 [SVG] 段后内容
    svg_seg = text
    m_seg = re.search(r"\[SVG\]\s*", text, flags=re.IGNORECASE)
    if m_seg:
        svg_seg = text[m_seg.end():]

    # 去 markdown 围栏
    svg_seg = re.sub(r"^```(?:svg|xml|html)?\s*", "", svg_seg.strip(), flags=re.IGNORECASE)
    svg_seg = re.sub(r"\s*```$", "", svg_seg.strip())

    m = re.search(r"<svg[\s\S]*?</svg>", svg_seg, flags=re.IGNORECASE)
    if not m:
        # 兼容：从原文找
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
    """轻量审计，返回质量警告（不阻塞）。"""
    warnings: list[str] = []
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return ["SVG 解析失败"]

    # 元素数量
    descendants = list(root.iter())
    n_paths = sum(1 for el in descendants if el.tag.endswith("path"))
    n_shapes = sum(
        1 for el in descendants
        if any(el.tag.endswith(t) for t in ("circle", "rect", "line", "polygon", "polyline", "ellipse", "path"))
    )
    if n_shapes > 8:
        warnings.append(f"形状数量较多 ({n_shapes})，建议 ≤ 5")
    if n_paths > 6:
        warnings.append(f"path 数量较多 ({n_paths})，能用基础形状的不要用 path")

    # 笔画一致性
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
    """返回 (svg, quality_warnings)。"""
    if not CLAUDE_KEY:
        raise ClaudeError("CLAUDE_KEY 未配置")

    dual = bool(color_primary or color_secondary) or library == "duotone"
    url = f"{CLAUDE_BASE}/v1/messages"
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 6144,
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

    svg = _extract_svg(text)
    warnings = _audit_svg(svg)
    if warnings:
        log.info("svg quality warnings: %s", warnings)
    return svg, warnings
