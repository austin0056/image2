"""生图前的「提示词优化」：用 chart(gpt-5.5) 通道，把用户的简短创意扩写成更利于成图的提示词。

设计要点：
- 复用 upstream_chart.complete_text（同一条 gpt-5.5 通道，按 api_style 走 Responses/Chat）。
- enhance_image_prompt() **绝不抛错**：任何失败（未配置/超时/空返回）都回退到用户原始提示词，
  不阻断生图。
- 文生图与图生图（带参考图）用不同口吻：前者扩写为画面描述，后者保持为"修改指令"，避免凭空换场景。
"""
from __future__ import annotations

import asyncio
import logging
import re

from . import upstream_chart

log = logging.getLogger("image2.prompt")

# 输出上限：优化后的提示词最长字符数（避免极端长文喂给上游）
_MAX_CHARS = 1600
# LLM 调用超时（秒）：优化只是锦上添花，超时即回退原文，不拖慢生图
_TIMEOUT = 30.0


def _system_prompt(*, has_ref: bool, aspect: str, tier_label: str) -> str:
    common = (
        "你是资深「AI 绘画提示词工程师」。把用户给的（往往简短、口语化的）创意，"
        "改写成一段高质量、可直接喂给文生图模型的提示词。\n"
        "规则：\n"
        "1. 忠实保留用户的核心主体、动作、场景与意图，绝不替换或偏离主题，只做丰富化。\n"
        "2. 在不违背原意的前提下，补全有助于成图的视觉要素：主体细节、构图与视角、"
        "光线与时间、色彩与氛围、材质质感、背景环境、艺术风格/媒介（写实摄影可加镜头、景深等）。\n"
        "3. 用户若指定画面中要出现的文字（如标语、店名、数字），原样保留该文字、不翻译，并用引号标出。\n"
        "4. 与用户输入使用同一种语言输出（中文输入→中文提示词）。\n"
        "5. 不要臆造用户没提到的具体身份信息（真实人名/品牌/版权角色）。\n"
        f"6. 目标画面比例约为 {aspect}、清晰度档位 {tier_label}；可在措辞上呼应构图，"
        "但不要在文中写出任何像素尺寸数字。\n"
        "7. 只输出最终提示词本身：不要解释、不要前缀（如“提示词：”）、不要用引号包裹整段、"
        "不要分点列表、不要 markdown 代码块，整体控制在 120 个词/字以内。"
    )
    if has_ref:
        common += (
            "\n\n注意：当前是「图生图 / 改图」——用户输入是对参考图的修改指令。"
            "请把它整理成清晰、忠实的修改指令，只澄清和细化要改动的部分，"
            "不要凭空编造全新场景，也不要描述与改动无关的内容。"
        )
    return common


_LABEL_RE = re.compile(
    r"^\s*(?:优化后(?:的)?提示词|提示词|改写后|最终提示词|prompt|enhanced prompt|final prompt)\s*[:：]\s*",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    """把模型输出整理成可直接使用的单段提示词。"""
    s = (text or "").strip()
    if not s:
        return ""
    # 去掉 ```/```lang 代码块围栏
    if s.startswith("```"):
        s = re.sub(r"^```[^\n]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s).strip()
    # 去掉前缀标签（提示词：/ Prompt: 等）
    s = _LABEL_RE.sub("", s).strip()
    # 去掉整体包裹的引号
    for lq, rq in (('"', '"'), ("'", "'"), ("“", "”"), ("「", "」"), ("『", "』")):
        if len(s) >= 2 and s[0] == lq and s[-1] == rq:
            s = s[1:-1].strip()
            break
    # 合并空白为单段（多数上游 prompt 字段更适合单行）
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > _MAX_CHARS:
        s = s[:_MAX_CHARS].rstrip()
    return s


async def enhance_image_prompt(
    user_prompt: str,
    *,
    has_ref: bool = False,
    aspect: str = "1:1",
    tier_label: str = "1K",
    timeout: float = _TIMEOUT,
) -> tuple[str, bool]:
    """返回 (最终提示词, 是否已优化)。任何失败都回退到原始提示词，绝不抛错、绝不阻断生图。"""
    orig = (user_prompt or "").strip()
    if not orig:
        return orig, False
    try:
        system = _system_prompt(has_ref=has_ref, aspect=aspect, tier_label=tier_label)
        text = await asyncio.wait_for(
            upstream_chart.complete_text(system, [{"role": "user", "content": orig}], max_tokens=800),
            timeout=timeout,
        )
        cleaned = _clean(text)
        if cleaned and cleaned.lower() != orig.lower():
            return cleaned, True
        return orig, False
    except Exception as e:  # noqa: BLE001 — 优化失败必须静默回退
        log.warning("提示词优化失败，回退原始提示词：%s", e)
        return orig, False
