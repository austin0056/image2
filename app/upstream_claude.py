"""Claude 中转上游封装：Anthropic 原生 /v1/messages 协议。
专注用例：生成单个 SVG 图标。"""
from __future__ import annotations

import os
import re
from xml.etree import ElementTree as ET

import httpx

CLAUDE_BASE = os.getenv("CLAUDE_BASE", "https://claude.moon9.cloud").rstrip("/")
CLAUDE_KEY = os.getenv("CLAUDE_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "kiro-claude-4.7")

_TIMEOUT = httpx.Timeout(connect=15.0, read=120.0, write=60.0, pool=15.0)


class ClaudeError(RuntimeError):
    pass


_SYSTEM_PROMPT = """你是一名专业的图标设计师。根据用户描述生成一个 SVG 图标。

严格要求：
1. 只返回一段完整的 SVG 代码，从 <svg 开头到 </svg> 结尾。
2. 绝对不要任何解释、markdown 围栏（```）、前后注释。
3. 必须包含 viewBox 属性，建议 0 0 24 24 或 0 0 64 64。
4. 不要使用 <script>、<foreignObject>、外部 url、xlink:href 远程引用。
5. 颜色用 currentColor 或 hex；线宽建议 stroke-width="2"，stroke-linecap/linejoin="round"。
6. 图形尽量保持几何对称、整洁，单图标主题，不堆元素。
7. 用户指定风格/颜色优先级最高。"""


def _build_user_prompt(prompt: str, style: str, color: str) -> str:
    parts = [f"主题：{prompt.strip()}"]
    if style and style != "auto":
        parts.append(f"风格：{style}")
    if color:
        parts.append(f"主色：{color}")
    parts.append("请直接输出 SVG，从 <svg 开始，</svg> 结束。")
    return "\n".join(parts)


def _extract_svg(text: str) -> str:
    """从 Claude 返回的内容里抽出第一段 <svg>...</svg>。"""
    if not text:
        raise ClaudeError("返回内容为空")
    # 去掉可能的 markdown 围栏
    text = re.sub(r"^```(?:svg|xml|html)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    m = re.search(r"<svg[\s\S]*?</svg>", text, flags=re.IGNORECASE)
    if not m:
        raise ClaudeError(f"未找到 SVG 标签。返回片段: {text[:200]}")
    svg = m.group(0)
    # 简单解析校验，防止恶意/损坏
    try:
        ET.fromstring(svg)
    except ET.ParseError as e:
        raise ClaudeError(f"SVG 解析失败: {e}")
    # 阻挡 script
    if re.search(r"<script\b", svg, flags=re.IGNORECASE):
        raise ClaudeError("SVG 含 <script>，已拒绝")
    return svg


async def generate_icon_svg(prompt: str, style: str = "auto", color: str = "") -> str:
    if not CLAUDE_KEY:
        raise ClaudeError("CLAUDE_KEY 未配置")

    url = f"{CLAUDE_BASE}/v1/messages"
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 4096,
        "system": _SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": _build_user_prompt(prompt, style, color)},
        ],
    }
    headers = {
        "x-api-key": CLAUDE_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(url, json=body, headers=headers)
        if resp.status_code >= 400:
            raise ClaudeError(f"messages {resp.status_code}: {resp.text[:500]}")
        data = resp.json()

    # Anthropic 原生格式：{"content": [{"type":"text","text":"..."}]}
    content = data.get("content")
    text = ""
    if isinstance(content, list):
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                text += blk.get("text", "")
    if not text:
        # 一些中转走 OpenAI 包装：{"choices":[{"message":{"content":"..."}}]}
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            text = choices[0].get("message", {}).get("content", "")

    return _extract_svg(text)
