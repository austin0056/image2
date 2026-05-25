"""OpenRouter Recraft v4.1 Pro Vector 上游封装。

走 OpenRouter Chat Completions 接口,modalities=["image"],
返回 data:image/svg+xml;base64 的真 SVG。

业界顶级矢量生成模型,质量远超 Claude few-shot。"""
from __future__ import annotations

import base64
import logging
import os
import re
from xml.etree import ElementTree as ET

import httpx

log = logging.getLogger("image2.recraft")

OR_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1").rstrip("/")
OR_KEY = os.getenv("OPENROUTER_KEY", "")
OR_MODEL = os.getenv("RECRAFT_MODEL", "recraft/recraft-v4.1-pro-vector")
OR_REFERER = os.getenv("OPENROUTER_REFERER", "https://image2.zeabur.app")
OR_TITLE = os.getenv("OPENROUTER_TITLE", "image2")

_TIMEOUT = httpx.Timeout(connect=15.0, read=120.0, write=60.0, pool=15.0)
_MAX_RETRIES = 1


class RecraftError(RuntimeError):
    pass


# Recraft 支持的官方风格 — 让用户/前端能精确选择
RECRAFT_STYLES = {
    "vector_illustration": "矢量插画(默认)",
    "vector_illustration/line_art": "线条艺术",
    "vector_illustration/line_circuit": "电路风格",
    "vector_illustration/linocut": "木版画",
    "vector_illustration/engraving": "雕刻",
    "vector_illustration/cartoon": "卡通",
    "vector_illustration/flat_2": "扁平 2.0",
    "vector_illustration/seamless": "无缝拼贴",
    "icon": "图标(简洁现代)",
    "icon/broken_line": "图标·虚线",
    "icon/colored_outline": "图标·彩色描边",
    "icon/colored_shapes": "图标·彩色块面",
    "icon/colored_shapes_gradient": "图标·渐变块面",
    "icon/doodle_fill": "图标·涂鸦填充",
    "icon/doodle_offset_fill": "图标·涂鸦偏移",
    "icon/offset_fill": "图标·偏移填充",
    "icon/outline": "图标·线性",
    "icon/outline_gradient": "图标·渐变线性",
    "icon/uneven_fill": "图标·不规则填充",
}

DEFAULT_STYLE = "icon/outline"


def _build_prompt(
    user_prompt: str,
    style: str,
    color_primary: str,
    color_secondary: str,
) -> str:
    """构造 Recraft 提示。Recraft 模型对自然语言提示理解力极强,
    不需要 CoT,直接描述意图 + 风格参数即可。"""
    parts = [user_prompt.strip()]
    if style and style != DEFAULT_STYLE:
        parts.append(f"style: {style}")
    if color_primary:
        parts.append(f"primary color {color_primary}")
    if color_secondary:
        parts.append(f"accent color {color_secondary}")
    return ", ".join(parts)


def _decode_data_url(data_url: str) -> str:
    """从 data:image/svg+xml;base64,... 解出原始 SVG 文本。"""
    if not data_url.startswith("data:image/svg"):
        raise RecraftError(f"返回非 SVG data URL: {data_url[:80]}")
    m = re.match(r"data:image/svg\+xml(?:;[^,]+)?,(.+)", data_url, flags=re.DOTALL)
    if not m:
        raise RecraftError("data URL 格式无法解析")
    payload = m.group(1)
    # base64 或 url-encoded
    if "base64" in data_url[:60]:
        try:
            return base64.b64decode(payload).decode("utf-8", errors="replace")
        except Exception as e:
            raise RecraftError(f"base64 解码失败: {e}")
    # 极少数情况是 url-encoded,简单处理
    from urllib.parse import unquote
    return unquote(payload)


def _sanitize_svg(svg: str) -> str:
    """轻量校验 + 安全处理。"""
    if not svg or "<svg" not in svg.lower():
        raise RecraftError("返回内容不含 SVG")
    # 抽 svg 段
    m = re.search(r"<svg[\s\S]*?</svg>", svg, flags=re.IGNORECASE)
    if not m:
        raise RecraftError("找不到完整 <svg>...</svg>")
    s = m.group(0)
    # 解析校验
    try:
        ET.fromstring(s)
    except ET.ParseError as e:
        raise RecraftError(f"SVG 解析失败: {e}")
    if re.search(r"<script\b", s, flags=re.IGNORECASE):
        raise RecraftError("SVG 含 <script>,已拒绝")
    if "xmlns" not in s[:200]:
        s = s.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    return s


async def generate_vector(
    prompt: str,
    *,
    style: str = DEFAULT_STYLE,
    color_primary: str = "",
    color_secondary: str = "",
) -> tuple[str, dict]:
    """调用 Recraft 生成 SVG。

    返回 (svg_text, meta)
    meta 含: cost_usd, completion_tokens, image_tokens, prompt_used
    """
    if not OR_KEY:
        raise RecraftError("OPENROUTER_KEY 未配置")
    if style not in RECRAFT_STYLES:
        log.warning("未知 style %r,降级为默认", style)
        style = DEFAULT_STYLE

    final_prompt = _build_prompt(prompt, style, color_primary, color_secondary)

    body = {
        "model": OR_MODEL,
        "messages": [{"role": "user", "content": final_prompt}],
        "modalities": ["image"],  # 关键:Recraft 只支持 image 输出
    }
    headers = {
        "Authorization": f"Bearer {OR_KEY}",
        "Content-Type": "application/json",
        # OpenRouter 推荐的标识头
        "HTTP-Referer": OR_REFERER,
        "X-Title": OR_TITLE,
    }
    url = f"{OR_BASE}/chat/completions"

    log.info(
        "recraft POST %s model=%s style=%s prompt=%r",
        url, OR_MODEL, style, final_prompt[:120],
    )

    last_err = "未知错误"
    data = None
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await client.post(url, json=body, headers=headers)
            except httpx.HTTPError as e:
                last_err = f"连接失败: {e}"
                log.warning("recraft attempt %d 连接失败: %s", attempt + 1, e)
                if attempt < _MAX_RETRIES:
                    continue
                raise RecraftError(last_err)

            if resp.status_code in (502, 503, 504):
                last_err = f"上游网关错误 {resp.status_code}"
                log.warning("recraft attempt %d %s", attempt + 1, last_err)
                if attempt < _MAX_RETRIES:
                    continue
                raise RecraftError(last_err)

            if resp.status_code >= 400:
                snippet = resp.text[:600]
                log.error("recraft %s: %s", resp.status_code, snippet)
                try:
                    j = resp.json()
                    msg = j.get("error", {}).get("message") or snippet
                except Exception:
                    msg = snippet
                raise RecraftError(f"recraft {resp.status_code}: {msg}")

            try:
                data = resp.json()
                break
            except Exception:
                log.error("recraft 返回非 JSON: %s", resp.text[:500])
                raise RecraftError("上游返回非 JSON")
        else:
            raise RecraftError(last_err)

    if data is None:
        raise RecraftError(last_err)

    # 解析返回
    choices = data.get("choices") or []
    if not choices:
        raise RecraftError("无 choices 返回")
    msg = choices[0].get("message", {})
    images = msg.get("images") or []
    if not images:
        # 检查 finish_reason / content 给出诊断
        fr = choices[0].get("finish_reason")
        content = msg.get("content")
        log.error("recraft 无图: finish=%s content=%r", fr, content)
        raise RecraftError(f"未返回图像,finish_reason={fr}")

    image_url = images[0].get("image_url", {}).get("url", "")
    if not image_url:
        raise RecraftError("image_url 字段为空")

    svg_text = _decode_data_url(image_url)
    svg_text = _sanitize_svg(svg_text)

    usage = data.get("usage") or {}
    meta = {
        "cost_usd": usage.get("cost", 0.0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "image_tokens": (usage.get("completion_tokens_details") or {}).get("image_tokens", 0),
        "prompt_used": final_prompt,
        "style": style,
    }
    log.info("recraft success: bytes=%d cost=%s", len(svg_text), meta["cost_usd"])
    return svg_text, meta
