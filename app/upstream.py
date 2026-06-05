"""上游调用封装：文生图与图生图。返回 PNG 字节。"""
from __future__ import annotations

import asyncio
import base64
import io
import itertools
import logging
import re
from typing import Any

import httpx
from PIL import Image

from . import db

log = logging.getLogger("image2.upstream")


class UpstreamError(RuntimeError):
    pass


def _verify_size(data: bytes, expected: str) -> None:
    """严格核对：实际产出图片尺寸是否等于请求尺寸（auto 跳过）。不一致仅告警，不丢弃可用图。"""
    if not expected or expected == "auto":
        return
    try:
        w, h = Image.open(io.BytesIO(data)).size
    except Exception:
        return
    actual = f"{w}x{h}"
    if actual != expected:
        log.warning("生成尺寸不一致：请求 %s，上游实际返回 %s", expected, actual)


# 复杂/大尺寸生图在上游可能要 100-200s，这里给足读超时（生图已改为后台异步任务，
# 不受 Cloudflare 100s 限制，所以可以放长）。
_TIMEOUT = httpx.Timeout(connect=15.0, read=300.0, write=60.0, pool=15.0)


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _normalize_png(data: bytes) -> bytes:
    """把任意图片字节统一转 PNG。"""
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


async def _decode_response(client: httpx.AsyncClient, payload: dict[str, Any], expected_size: str = "auto") -> bytes:
    """从上游响应里拿到图片字节。优先 b64_json，否则下载 url。并核对产出尺寸。"""
    if not payload.get("data"):
        raise UpstreamError(f"上游响应无 data 字段: {str(payload)[:200]}")
    item = payload["data"][0]
    b64 = item.get("b64_json")
    if b64:
        png = _normalize_png(base64.b64decode(b64))
        _verify_size(png, expected_size)
        return png
    url = item.get("url")
    if url:
        r = await client.get(url, timeout=_TIMEOUT)
        r.raise_for_status()
        png = _normalize_png(r.content)
        _verify_size(png, expected_size)
        return png
    raise UpstreamError(f"上游响应缺少图片字段: {str(item)[:200]}")


def _err_text(resp: httpx.Response) -> str:
    try:
        j = resp.json()
        if isinstance(j, dict) and "error" in j:
            return str(j["error"])
        return str(j)[:500]
    except Exception:
        return resp.text[:500]


# ----- Gemini 等“对话式出图”模型：走 /chat/completions，图片在返回消息里 -----
# （这类中转的 OpenAI images API 往往不支持；图片以 data:base64 / url / inline_data 形式回来。）
def _is_chat_image(model: str) -> bool:
    m = (model or "").lower()
    if "gpt-image" in m or "dall-e" in m or "dalle" in m:
        return False  # OpenAI 原生 images API
    return ("gemini" in m and "image" in m) or "flash-image" in m \
        or "nano-banana" in m or "nano_banana" in m


def _img_part(png: bytes) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


def _collect_image_urls(payload: dict[str, Any]) -> list[str]:
    """从对话式出图响应里收集所有“图片来源”（data:base64 或 http url），尽量兼容各家中转。"""
    urls: list[str] = []

    def push(u):
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())

    choices = payload.get("choices") or []
    msg = (choices[0].get("message", {}) if choices else {}) or {}

    # 1) message.images: [{image_url:{url}}] / [{url}] / ["data:..."]
    for it in (msg.get("images") or []):
        if isinstance(it, str):
            push(it)
        elif isinstance(it, dict):
            iu = it.get("image_url")
            if isinstance(iu, dict):
                push(iu.get("url"))
            elif isinstance(iu, str):
                push(iu)
            push(it.get("url") or it.get("b64_json") and ("data:image/png;base64," + it["b64_json"]))

    # 2) message.content 为分段数组：image_url / inline_data(Gemini 原生)
    content = msg.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("image_url", "output_image", "image"):
                iu = part.get("image_url")
                push(iu.get("url") if isinstance(iu, dict) else iu)
                push(part.get("url"))
            idata = part.get("inline_data") or part.get("inlineData")
            if isinstance(idata, dict) and idata.get("data"):
                mime = idata.get("mime_type") or idata.get("mimeType") or "image/png"
                push(f"data:{mime};base64,{idata['data']}")

    # 3) message.content 为字符串：markdown ![](data:/url) 或裸 data-uri / url
    if isinstance(content, str) and content:
        m = re.search(r"data:image/[\w.+-]+;base64,[A-Za-z0-9+/=\s]+", content)
        if m:
            push(re.sub(r"\s+", "", m.group(0)))
        else:
            m2 = re.search(r"!\[[^\]]*\]\((https?://[^)\s]+)\)", content) or re.search(r"https?://\S+\.(?:png|jpe?g|webp)\b", content)
            if m2:
                push(m2.group(1) if m2.lastindex else m2.group(0))

    # 4) 顶层 data[]（个别中转把图放这）
    for it in (payload.get("data") or []):
        if isinstance(it, dict):
            if it.get("b64_json"):
                push("data:image/png;base64," + it["b64_json"])
            push(it.get("url"))
    return urls


async def _decode_chat_image(client: httpx.AsyncClient, payload: dict[str, Any]) -> bytes:
    """从对话式出图响应里取出 PNG 字节（不校验尺寸——这类模型一般不接受 size 控制）。"""
    cands = _collect_image_urls(payload)
    last = ""
    for u in cands:
        try:
            if u.startswith("data:"):
                return _normalize_png(base64.b64decode(u.split(",", 1)[1]))
            if u.startswith("http"):
                r = await client.get(u, timeout=_TIMEOUT)
                r.raise_for_status()
                return _normalize_png(r.content)
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            continue
    raise UpstreamError(f"出图响应里没找到可用图片（{last}）。响应片段：{str(payload)[:300]}")


# 瞬时错误：限流(429) + 网关错误(5xx)。上游图片接口的 input-images 限流通常几十毫秒~几秒就恢复，
# 中转有时把限流包成 502，所以连同响应体里的 rate_limit 一起识别。
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_IMG_RETRIES = 4
_RETRY_BACKOFF = [0.5, 1.0, 2.0, 4.0]  # 秒


def _is_rate_limited(status: int, err: str) -> bool:
    e = (err or "").lower()
    return status == 429 or "rate_limit" in e or "rate limit" in e or "too many requests" in e


def _is_transient(status: int, err: str) -> bool:
    return status in _RETRY_STATUS or _is_rate_limited(status, err)


_rr_counter = itertools.count()  # 轮询起点（asyncio 单线程，简单计数即可）


async def _post_image(make_request, decode, *, label: str, sources: list[dict[str, str]]) -> bytes:
    """在来源池上轮询 + 限流故障转移地发图片请求。

    make_request(src, with_rf) -> httpx.Response；decode(src, resp) -> bytes（按该来源的协议解码图片）。
    每轮按轮询起点遍历所有来源，命中即返回；某来源限流/瞬时错误/连不上就立刻切下一个来源；
    一整轮所有来源都繁忙才退避后再来一轮。非瞬时 4xx（内容策略/参数错误，各来源一致）直接报错。
    """
    n = len(sources)
    start = next(_rr_counter)
    last_status, last_err = 0, "未知错误"
    for round_i in range(_MAX_IMG_RETRIES + 1):
        for i in range(n):
            src = sources[(start + i) % n]
            try:
                resp = await make_request(src, True)
            except httpx.HTTPError as e:
                last_status, last_err = 0, f"连接失败: {e}"
                continue  # 这个来源连不上 → 下一个
            if resp.status_code < 400:
                return await decode(src, resp)
            last_status, last_err = resp.status_code, _err_text(resp)
            if _is_transient(last_status, last_err):
                log.warning("%s 来源繁忙 %s，切下一个来源：%s", label, last_status, last_err[:140])
                continue  # 限流/网关错误 → 立刻切下一个来源
            # 非瞬时 4xx：去掉 response_format 降级再试一次（个别中转不支持 b64_json）
            try:
                resp2 = await make_request(src, False)
            except httpx.HTTPError as e:
                last_status, last_err = 0, f"连接失败: {e}"
                continue
            if resp2.status_code < 400:
                return await decode(src, resp2)
            s2, e2 = resp2.status_code, _err_text(resp2)
            if not _is_transient(s2, e2):
                raise UpstreamError(f"{label} {s2}: {e2}")  # 内容策略等，各来源一致 → 直接报错
            last_status, last_err = s2, e2
        # 一整轮所有来源都繁忙 → 退避后再来一轮
        if round_i < _MAX_IMG_RETRIES:
            delay = _RETRY_BACKOFF[min(round_i, len(_RETRY_BACKOFF) - 1)]
            log.warning("%s 全部 %d 个来源本轮都繁忙，%.1fs 后重试", label, n, delay)
            await asyncio.sleep(delay)
    if _is_rate_limited(last_status, last_err):
        raise UpstreamError(f"{label} 所有来源都在限流，请稍后再试（共 {n} 个来源）")
    raise UpstreamError(f"{label} {last_status}: {last_err}")


async def generate_image(prompt: str, size: str, quality: str = "auto") -> bytes:
    sources = await db.get_image_pool()
    if not sources:
        raise UpstreamError("图片生成未配置任何可用来源，请在管理后台设置")
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        async def make_request(src: dict[str, str], with_rf: bool):
            if _is_chat_image(src["model"]):  # Gemini 等对话式出图：/chat/completions
                body = {"model": src["model"], "messages": [{"role": "user", "content": prompt}]}
                return await client.post(f"{src['base']}/chat/completions", json=body, headers=_auth_headers(src["key"]))
            body = {"model": src["model"], "prompt": prompt, "size": size, "quality": quality, "n": 1}
            if with_rf:
                body["response_format"] = "b64_json"
            return await client.post(f"{src['base']}/images/generations", json=body, headers=_auth_headers(src["key"]))

        async def decode(src: dict[str, str], resp: httpx.Response):
            if _is_chat_image(src["model"]):
                return await _decode_chat_image(client, resp.json())
            return await _decode_response(client, resp.json(), size)

        return await _post_image(make_request, decode, label="generations", sources=sources)


def _build_edit_files(ref_pngs: list[bytes]) -> list[tuple[str, tuple[str, bytes, str]]]:
    """构造 multipart 文件列表。单图用字段名 image，多图用 image[]（OpenAI 约定）。"""
    field = "image" if len(ref_pngs) == 1 else "image[]"
    return [
        (field, (f"ref{i}.png", png, "image/png"))
        for i, png in enumerate(ref_pngs)
    ]


async def edit_image(prompt: str, size: str, ref_pngs: list[bytes], quality: str = "auto") -> bytes:
    if not ref_pngs:
        raise UpstreamError("缺少参考图")
    sources = await db.get_image_pool()
    if not sources:
        raise UpstreamError("图片生成未配置任何可用来源，请在管理后台设置")
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        async def make_request(src: dict[str, str], with_rf: bool):
            if _is_chat_image(src["model"]):  # Gemini 等：参考图作为多模态消息传入
                content = [{"type": "text", "text": prompt}] + [_img_part(p) for p in ref_pngs]
                body = {"model": src["model"], "messages": [{"role": "user", "content": content}]}
                return await client.post(f"{src['base']}/chat/completions", json=body, headers=_auth_headers(src["key"]))
            data = {"model": src["model"], "prompt": prompt, "size": size, "quality": quality, "n": "1"}
            if with_rf:
                data["response_format"] = "b64_json"
            return await client.post(f"{src['base']}/images/edits", data=data,
                                     files=_build_edit_files(ref_pngs), headers=_auth_headers(src["key"]))

        async def decode(src: dict[str, str], resp: httpx.Response):
            if _is_chat_image(src["model"]):
                return await _decode_chat_image(client, resp.json())
            return await _decode_response(client, resp.json(), size)

        return await _post_image(make_request, decode, label="edits", sources=sources)
