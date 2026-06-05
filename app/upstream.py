"""上游调用封装：文生图与图生图。返回 PNG 字节。"""
from __future__ import annotations

import asyncio
import base64
import io
import itertools
import logging
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


async def _post_image(make_request, *, label: str, client: httpx.AsyncClient,
                      expected_size: str, sources: list[dict[str, str]]) -> bytes:
    """在来源池上轮询 + 限流故障转移地发图片请求。

    make_request(src, with_rf) -> httpx.Response。每轮按轮询起点遍历所有来源，命中即返回；
    某来源限流/瞬时错误/连不上就立刻切下一个来源；一整轮所有来源都繁忙才退避后再来一轮。
    非瞬时 4xx（内容策略/参数错误，各来源结果一致）直接报错，不浪费故障转移。
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
                return await _decode_response(client, resp.json(), expected_size)
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
                return await _decode_response(client, resp2.json(), expected_size)
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
            body = {"model": src["model"], "prompt": prompt, "size": size, "quality": quality, "n": 1}
            if with_rf:
                body["response_format"] = "b64_json"
            return await client.post(f"{src['base']}/images/generations", json=body, headers=_auth_headers(src["key"]))
        return await _post_image(make_request, label="generations", client=client, expected_size=size, sources=sources)


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
            data = {"model": src["model"], "prompt": prompt, "size": size, "quality": quality, "n": "1"}
            if with_rf:
                data["response_format"] = "b64_json"
            return await client.post(f"{src['base']}/images/edits", data=data,
                                     files=_build_edit_files(ref_pngs), headers=_auth_headers(src["key"]))
        return await _post_image(make_request, label="edits", client=client, expected_size=size, sources=sources)
