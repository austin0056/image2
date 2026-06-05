"""上游调用封装：文生图与图生图。返回 PNG 字节。"""
from __future__ import annotations

import asyncio
import base64
import io
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


async def _provider_settings() -> dict[str, Any]:
    cfg = await db.get_image_provider_settings(reveal_key=True)
    if not cfg["upstream_base"]:
        raise UpstreamError("图片生成 API Base URL 未配置，请在管理后台设置")
    if not cfg["upstream_key"]:
        raise UpstreamError("图片生成 API Key 未配置，请在管理后台设置")
    if not cfg["upstream_model"]:
        raise UpstreamError("图片生成模型未配置，请在管理后台设置")
    return cfg


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


async def _post_image_with_retry(do_post, *, label: str, client: httpx.AsyncClient, expected_size: str) -> bytes:
    """统一的图片请求执行：限流/网关瞬时错误退避重试；非瞬时 4xx 去掉 response_format 降级一次。

    do_post(with_rf: bool) -> httpx.Response：with_rf=False 时不带 response_format（兼容个别中转）。
    """
    last_status, last_err = 0, "未知错误"
    for attempt in range(_MAX_IMG_RETRIES + 1):
        resp = await do_post(True)
        if resp.status_code < 400:
            return await _decode_response(client, resp.json(), expected_size)
        last_status, last_err = resp.status_code, _err_text(resp)

        if _is_transient(last_status, last_err) and attempt < _MAX_IMG_RETRIES:
            delay = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
            log.warning("%s 瞬时错误 %s（第 %d 次），%.1fs 后重试：%s", label, last_status, attempt + 1, delay, last_err[:160])
            await asyncio.sleep(delay)
            continue

        # 非瞬时错误：可能是中转不支持 response_format=b64_json，去掉再试一次
        resp2 = await do_post(False)
        if resp2.status_code < 400:
            return await _decode_response(client, resp2.json(), expected_size)
        last_status, last_err = resp2.status_code, _err_text(resp2)
        # 降级后若仍是瞬时限流且还有次数，退避后整轮再来
        if _is_transient(last_status, last_err) and attempt < _MAX_IMG_RETRIES:
            await asyncio.sleep(_RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)])
            continue
        break

    if _is_rate_limited(last_status, last_err):
        raise UpstreamError(f"{label} 上游限流，多次重试仍繁忙，请稍后再试（{last_status}）")
    raise UpstreamError(f"{label} {last_status}: {last_err}")


async def generate_image(prompt: str, size: str, quality: str = "auto") -> bytes:
    cfg = await _provider_settings()
    url = f"{cfg['upstream_base']}/images/generations"
    headers = _auth_headers(cfg["upstream_key"])
    body = {
        "model": cfg["upstream_model"],
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "n": 1,
        "response_format": "b64_json",
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        async def do_post(with_rf: bool):
            b = body if with_rf else {k: v for k, v in body.items() if k != "response_format"}
            return await client.post(url, json=b, headers=headers)
        return await _post_image_with_retry(do_post, label="generations", client=client, expected_size=size)


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
    cfg = await _provider_settings()
    url = f"{cfg['upstream_base']}/images/edits"
    headers = _auth_headers(cfg["upstream_key"])
    data = {
        "model": cfg["upstream_model"],
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "n": "1",
        "response_format": "b64_json",
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        async def do_post(with_rf: bool):
            d = data if with_rf else {k: v for k, v in data.items() if k != "response_format"}
            return await client.post(url, data=d, files=_build_edit_files(ref_pngs), headers=headers)
        return await _post_image_with_retry(do_post, label="edits", client=client, expected_size=size)
