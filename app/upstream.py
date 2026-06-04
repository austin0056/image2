"""上游调用封装：文生图与图生图。返回 PNG 字节。"""
from __future__ import annotations

import base64
import io
from typing import Any

import httpx
from PIL import Image

from . import db


class UpstreamError(RuntimeError):
    pass


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


async def _decode_response(client: httpx.AsyncClient, payload: dict[str, Any]) -> bytes:
    """从上游响应里拿到图片字节。优先 b64_json，否则下载 url。"""
    if not payload.get("data"):
        raise UpstreamError(f"上游响应无 data 字段: {str(payload)[:200]}")
    item = payload["data"][0]
    b64 = item.get("b64_json")
    if b64:
        return _normalize_png(base64.b64decode(b64))
    url = item.get("url")
    if url:
        r = await client.get(url, timeout=_TIMEOUT)
        r.raise_for_status()
        return _normalize_png(r.content)
    raise UpstreamError(f"上游响应缺少图片字段: {str(item)[:200]}")


def _err_text(resp: httpx.Response) -> str:
    try:
        j = resp.json()
        if isinstance(j, dict) and "error" in j:
            return str(j["error"])
        return str(j)[:500]
    except Exception:
        return resp.text[:500]


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
        resp = await client.post(url, json=body, headers=headers)
        if resp.status_code >= 400:
            # 降级：去掉 response_format 重试一次（部分中转不支持）
            body2 = {k: v for k, v in body.items() if k != "response_format"}
            resp2 = await client.post(url, json=body2, headers=headers)
            if resp2.status_code >= 400:
                raise UpstreamError(f"generations {resp.status_code}: {_err_text(resp)}")
            return await _decode_response(client, resp2.json())
        return await _decode_response(client, resp.json())


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
        resp = await client.post(url, data=data, files=_build_edit_files(ref_pngs), headers=headers)
        if resp.status_code >= 400:
            data2 = {k: v for k, v in data.items() if k != "response_format"}
            resp2 = await client.post(url, data=data2, files=_build_edit_files(ref_pngs), headers=headers)
            if resp2.status_code >= 400:
                raise UpstreamError(f"edits {resp.status_code}: {_err_text(resp)}")
            return await _decode_response(client, resp2.json())
        return await _decode_response(client, resp.json())
