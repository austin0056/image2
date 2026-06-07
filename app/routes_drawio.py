"""自托管 draw.io：webapp 静态资源放在 MinIO 的 drawio/ 前缀下，经此路由同源代理给浏览器。

MinIO 不直接对浏览器公开，沿用 /files 的代理模式：浏览器请求 /drawio/...，这里从 MinIO 取对象回传。
嵌入编辑器走 /drawio/?embed=1&proto=json（父子用 postMessage 通信）。
资源先用 scripts/upload_drawio.py 一次性同步进 MinIO（用线上同一套 S3 凭证）。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from . import storage

router = APIRouter()

_PREFIX = "drawio/"


@router.get("/drawio")
@router.get("/drawio/")
async def drawio_index() -> Response:
    return await _serve("index.html")


@router.get("/drawio/{path:path}")
async def drawio_asset(path: str) -> Response:
    return await _serve(path or "index.html")


async def _serve(path: str) -> Response:
    path = path.lstrip("/")
    if ".." in path:
        raise HTTPException(404)
    key = _PREFIX + path
    try:
        body, ctype = await storage.fetch_object(key)
    except Exception:
        raise HTTPException(
            404, "draw.io 资源未就绪：请先运行 scripts/upload_drawio.py 把 webapp 同步到 MinIO"
        )
    # index.html / 配置随版本变，禁缓存；其余资源按版本固定，长缓存。
    if path.endswith((".html", ".xml")):
        cache = "no-cache, no-store, must-revalidate"
    else:
        cache = "public, max-age=31536000, immutable"
    return Response(content=body, media_type=ctype, headers={"Cache-Control": cache})
