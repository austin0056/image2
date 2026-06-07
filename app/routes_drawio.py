"""自托管 draw.io：webapp 静态资源放在 MinIO 的 drawio/ 前缀下，经此路由同源代理给浏览器。

MinIO 走 Zeabur 内网，只有 app 能连到它，所以资源同步必须从服务端跑（用 app 自己的 storage，
端点/桶自动与代理一致）。同步流程：服务端下载 drawio tarball → 解压 src/main/webapp → 全量上传 MinIO。
嵌入编辑器走 /drawio/?embed=1&proto=json（父子用 postMessage 通信）。
"""
from __future__ import annotations

import asyncio
import mimetypes
import os
import tarfile
import tempfile

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from . import storage
from .config import settings
from .deps import require_admin

router = APIRouter()

_PREFIX = "drawio/"
_VERSION = "v30.0.4"
_TARBALL_URL = f"https://codeload.github.com/jgraph/drawio/tar.gz/refs/tags/{_VERSION}"
_WEBAPP_MARKER = "/src/main/webapp/"

# 显式补全 webapp 常见类型，避免 mimetypes 猜错（尤其 js/wasm/woff2）。
_CT = {
    ".js": "application/javascript", ".mjs": "application/javascript",
    ".css": "text/css", ".html": "text/html; charset=utf-8",
    ".json": "application/json", ".map": "application/json",
    ".svg": "image/svg+xml", ".xml": "application/xml",
    ".wasm": "application/wasm",
    ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf",
    ".eot": "application/vnd.ms-fontobject",
    ".png": "image/png", ".gif": "image/gif", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8", ".md": "text/plain; charset=utf-8",
    ".appcache": "text/cache-manifest",
}


def _ctype(name: str) -> str:
    e = os.path.splitext(name)[1].lower()
    return _CT.get(e) or mimetypes.guess_type(name)[0] or "application/octet-stream"


# ===================== 静态资源代理 =====================
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
            404, "draw.io 资源未就绪：用管理员触发 POST /api/admin/drawio/sync 同步到 MinIO"
        )
    cache = "no-cache, no-store, must-revalidate" if path.endswith((".html", ".xml")) else \
            "public, max-age=31536000, immutable"
    return Response(content=body, media_type=ctype, headers={"Cache-Control": cache})


# ===================== 同步：tarball → MinIO =====================
_sync = {"running": False, "status": "idle", "done": 0, "total": 0, "error": None, "version": _VERSION}


async def _run_sync() -> None:
    tmp_path = ""
    try:
        _sync.update(running=True, status="downloading", done=0, total=0, error=None)
        fd, tmp_path = tempfile.mkstemp(suffix=".tgz")
        os.close(fd)
        # 流式下载到磁盘（~61MB），避免整包驻留内存
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
            async with client.stream("GET", _TARBALL_URL, follow_redirects=True) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    async for chunk in r.aiter_bytes(1 << 16):
                        f.write(chunk)

        _sync["status"] = "extracting"
        cli = storage._client()  # 复用一个 S3 client，避免每文件新建
        bucket = settings.s3_bucket

        with tarfile.open(tmp_path, mode="r:gz") as tf:
            members = [m for m in tf.getmembers() if m.isfile() and _WEBAPP_MARKER in m.name]
            _sync.update(total=len(members), status="uploading")
            for m in members:
                rel = m.name.split(_WEBAPP_MARKER, 1)[1].lstrip("/")
                if not rel or ".." in rel:
                    continue
                ef = tf.extractfile(m)
                if ef is None:
                    continue
                body = ef.read()
                key = _PREFIX + rel
                await asyncio.to_thread(
                    cli.put_object, Bucket=bucket, Key=key, Body=body, ContentType=_ctype(rel)
                )
                _sync["done"] += 1
        _sync.update(running=False, status="done")
    except Exception as e:  # noqa: BLE001 —— 后台任务绝不抛出，错误写进状态
        _sync.update(running=False, status="error", error=str(e)[:400])
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@router.post("/api/admin/drawio/sync", dependencies=[Depends(require_admin)])
async def drawio_sync_start() -> dict:
    if _sync["running"]:
        return {"started": False, **_sync}
    asyncio.create_task(_run_sync())
    return {"started": True, "version": _VERSION}


@router.get("/api/admin/drawio/sync/status", dependencies=[Depends(require_admin)])
async def drawio_sync_status() -> dict:
    return dict(_sync)
