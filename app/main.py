"""FastAPI 入口。挂载路由 + 静态页面 + 启动钩子。"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import db, storage
from .config import settings
from .routes_admin import router as admin_router
from .routes_user import router as user_router

log = logging.getLogger("image2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"


def _redact(value: str | None) -> str:
    if not value:
        return "<EMPTY>"
    return f"<set, len={len(value)}>"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("启动诊断: DATABASE_URL=%s S3_ENDPOINT=%s S3_BUCKET=%s UPSTREAM_BASE=%s",
             _redact(settings.database_url),
             settings.s3_endpoint or "<EMPTY>",
             settings.s3_bucket or "<EMPTY>",
             settings.upstream_base or "<EMPTY>")
    if not settings.database_url:
        log.error("DATABASE_URL 为空。请在 Zeabur 应用的 Variables 里填上 PostgreSQL 连接串。")
        raise RuntimeError("DATABASE_URL is empty")
    if not settings.database_url.startswith(("postgres://", "postgresql://")):
        log.error("DATABASE_URL 格式不对：必须以 postgres:// 或 postgresql:// 开头。实际拿到的前缀: %r",
                  settings.database_url[:30])
        raise RuntimeError("DATABASE_URL has invalid scheme")
    log.info("启动: 初始化 DB 与对象存储")
    await db.init_pool()
    try:
        await asyncio.to_thread(storage.ensure_bucket)
    except Exception as e:
        log.error("MinIO bucket 初始化失败: %s", e)
        raise
    log.info("启动完成")
    yield
    log.info("关闭: 释放 DB 连接池")
    await db.close_pool()


app = FastAPI(title="image2", lifespan=lifespan)

app.include_router(user_router)
app.include_router(admin_router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"ok": "1"}


@app.get("/favicon.ico")
async def favicon():
    # 返回一个简洁的透明 SVG 当 favicon
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        b'<rect width="32" height="32" rx="6" fill="#6aa6ff"/>'
        b'<text x="50%" y="58%" text-anchor="middle" font-family="system-ui" font-size="18" font-weight="700" fill="#0b1020">i2</text>'
        b'</svg>'
    )
    from fastapi.responses import Response as FResp
    return FResp(content=svg, media_type="image/svg+xml")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "user.html")


@app.get("/admin")
async def admin_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "admin.html")


# 静态资源（CSS、图标等）
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
