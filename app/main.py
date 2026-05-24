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
from .routes_admin import router as admin_router
from .routes_user import router as user_router

log = logging.getLogger("image2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
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


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "user.html")


@app.get("/admin")
async def admin_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "admin.html")


# 静态资源（CSS、图标等）
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
