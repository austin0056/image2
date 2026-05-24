"""集中读取环境变量。所有模块只从这里取配置。"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # 生产环境（Zeabur）没有 .env 也无所谓，环境变量直接注入
    pass


def _env(key: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.getenv(key, default)
    if required and not val:
        raise RuntimeError(f"环境变量 {key} 未设置")
    return val or ""


@dataclass(frozen=True)
class Settings:
    # 上游
    upstream_base: str
    upstream_key: str
    upstream_model: str
    price_cents: int

    # 鉴权
    admin_password: str
    session_secret: str

    # 数据库
    database_url: str

    # 对象存储
    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str
    s3_bucket: str
    s3_region: str

    # 服务
    port: int


def load_settings() -> Settings:
    session_secret = _env("SESSION_SECRET") or secrets.token_urlsafe(32)
    return Settings(
        upstream_base=_env("UPSTREAM_BASE", "https://haochi.moon9.cloud/v1"),
        upstream_key=_env("UPSTREAM_KEY", required=True),
        upstream_model=_env("UPSTREAM_MODEL", "gpt-image-2"),
        price_cents=int(_env("PRICE_CENTS", "5")),
        admin_password=_env("ADMIN_PASSWORD", required=True),
        session_secret=session_secret,
        database_url=_env("DATABASE_URL", required=True),
        s3_endpoint=_env("S3_ENDPOINT", required=True),
        s3_access_key=_env("S3_ACCESS_KEY", required=True),
        s3_secret_key=_env("S3_SECRET_KEY", required=True),
        s3_bucket=_env("S3_BUCKET", "images"),
        s3_region=_env("S3_REGION", "us-east-1"),
        port=int(_env("PORT", "8000")),
    )


settings = load_settings()
