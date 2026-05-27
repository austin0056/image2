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


def _resolve_database_url() -> str:
    """DATABASE_URL 优先；否则尝试从 PG/POSTGRES 系列组件变量拼出来。"""
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        # 兼容某些平台给的 postgresql:// 与 postgres:// 双写法
        if url.startswith("postgres://"):
            return url
        if url.startswith("postgresql://"):
            return url
        # 不带 scheme 的话也尝试加上
        if "://" not in url and "@" in url:
            return "postgres://" + url
        # 其它格式直接返回，让 asyncpg 报错给出更细信息
        return url

    # Zeabur 也常见 POSTGRES_CONNECTION_STRING / PG_URL 这类别名
    for alias in ("POSTGRES_CONNECTION_STRING", "POSTGRES_URL", "PG_URL"):
        v = os.getenv(alias, "").strip()
        if v:
            return v

    # 最后兜底：从组件拼
    host = os.getenv("POSTGRES_HOST") or os.getenv("PGHOST")
    port = os.getenv("POSTGRES_PORT") or os.getenv("PGPORT") or "5432"
    user = os.getenv("POSTGRES_USER") or os.getenv("PGUSER")
    pwd = os.getenv("POSTGRES_PASSWORD") or os.getenv("PGPASSWORD")
    db = os.getenv("POSTGRES_DATABASE") or os.getenv("POSTGRES_DB") or os.getenv("PGDATABASE")
    if host and user and pwd and db:
        from urllib.parse import quote
        return f"postgres://{quote(user)}:{quote(pwd)}@{host}:{port}/{db}"

    raise RuntimeError(
        "DATABASE_URL 未设置或为空。请在 Zeabur Variables 里把 PostgreSQL 服务的连接串填到 DATABASE_URL，"
        "或提供 POSTGRES_HOST/PORT/USER/PASSWORD/DATABASE 这些组件变量。"
    )


@dataclass(frozen=True)
class Settings:
    # 上游
    upstream_base: str
    upstream_key: str
    upstream_model: str
    price_cents: int
    price_recraft_cents: int

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

    # zpay
    zpay_pid: str
    zpay_key: str
    zpay_base: str
    public_base_url: str
    recharge_min_cents: int
    recharge_max_cents: int
    recharge_presets_yuan: tuple[int, ...]

    # 服务
    port: int


def _parse_presets(s: str) -> tuple[int, ...]:
    out: list[int] = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            v = int(p)
            if v > 0:
                out.append(v)
        except ValueError:
            pass
    return tuple(out) if out else (5, 10, 20, 50, 100, 200)


def load_settings() -> Settings:
    session_secret = _env("SESSION_SECRET") or secrets.token_urlsafe(32)
    return Settings(
        upstream_base=_env("UPSTREAM_BASE", "https://haochi.moon9.cloud/v1"),
        # UPSTREAM_KEY 现在可在管理面板配置；环境变量仅作为首次初始化/兜底默认值。
        upstream_key=_env("UPSTREAM_KEY", ""),
        upstream_model=_env("UPSTREAM_MODEL", "gpt-image-2"),
        price_cents=int(_env("PRICE_CENTS", "5")),
        price_recraft_cents=int(_env("PRICE_RECRAFT_CENTS", "300")),
        admin_password=_env("ADMIN_PASSWORD", required=True),
        session_secret=session_secret,
        database_url=_resolve_database_url(),
        s3_endpoint=_env("S3_ENDPOINT", required=True),
        s3_access_key=_env("S3_ACCESS_KEY", required=True),
        s3_secret_key=_env("S3_SECRET_KEY", required=True),
        s3_bucket=_env("S3_BUCKET", "images"),
        s3_region=_env("S3_REGION", "us-east-1"),
        zpay_pid=_env("ZPAY_PID", ""),
        zpay_key=_env("ZPAY_KEY", ""),
        zpay_base=_env("ZPAY_BASE", "https://zpayz.cn"),
        public_base_url=_env("PUBLIC_BASE_URL", "").rstrip("/"),
        recharge_min_cents=int(_env("RECHARGE_MIN_CENTS", "100")),
        recharge_max_cents=int(_env("RECHARGE_MAX_CENTS", "100000")),
        recharge_presets_yuan=_parse_presets(_env("RECHARGE_PRESETS_YUAN", "5,10,20,50,100,200")),
        port=int(_env("PORT", "8000")),
    )


settings = load_settings()
