"""PostgreSQL 接入层。提供连接池、建表、用户与生成记录的增删改查。"""
from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from typing import Any

import asyncpg

from .config import settings

_pool: asyncpg.Pool | None = None


async def init_pool() -> None:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=2,
            max_size=10,
            command_timeout=15,
            max_inactive_connection_lifetime=300,  # 5 分钟闲连接重建
            # 如果 PostgreSQL 上下走了 PgBouncer 类代理，该参数避免 statement
            # cache 不一致导致的参数绑定错误。启用后性能有轻微损失，但场景
            # 安全。Zeabur PostgreSQL 不走代理，重启不会丢，保留默认 None 即可。
        )
        await _init_schema()


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("DB pool 未初始化")
    return _pool


async def _init_schema() -> None:
    async with pool().acquire() as con:
        await con.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                access_key TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                balance_cents INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS generations (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                prompt TEXT NOT NULL,
                size TEXT NOT NULL,
                has_ref BOOLEAN NOT NULL DEFAULT false,
                ref_key TEXT,
                result_key TEXT,
                cost_cents INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                error TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );

            CREATE INDEX IF NOT EXISTS idx_gen_user_created
                ON generations(user_id, created_at DESC);
            """
        )
        # 渐进式加列（幂等）
        await con.execute(
            "ALTER TABLE generations ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'image'"
        )
        await con.execute(
            "ALTER TABLE generations ADD COLUMN IF NOT EXISTS result_svg TEXT"
        )
        # CAD 产物：{"step": key, "stl": key, "glb": key}
        await con.execute(
            "ALTER TABLE generations ADD COLUMN IF NOT EXISTS result_files JSONB"
        )
        # 多张参考图的对象 key 列表（生图）。ref_key 仍保留为第一张，兼容旧缩略图逻辑。
        await con.execute(
            "ALTER TABLE generations ADD COLUMN IF NOT EXISTS ref_keys JSONB"
        )
        # payments 表
        await con.execute(
            """
            CREATE TABLE IF NOT EXISTS payments (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                out_trade_no VARCHAR(40) UNIQUE NOT NULL,
                trade_no VARCHAR(64),
                amount_cents INTEGER NOT NULL,
                pay_type VARCHAR(16) NOT NULL DEFAULT 'alipay',
                status VARCHAR(16) NOT NULL DEFAULT 'pending',
                notify_raw TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                paid_at TIMESTAMPTZ
            );
            CREATE INDEX IF NOT EXISTS idx_pay_user_created
                ON payments(user_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_pay_status
                ON payments(status);
            """
        )
        # 系统设置表：用于运行时配置图片生成上游，环境变量只作为首次初始化默认值。
        await con.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL DEFAULT '',
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        defaults = {
            "image_provider.base": settings.upstream_base,
            "image_provider.key": settings.upstream_key,
            "image_provider.model": settings.upstream_model,
            "image_provider.price_cents": str(settings.price_cents),
        }
        # 其它上游（CAD/矢量Claude、Recraft、公式图表）也用 app_settings 存，环境变量仅首次初始化
        for name, d in _PROVIDER_DEFAULTS().items():
            defaults[f"provider.{name}.base"] = d["base"]
            defaults[f"provider.{name}.key"] = d["key"]
            defaults[f"provider.{name}.model"] = d["model"]
            defaults[f"provider.{name}.api_style"] = d.get("api_style", "auto")
        for key, value in defaults.items():
            await con.execute(
                """
                INSERT INTO app_settings(key, value)
                VALUES($1, $2)
                ON CONFLICT (key) DO NOTHING
                """,
                key, value or "",
            )


# ----- 系统设置 -----

_IMAGE_PROVIDER_KEYS = {
    "base": "image_provider.base",
    "key": "image_provider.key",
    "model": "image_provider.model",
    "price_cents": "image_provider.price_cents",
    "price_2k_cents": "image_provider.price_2k_cents",
    "price_4k_cents": "image_provider.price_4k_cents",
}

# 分辨率档位（1K/2K/4K）：用户可选，按档分开计价；上游原生出对应尺寸。
IMAGE_TIERS = ("1k", "2k", "4k")
_TIER_BASE = {"1k": 1024, "2k": 2048, "4k": 4096}
_TIER_ASPECT = {"1:1": (1, 1), "3:2": (3, 2), "2:3": (2, 3)}


def normalize_tier(tier: str | None) -> str:
    t = (tier or "1k").lower()
    return t if t in IMAGE_TIERS else "1k"


def tier_size(tier: str | None, aspect: str | None = "1:1") -> str:
    """档位 + 比例 → 上游尺寸字符串（如 2048x2048 / 3072x2048 / 2048x3072）。"""
    base = _TIER_BASE.get(normalize_tier(tier), 1024)
    aw, ah = _TIER_ASPECT.get(aspect or "1:1", (1, 1))
    if aw == ah:
        return f"{base}x{base}"
    if aw > ah:  # 3:2 横
        return f"{base + base // 2}x{base}"
    return f"{base}x{base + base // 2}"  # 2:3 竖


def _coerce_cents(v: Any, fallback: int = 0) -> int:
    try:
        return max(0, int(v))
    except (TypeError, ValueError):
        return fallback


def _tier_prices_of(m: dict[str, Any], *, base_key: str = "price_cents") -> dict[str, int]:
    """从模型/提供商配置取三档价；2K/4K 未设则回退到 1K 价（兼容旧配置）。"""
    p1 = _coerce_cents(m.get(base_key, 0))
    raw2, raw4 = m.get("price_2k_cents"), m.get("price_4k_cents")
    p2 = _coerce_cents(raw2, p1) if raw2 not in (None, "") else p1
    p4 = _coerce_cents(raw4, p1) if raw4 not in (None, "") else p1
    return {"1k": p1, "2k": p2, "4k": p4}


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "••••"
    return value[:4] + "…" + value[-4:]


async def get_image_provider_settings(*, reveal_key: bool = True) -> dict[str, Any]:
    """返回图片生成上游配置。reveal_key=False 时仅返回脱敏 key。"""
    async with pool().acquire() as con:
        rows = await con.fetch(
            "SELECT key, value FROM app_settings WHERE key = ANY($1::text[])",
            list(_IMAGE_PROVIDER_KEYS.values()),
        )
    values = {r["key"]: r["value"] for r in rows}
    raw_key = values.get(_IMAGE_PROVIDER_KEYS["key"], settings.upstream_key) or ""
    try:
        price_cents = int(values.get(_IMAGE_PROVIDER_KEYS["price_cents"], str(settings.price_cents)) or settings.price_cents)
    except (TypeError, ValueError):
        price_cents = settings.price_cents
    price_cents = max(0, price_cents)

    def _opt_cents(name: str, fallback: int) -> int:
        v = values.get(_IMAGE_PROVIDER_KEYS[name])
        return _coerce_cents(v, fallback) if v not in (None, "") else fallback

    price_2k_cents = _opt_cents("price_2k_cents", price_cents)
    price_4k_cents = _opt_cents("price_4k_cents", price_cents)
    return {
        "upstream_base": (values.get(_IMAGE_PROVIDER_KEYS["base"], settings.upstream_base) or settings.upstream_base).rstrip("/"),
        "upstream_key": raw_key if reveal_key else "",
        "upstream_key_set": bool(raw_key),
        "upstream_key_preview": _mask_secret(raw_key),
        "upstream_model": values.get(_IMAGE_PROVIDER_KEYS["model"], settings.upstream_model) or settings.upstream_model,
        "price_cents": price_cents,
        "price_2k_cents": price_2k_cents,
        "price_4k_cents": price_4k_cents,
        "prices": {"1k": price_cents, "2k": price_2k_cents, "4k": price_4k_cents},
    }


async def update_image_provider_settings(
    *,
    upstream_base: str,
    upstream_model: str,
    price_cents: int,
    price_2k_cents: int | None = None,
    price_4k_cents: int | None = None,
    upstream_key: str | None = None,
) -> dict[str, Any]:
    base_price = max(0, int(price_cents))
    updates = {
        _IMAGE_PROVIDER_KEYS["base"]: upstream_base.rstrip("/"),
        _IMAGE_PROVIDER_KEYS["model"]: upstream_model,
        _IMAGE_PROVIDER_KEYS["price_cents"]: str(base_price),
        _IMAGE_PROVIDER_KEYS["price_2k_cents"]: str(_coerce_cents(price_2k_cents, base_price) if price_2k_cents is not None else base_price),
        _IMAGE_PROVIDER_KEYS["price_4k_cents"]: str(_coerce_cents(price_4k_cents, base_price) if price_4k_cents is not None else base_price),
    }
    if upstream_key is not None:
        updates[_IMAGE_PROVIDER_KEYS["key"]] = upstream_key
    async with pool().acquire() as con:
        async with con.transaction():
            for key, value in updates.items():
                await con.execute(
                    """
                    INSERT INTO app_settings(key, value, updated_at)
                    VALUES($1, $2, now())
                    ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = now()
                    """,
                    key, value or "",
                )
    return await get_image_provider_settings(reveal_key=False)


# ----- 图片生成 · 额外来源池（负载均衡 / 抗限流，提高并发） -----
# 存成单条 JSON：image_provider.extra_sources = [{id, base, model, key, enabled}, ...]
# 生效来源池 = 主来源 + 启用的额外来源；upstream 在池上轮询 + 限流故障转移。

_EXTRA_SOURCES_KEY = "image_provider.extra_sources"


async def _read_extra_sources_raw() -> list[dict[str, Any]]:
    async with pool().acquire() as con:
        row = await con.fetchrow("SELECT value FROM app_settings WHERE key=$1", _EXTRA_SOURCES_KEY)
    raw = row["value"] if row else ""
    if not raw:
        return []
    try:
        arr = json.loads(raw)
        return [s for s in arr if isinstance(s, dict)] if isinstance(arr, list) else []
    except (ValueError, TypeError):
        return []


async def _write_extra_sources_raw(arr: list[dict[str, Any]]) -> None:
    async with pool().acquire() as con:
        await con.execute(
            """
            INSERT INTO app_settings(key, value, updated_at) VALUES($1, $2, now())
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
            """,
            _EXTRA_SOURCES_KEY, json.dumps(arr, ensure_ascii=False),
        )


def _shape_source(s: dict[str, Any], *, reveal_key: bool) -> dict[str, Any]:
    k = (s.get("key") or "")
    return {
        "id": s.get("id", ""),
        "base": (s.get("base") or "").rstrip("/"),
        "model": s.get("model") or "",
        "enabled": bool(s.get("enabled", True)),
        "key": k if reveal_key else "",
        "key_set": bool(k),
        "key_preview": _mask_secret(k),
    }


async def get_image_extra_sources(*, reveal_key: bool = False) -> list[dict[str, Any]]:
    return [_shape_source(s, reveal_key=reveal_key) for s in await _read_extra_sources_raw()]


async def add_image_extra_source(*, base: str, model: str, key: str) -> list[dict[str, Any]]:
    arr = await _read_extra_sources_raw()
    arr.append({
        "id": secrets.token_hex(6),
        "base": base.rstrip("/"), "model": model, "key": key, "enabled": True,
    })
    await _write_extra_sources_raw(arr)
    return await get_image_extra_sources(reveal_key=False)


async def update_image_extra_source(sid: str, *, enabled: bool | None = None,
                                    base: str | None = None, model: str | None = None,
                                    key: str | None = None) -> list[dict[str, Any]]:
    arr = await _read_extra_sources_raw()
    for s in arr:
        if s.get("id") == sid:
            if enabled is not None:
                s["enabled"] = bool(enabled)
            if base is not None:
                s["base"] = base.rstrip("/")
            if model is not None:
                s["model"] = model
            if key:  # 空 key = 保留原有
                s["key"] = key
            break
    await _write_extra_sources_raw(arr)
    return await get_image_extra_sources(reveal_key=False)


async def delete_image_extra_source(sid: str) -> list[dict[str, Any]]:
    arr = [s for s in await _read_extra_sources_raw() if s.get("id") != sid]
    await _write_extra_sources_raw(arr)
    return await get_image_extra_sources(reveal_key=False)


async def get_image_pool() -> list[dict[str, str]]:
    """可用来源池（主来源 + 启用且配置完整的额外来源），含真实 key，供 upstream 使用。"""
    out: list[dict[str, str]] = []
    primary = await get_image_provider_settings(reveal_key=True)
    if primary["upstream_base"] and primary["upstream_key"] and primary["upstream_model"]:
        out.append({"base": primary["upstream_base"], "key": primary["upstream_key"], "model": primary["upstream_model"]})
    for s in await get_image_extra_sources(reveal_key=True):
        if s["enabled"] and s["base"] and s["key"] and s["model"]:
            out.append({"base": s["base"], "key": s["key"], "model": s["model"]})
    return out


# ----- 用户可切换的图片模型（如 GPT-Image-2 / Banana Pro 等） -----
# 默认模型 = 主图片提供商(image_provider.*)，含其抗限流来源池；额外可选模型存
# image_provider.models（JSON: [{id,label,base,key,model,price_cents,enabled}]）。

_IMAGE_MODELS_KEY = "image_provider.models"
_DEFAULT_LABELS = (("gpt-image", "GPT-Image-2"), ("dall-e", "DALL·E"),
                   ("gemini", "Gemini"), ("flux", "FLUX"), ("nano-banana", "Nano Banana"))


def _default_model_label(model: str) -> str:
    m = (model or "").lower()
    for k, lbl in _DEFAULT_LABELS:
        if k in m:
            return lbl
    return model or "默认模型"


async def _read_image_models_raw() -> list[dict[str, Any]]:
    async with pool().acquire() as con:
        row = await con.fetchrow("SELECT value FROM app_settings WHERE key=$1", _IMAGE_MODELS_KEY)
    raw = row["value"] if row else ""
    if not raw:
        return []
    try:
        arr = json.loads(raw)
        return [m for m in arr if isinstance(m, dict)] if isinstance(arr, list) else []
    except (ValueError, TypeError):
        return []


async def _write_image_models_raw(arr: list[dict[str, Any]]) -> None:
    async with pool().acquire() as con:
        await con.execute(
            """INSERT INTO app_settings(key, value, updated_at) VALUES($1, $2, now())
               ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()""",
            _IMAGE_MODELS_KEY, json.dumps(arr, ensure_ascii=False),
        )


def _price_of(m: dict[str, Any]) -> int:
    return _coerce_cents(m.get("price_cents", 0))


def _shape_image_model(m: dict[str, Any], *, reveal_key: bool) -> dict[str, Any]:
    k = m.get("key") or ""
    prices = _tier_prices_of(m)
    return {
        "id": m.get("id", ""), "label": m.get("label") or (m.get("model") or ""),
        "base": (m.get("base") or "").rstrip("/"), "model": m.get("model") or "",
        "price_cents": prices["1k"], "price_2k_cents": prices["2k"], "price_4k_cents": prices["4k"],
        "prices": prices, "enabled": bool(m.get("enabled", True)), "is_default": False,
        "key": k if reveal_key else "", "key_set": bool(k), "key_preview": _mask_secret(k),
    }


async def get_image_models(*, reveal_key: bool = False, only_enabled: bool = False) -> list[dict[str, Any]]:
    """用户可选图片模型：默认模型(主提供商) + 额外模型。"""
    out: list[dict[str, Any]] = []
    main = await get_image_provider_settings(reveal_key=reveal_key)
    if main["upstream_base"] and main["upstream_model"]:
        out.append({
            "id": "default", "label": _default_model_label(main["upstream_model"]),
            "base": main["upstream_base"], "model": main["upstream_model"],
            "price_cents": main["price_cents"], "price_2k_cents": main["price_2k_cents"],
            "price_4k_cents": main["price_4k_cents"], "prices": main["prices"],
            "enabled": True, "is_default": True,
            "key": main["upstream_key"] if reveal_key else "",
            "key_set": main["upstream_key_set"], "key_preview": main["upstream_key_preview"],
        })
    for m in await _read_image_models_raw():
        sm = _shape_image_model(m, reveal_key=reveal_key)
        if only_enabled and not sm["enabled"]:
            continue
        out.append(sm)
    return out


async def add_image_model(*, label: str, base: str, model: str, key: str, price_cents: int,
                          price_2k_cents: int | None = None, price_4k_cents: int | None = None) -> list[dict[str, Any]]:
    arr = await _read_image_models_raw()
    base_price = max(0, int(price_cents))
    arr.append({"id": secrets.token_hex(6), "label": label, "base": base.rstrip("/"),
                "model": model, "key": key, "price_cents": base_price,
                "price_2k_cents": _coerce_cents(price_2k_cents, base_price) if price_2k_cents is not None else base_price,
                "price_4k_cents": _coerce_cents(price_4k_cents, base_price) if price_4k_cents is not None else base_price,
                "enabled": True})
    await _write_image_models_raw(arr)
    return await get_image_models(reveal_key=False)


async def update_image_model(mid: str, *, label: str | None = None, base: str | None = None,
                             model: str | None = None, key: str | None = None,
                             price_cents: int | None = None, price_2k_cents: int | None = None,
                             price_4k_cents: int | None = None, enabled: bool | None = None) -> list[dict[str, Any]]:
    arr = await _read_image_models_raw()
    for m in arr:
        if m.get("id") == mid:
            if label is not None:
                m["label"] = label
            if base is not None:
                m["base"] = base.rstrip("/")
            if model is not None:
                m["model"] = model
            if key:
                m["key"] = key
            if price_cents is not None:
                m["price_cents"] = max(0, int(price_cents))
            if price_2k_cents is not None:
                m["price_2k_cents"] = max(0, int(price_2k_cents))
            if price_4k_cents is not None:
                m["price_4k_cents"] = max(0, int(price_4k_cents))
            if enabled is not None:
                m["enabled"] = bool(enabled)
            break
    await _write_image_models_raw(arr)
    return await get_image_models(reveal_key=False)


async def delete_image_model(mid: str) -> list[dict[str, Any]]:
    arr = [m for m in await _read_image_models_raw() if m.get("id") != mid]
    await _write_image_models_raw(arr)
    return await get_image_models(reveal_key=False)


async def resolve_image_model(model_id: str | None, tier: str | None = "1k") -> dict[str, Any] | None:
    """解析所选模型 + 分辨率档位 → {id,label,tier,price_cents(该档价),sources(含真实 key)}；不可用返回 None。"""
    t = normalize_tier(tier)
    if not model_id or model_id == "default":
        main = await get_image_provider_settings(reveal_key=True)
        sources = await get_image_pool()
        if not sources:
            return None
        return {"id": "default", "label": _default_model_label(main["upstream_model"]),
                "tier": t, "price_cents": main["prices"][t], "sources": sources}
    for m in await _read_image_models_raw():
        if m.get("id") == model_id and bool(m.get("enabled", True)):
            base, key, model = (m.get("base") or "").rstrip("/"), m.get("key") or "", m.get("model") or ""
            if not (base and key and model):
                return None
            return {"id": model_id, "label": m.get("label") or model, "tier": t,
                    "price_cents": _tier_prices_of(m)[t],
                    "sources": [{"base": base, "key": key, "model": model}]}
    return None


# ----- 通用上游提供商设置（claude / recraft / chart） -----

def _PROVIDER_DEFAULTS() -> dict[str, dict[str, str]]:
    # api_style：chart 走 OpenAI 兼容文本接口，可选 auto/chat/responses。
    # auto = 按模型名自动判别（gpt-5.x / o系列 → Responses API，否则 Chat Completions）。
    # claude/recraft 不读取该字段（claude 用 Messages API，recraft 是图像接口），留默认即可。
    return {
        "claude": {"base": settings.claude_base, "key": settings.claude_key, "model": settings.claude_model, "api_style": "auto"},
        "recraft": {"base": settings.recraft_base, "key": settings.recraft_key, "model": settings.recraft_model, "api_style": "auto"},
        "chart": {"base": settings.chart_base, "key": settings.chart_key, "model": settings.chart_model, "api_style": "auto"},
    }

PROVIDER_NAMES = ("claude", "recraft", "chart")


async def get_provider(name: str, *, reveal_key: bool = True) -> dict[str, Any]:
    """返回某上游的 base/key/model。reveal_key=False 时仅脱敏。"""
    if name not in PROVIDER_NAMES:
        raise ValueError(f"未知 provider: {name}")
    d = _PROVIDER_DEFAULTS()[name]
    keys = {k: f"provider.{name}.{k}" for k in ("base", "key", "model", "api_style")}
    async with pool().acquire() as con:
        rows = await con.fetch(
            "SELECT key, value FROM app_settings WHERE key = ANY($1::text[])",
            list(keys.values()),
        )
    values = {r["key"]: r["value"] for r in rows}
    raw_key = values.get(keys["key"], d["key"]) or ""
    return {
        "base": (values.get(keys["base"]) or d["base"] or "").rstrip("/"),
        "key": raw_key if reveal_key else "",
        "key_set": bool(raw_key),
        "key_preview": _mask_secret(raw_key),
        "model": values.get(keys["model"]) or d["model"] or "",
        "api_style": (values.get(keys["api_style"]) or d.get("api_style") or "auto"),
    }


async def update_provider(name: str, *, base: str, model: str, key: str | None = None, api_style: str | None = None) -> dict[str, Any]:
    if name not in PROVIDER_NAMES:
        raise ValueError(f"未知 provider: {name}")
    updates = {
        f"provider.{name}.base": base.rstrip("/"),
        f"provider.{name}.model": model,
    }
    if api_style is not None:
        updates[f"provider.{name}.api_style"] = api_style if api_style in ("auto", "chat", "responses") else "auto"
    if key is not None:
        updates[f"provider.{name}.key"] = key
    async with pool().acquire() as con:
        async with con.transaction():
            for k, v in updates.items():
                await con.execute(
                    """
                    INSERT INTO app_settings(key, value, updated_at)
                    VALUES($1, $2, now())
                    ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = now()
                    """,
                    k, v or "",
                )
    return await get_provider(name, reveal_key=False)


# ----- 用户 -----

def _row_to_dict(row: asyncpg.Record | None) -> dict[str, Any] | None:
    return dict(row) if row else None


async def get_user_by_key(access_key: str) -> dict[str, Any] | None:
    async with pool().acquire() as con:
        row = await con.fetchrow(
            "SELECT id, access_key, name, balance_cents, created_at FROM users WHERE access_key=$1",
            access_key,
        )
    return _row_to_dict(row)


async def list_users() -> list[dict[str, Any]]:
    async with pool().acquire() as con:
        rows = await con.fetch(
            """
            SELECT u.id, u.access_key, u.name, u.balance_cents, u.created_at,
                   COALESCE(c.cnt, 0) AS gen_count
            FROM users u
            LEFT JOIN (
                SELECT user_id, COUNT(*) AS cnt
                FROM generations
                WHERE status='success'
                GROUP BY user_id
            ) c ON c.user_id = u.id
            ORDER BY u.id DESC
            """
        )
    return [dict(r) for r in rows]


async def create_user(name: str) -> dict[str, Any]:
    key = "ak_" + secrets.token_urlsafe(24)
    async with pool().acquire() as con:
        row = await con.fetchrow(
            "INSERT INTO users(access_key, name) VALUES($1, $2) RETURNING id, access_key, name, balance_cents, created_at",
            key, name,
        )
    return dict(row)


async def update_user_name(user_id: int, name: str) -> None:
    async with pool().acquire() as con:
        await con.execute("UPDATE users SET name=$1 WHERE id=$2", name, user_id)


async def topup_user(user_id: int, cents: int) -> int:
    async with pool().acquire() as con:
        row = await con.fetchrow(
            "UPDATE users SET balance_cents = balance_cents + $1 WHERE id=$2 RETURNING balance_cents",
            cents, user_id,
        )
    return int(row["balance_cents"]) if row else 0


async def delete_user(user_id: int) -> None:
    async with pool().acquire() as con:
        await con.execute("DELETE FROM users WHERE id=$1", user_id)


# ----- 生成记录 -----

async def try_charge_and_create(
    user_id: int,
    prompt: str,
    size: str,
    has_ref: bool,
    ref_key: str | None,
    cost_cents: int,
    kind: str = "image",
    ref_keys: list[str] | None = None,
) -> tuple[int | None, int | None]:
    """原子扣费 + 插入 pending 记录。返回 (generation_id, balance_after)。
    余额不足返回 (None, None)。"""
    ref_keys_json = json.dumps(ref_keys) if ref_keys else None
    async with pool().acquire() as con:
        async with con.transaction():
            row = await con.fetchrow(
                """
                UPDATE users
                SET balance_cents = balance_cents - $1
                WHERE id = $2 AND balance_cents >= $1
                RETURNING balance_cents
                """,
                cost_cents, user_id,
            )
            if row is None:
                return None, None
            balance = int(row["balance_cents"])
            gen = await con.fetchrow(
                """
                INSERT INTO generations(user_id, prompt, size, has_ref, ref_key, cost_cents, status, kind, ref_keys)
                VALUES($1, $2, $3, $4, $5, $6, 'pending', $7, $8::jsonb)
                RETURNING id
                """,
                user_id, prompt, size, has_ref, ref_key, cost_cents, kind, ref_keys_json,
            )
            return int(gen["id"]), balance


async def mark_success(generation_id: int, result_key: str) -> None:
    async with pool().acquire() as con:
        await con.execute(
            "UPDATE generations SET status='success', result_key=$1 WHERE id=$2",
            result_key, generation_id,
        )


async def mark_success_svg(generation_id: int, svg_text: str) -> None:
    async with pool().acquire() as con:
        await con.execute(
            "UPDATE generations SET status='success', result_svg=$1 WHERE id=$2",
            svg_text, generation_id,
        )


async def mark_success_cad(generation_id: int, files: dict[str, str]) -> None:
    """CAD 成功：写入 result_files（{"step":key,"stl":key,"glb":key}）。"""
    async with pool().acquire() as con:
        await con.execute(
            "UPDATE generations SET status='success', result_files=$1::jsonb WHERE id=$2",
            json.dumps(files), generation_id,
        )


async def mark_failed_and_refund(generation_id: int, user_id: int, cost_cents: int, err: str) -> int:
    """失败：标记记录 + 退款。返回退款后余额。

    注意：保留 cost_cents 原值，以便在账户流水里展示退款金额。
    """
    async with pool().acquire() as con:
        async with con.transaction():
            await con.execute(
                "UPDATE generations SET status='failed', error=$1 WHERE id=$2",
                err[:1000], generation_id,
            )
            row = await con.fetchrow(
                "UPDATE users SET balance_cents = balance_cents + $1 WHERE id=$2 RETURNING balance_cents",
                cost_cents, user_id,
            )
            return int(row["balance_cents"]) if row else 0


async def list_history(user_id: int, limit: int = 30) -> list[dict[str, Any]]:
    async with pool().acquire() as con:
        rows = await con.fetch(
            """
            SELECT id, prompt, size, has_ref, ref_key, ref_keys, result_key, result_svg,
                   result_files, kind, cost_cents, status, error, created_at
            FROM generations
            WHERE user_id=$1
            ORDER BY id DESC
            LIMIT $2
            """,
            user_id, limit,
        )
    return [dict(r) for r in rows]


async def get_generation(generation_id: int) -> dict[str, Any] | None:
    async with pool().acquire() as con:
        row = await con.fetchrow("SELECT * FROM generations WHERE id=$1", generation_id)
    return _row_to_dict(row)


async def delete_generation(generation_id: int, user_id: int | None = None) -> dict[str, Any] | None:
    """删除一条记录。user_id 不为 None 时必须属于该用户。返回被删记录。"""
    async with pool().acquire() as con:
        if user_id is None:
            row = await con.fetchrow("DELETE FROM generations WHERE id=$1 RETURNING ref_key, ref_keys, result_key, result_files", generation_id)
        else:
            row = await con.fetchrow(
                "DELETE FROM generations WHERE id=$1 AND user_id=$2 RETURNING ref_key, ref_keys, result_key, result_files",
                generation_id, user_id,
            )
    return _row_to_dict(row)


# ----- 支付订单 -----

async def create_payment(
    user_id: int,
    out_trade_no: str,
    amount_cents: int,
    pay_type: str = "alipay",
) -> int:
    """创建 pending 订单。返回 payments.id。"""
    async with pool().acquire() as con:
        row = await con.fetchrow(
            """
            INSERT INTO payments(user_id, out_trade_no, amount_cents, pay_type, status)
            VALUES($1, $2, $3, $4, 'pending')
            RETURNING id
            """,
            user_id, out_trade_no, amount_cents, pay_type,
        )
    return int(row["id"])


async def get_payment(out_trade_no: str) -> dict[str, Any] | None:
    async with pool().acquire() as con:
        row = await con.fetchrow(
            "SELECT * FROM payments WHERE out_trade_no=$1",
            out_trade_no,
        )
    return _row_to_dict(row)


async def settle_payment(
    out_trade_no: str,
    expected_amount_cents: int,
    trade_no: str,
    notify_raw: str,
) -> tuple[str, int | None, int | None]:
    """幂等结算订单。返回 (result, user_id, balance_after)。

    result 可能值:
      - "success"      首次结算成功，余额已加
      - "already_paid" 订单已 paid，幂等返回 success，不重复加额
      - "not_found"    订单不存在
      - "amount_mismatch" 金额不一致
      - "no_user"      用户已被删除 (user_id IS NULL)
    """
    async with pool().acquire() as con:
        async with con.transaction():
            # 锁订单行
            row = await con.fetchrow(
                "SELECT id, user_id, amount_cents, status FROM payments "
                "WHERE out_trade_no=$1 FOR UPDATE",
                out_trade_no,
            )
            if row is None:
                return "not_found", None, None
            if int(row["amount_cents"]) != expected_amount_cents:
                return "amount_mismatch", None, None
            user_id = row["user_id"]
            if row["status"] == "paid":
                # 幂等：查余额返回但不加额
                if user_id is None:
                    return "already_paid", None, None
                bal = await con.fetchval(
                    "SELECT balance_cents FROM users WHERE id=$1",
                    user_id,
                )
                return "already_paid", int(user_id), int(bal or 0)
            if user_id is None:
                # 用户被删了，订单标为 paid 但不加额，避免重复推送
                await con.execute(
                    "UPDATE payments SET status='paid', trade_no=$1, "
                    "notify_raw=$2, paid_at=now() WHERE out_trade_no=$3",
                    trade_no, notify_raw[:4000], out_trade_no,
                )
                return "no_user", None, None
            # 锁用户行加额
            bal_row = await con.fetchrow(
                "UPDATE users SET balance_cents = balance_cents + $1 "
                "WHERE id=$2 RETURNING balance_cents",
                expected_amount_cents, user_id,
            )
            await con.execute(
                "UPDATE payments SET status='paid', trade_no=$1, "
                "notify_raw=$2, paid_at=now() WHERE out_trade_no=$3",
                trade_no, notify_raw[:4000], out_trade_no,
            )
            return "success", int(user_id), int(bal_row["balance_cents"])


async def list_user_ledger(
    user_id: int,
    limit: int = 50,
    type_filter: str | None = None,
) -> list[dict[str, Any]]:
    """用户账户流水：联合 generations + payments 统一返回。

    返回字段：
      kind:        recharge | consume | refund
      status:      success(已到账/已扣费) | pending | failed | expired
      amount_cents:绝对值，前端结合 kind/status 决定方向
      ref_id:      源表 id
      ref_no:      订单号（仅 recharge）
      pay_type:    支付方式（仅 recharge）
      prompt/gen_kind/error: generations 相关字段（仅 consume/refund）
      occur_at:    事件发生时间，paid_at 优先，否则 created_at

    type_filter: None / 'all' / 'recharge' / 'consume_refund'
    任何未识别值与 None 等价，返回全部。
    """
    tf = (type_filter or "all").strip().lower()
    if tf not in ("all", "recharge", "consume_refund"):
        tf = "all"

    # 按过滤器选择调用哪些子查询。避免 SQL 拼接，改用参数化 + 代码分支。
    sql_payments = """
        SELECT 'recharge'::TEXT       AS kind,
               id                     AS ref_id,
               out_trade_no           AS ref_no,
               amount_cents,
               status,
               pay_type,
               COALESCE(paid_at, created_at) AS occur_at,
               created_at,
               paid_at,
               NULL::TEXT             AS prompt,
               NULL::TEXT             AS gen_kind,
               NULL::TEXT             AS error
        FROM payments
        WHERE user_id=$1
    """
    sql_generations = """
        SELECT CASE WHEN status='failed' THEN 'refund' ELSE 'consume' END AS kind,
               id                     AS ref_id,
               NULL::VARCHAR          AS ref_no,
               cost_cents             AS amount_cents,
               status,
               NULL::VARCHAR          AS pay_type,
               created_at             AS occur_at,
               created_at,
               NULL::TIMESTAMPTZ      AS paid_at,
               prompt,
               kind                   AS gen_kind,
               error
        FROM generations
        WHERE user_id=$1
    """
    if tf == "recharge":
        body = sql_payments
    elif tf == "consume_refund":
        body = sql_generations
    else:
        body = f"{sql_payments}\nUNION ALL\n{sql_generations}"
    sql = f"""
        SELECT * FROM ({body}) t
        ORDER BY occur_at DESC, ref_id DESC
        LIMIT $2
    """
    async with pool().acquire() as con:
        rows = await con.fetch(sql, user_id, limit)
    return [dict(r) for r in rows]


async def list_all_payments(
    limit: int = 100, status: str | None = None, key_prefix: str | None = None
) -> list[dict[str, Any]]:
    """管理端列出所有订单（含用户 access_key 前缀以便定位）。"""
    where = []
    args: list[Any] = []
    if status:
        args.append(status)
        where.append(f"p.status=${len(args)}")
    if key_prefix:
        args.append(key_prefix + "%")
        where.append(f"u.access_key LIKE ${len(args)}")
    args.append(limit)
    sql = (
        "SELECT p.id, p.out_trade_no, p.trade_no, p.amount_cents, p.status, "
        "p.pay_type, p.created_at, p.paid_at, p.user_id, "
        "LEFT(u.access_key, 12) AS user_key_prefix, u.name AS user_name "
        "FROM payments p LEFT JOIN users u ON u.id=p.user_id "
    )
    if where:
        sql += "WHERE " + " AND ".join(where) + " "
    sql += f"ORDER BY p.id DESC LIMIT ${len(args)}"
    async with pool().acquire() as con:
        rows = await con.fetch(sql, *args)
    return [dict(r) for r in rows]


async def admin_payment_stats() -> dict[str, Any]:
    async with pool().acquire() as con:
        total_paid = await con.fetchval(
            "SELECT COALESCE(SUM(amount_cents),0) FROM payments WHERE status='paid'"
        ) or 0
        today_paid = await con.fetchval(
            "SELECT COALESCE(SUM(amount_cents),0) FROM payments "
            "WHERE status='paid' AND paid_at >= date_trunc('day', now())"
        ) or 0
        pending_cnt = await con.fetchval(
            "SELECT COUNT(*) FROM payments WHERE status='pending'"
        ) or 0
        paid_cnt = await con.fetchval(
            "SELECT COUNT(*) FROM payments WHERE status='paid'"
        ) or 0
    return {
        "total_paid_cents": int(total_paid),
        "today_paid_cents": int(today_paid),
        "pending_count": int(pending_cnt),
        "paid_count": int(paid_cnt),
    }


async def admin_stats() -> dict[str, Any]:
    async with pool().acquire() as con:
        users_total = await con.fetchval("SELECT COUNT(*) FROM users")
        today_calls = await con.fetchval(
            "SELECT COUNT(*) FROM generations WHERE created_at >= date_trunc('day', now())"
        )
        total_cost = await con.fetchval(
            "SELECT COALESCE(SUM(cost_cents),0) FROM generations WHERE status='success'"
        )
        failed = await con.fetchval("SELECT COUNT(*) FROM generations WHERE status='failed'")
    return {
        "users_total": int(users_total or 0),
        "today_calls": int(today_calls or 0),
        "total_cost_cents": int(total_cost or 0),
        "failed_calls": int(failed or 0),
    }


async def admin_list_generations(
    user_id: int | None = None,
    status: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    sql = (
        "SELECT g.id, g.user_id, u.name AS user_name, g.prompt, g.size, g.has_ref, "
        "g.ref_key, g.ref_keys, g.result_key, g.result_svg, g.result_files, g.kind, g.cost_cents, g.status, g.error, g.created_at "
        "FROM generations g LEFT JOIN users u ON u.id = g.user_id WHERE 1=1"
    )
    args: list[Any] = []
    if user_id is not None:
        args.append(user_id)
        sql += f" AND g.user_id = ${len(args)}"
    if status:
        args.append(status)
        sql += f" AND g.status = ${len(args)}"
    args.append(limit)
    sql += f" ORDER BY g.id DESC LIMIT ${len(args)}"
    async with pool().acquire() as con:
        rows = await con.fetch(sql, *args)
    return [dict(r) for r in rows]
