"""PostgreSQL 接入层。提供连接池、建表、用户与生成记录的增删改查。"""
from __future__ import annotations

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
            min_size=1,
            max_size=10,
            command_timeout=30,
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
) -> tuple[int | None, int | None]:
    """原子扣费 + 插入 pending 记录。返回 (generation_id, balance_after)。
    余额不足返回 (None, None)。"""
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
                INSERT INTO generations(user_id, prompt, size, has_ref, ref_key, cost_cents, status)
                VALUES($1, $2, $3, $4, $5, $6, 'pending')
                RETURNING id
                """,
                user_id, prompt, size, has_ref, ref_key, cost_cents,
            )
            return int(gen["id"]), balance


async def mark_success(generation_id: int, result_key: str) -> None:
    async with pool().acquire() as con:
        await con.execute(
            "UPDATE generations SET status='success', result_key=$1 WHERE id=$2",
            result_key, generation_id,
        )


async def mark_failed_and_refund(generation_id: int, user_id: int, cost_cents: int, err: str) -> int:
    """失败：标记记录 + 退款。返回退款后余额。"""
    async with pool().acquire() as con:
        async with con.transaction():
            await con.execute(
                "UPDATE generations SET status='failed', error=$1, cost_cents=0 WHERE id=$2",
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
            SELECT id, prompt, size, has_ref, ref_key, result_key, cost_cents, status, error, created_at
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
        "g.ref_key, g.result_key, g.cost_cents, g.status, g.error, g.created_at "
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
