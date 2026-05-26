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
                INSERT INTO generations(user_id, prompt, size, has_ref, ref_key, cost_cents, status, kind)
                VALUES($1, $2, $3, $4, $5, $6, 'pending', $7)
                RETURNING id
                """,
                user_id, prompt, size, has_ref, ref_key, cost_cents, kind,
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
            SELECT id, prompt, size, has_ref, ref_key, result_key, result_svg,
                   kind, cost_cents, status, error, created_at
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
            row = await con.fetchrow("DELETE FROM generations WHERE id=$1 RETURNING ref_key, result_key", generation_id)
        else:
            row = await con.fetchrow(
                "DELETE FROM generations WHERE id=$1 AND user_id=$2 RETURNING ref_key, result_key",
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


async def list_user_payments(user_id: int, limit: int = 20) -> list[dict[str, Any]]:
    async with pool().acquire() as con:
        rows = await con.fetch(
            """
            SELECT id, out_trade_no, trade_no, amount_cents, status, pay_type,
                   created_at, paid_at
            FROM payments WHERE user_id=$1 ORDER BY id DESC LIMIT $2
            """,
            user_id, limit,
        )
    return [dict(r) for r in rows]


async def list_user_ledger(
    user_id: int,
    limit: int = 50,
    type_filter: str | None = None,
) -> list[dict[str, Any]]:
    """用户账户流水：联合 generations + payments 统一返回。

    返回字段：
      kind:        recharge | consume | refund
      status:      success(已到账/已扣费) | pending | failed | expired
      delta_cents: 有符号，+为入账 -为出账，pending 为 0
      title / sub: 用于前端展示的主、次文本
      ref_id:      源表 id
      ref_no:      订单号（仅 recharge）

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
        ORDER BY occur_at DESC NULLS LAST, ref_id DESC
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
        "g.ref_key, g.result_key, g.result_svg, g.kind, g.cost_cents, g.status, g.error, g.created_at "
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
