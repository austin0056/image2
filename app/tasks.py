"""后台清扫任务：处理超时未完成的 generation 与 payment。

启动机制：在 main.lifespan 里 asyncio.create_task(reaper_loop()) 起一个常驻协程。
异常时仅记日志，不让循环退出。
"""
from __future__ import annotations

import asyncio
import logging

from . import db

log = logging.getLogger("image2.tasks")

# 生成任务超时 5 分钟（含上游耗时 + 缓冲）
GEN_TIMEOUT_SECONDS = 5 * 60
# 充值订单超时 30 分钟（zpay 收银台允许长一些）
PAY_TIMEOUT_SECONDS = 30 * 60
# 每轮扫描间隔
LOOP_INTERVAL_SECONDS = 60


async def _reap_pending_generations() -> int:
    """把 pending 超过 5 分钟的 generation 标 failed 并退款。返回处理条数。"""
    sql_select = (
        "SELECT id, user_id, cost_cents FROM generations "
        "WHERE status='pending' AND created_at < now() - INTERVAL '%d seconds' "
        "LIMIT 100"
    ) % GEN_TIMEOUT_SECONDS
    async with db.pool().acquire() as con:
        rows = await con.fetch(sql_select)
    if not rows:
        return 0
    n = 0
    for r in rows:
        try:
            await db.mark_failed_and_refund(
                int(r["id"]),
                int(r["user_id"]),
                int(r["cost_cents"]),
                "timeout: 上游/进程超时由清扫任务自动退款",
            )
            n += 1
            log.info("reaper: refunded generation id=%s user=%s cents=%s",
                     r["id"], r["user_id"], r["cost_cents"])
        except Exception as e:
            log.exception("reaper: failed to refund gen=%s: %s", r["id"], e)
    return n


async def _reap_pending_payments() -> int:
    """把 pending 超过 30 分钟的 payment 标记为 expired，避免列表里堆积无效单。"""
    sql = (
        "UPDATE payments SET status='expired' "
        "WHERE status='pending' AND created_at < now() - INTERVAL '%d seconds' "
        "RETURNING id"
    ) % PAY_TIMEOUT_SECONDS
    async with db.pool().acquire() as con:
        rows = await con.fetch(sql)
    if rows:
        log.info("reaper: marked %d expired payments", len(rows))
    return len(rows)


async def reaper_loop() -> None:
    """常驻清扫循环。出错不退出。"""
    log.info("reaper started: gen_timeout=%ds pay_timeout=%ds interval=%ds",
             GEN_TIMEOUT_SECONDS, PAY_TIMEOUT_SECONDS, LOOP_INTERVAL_SECONDS)
    while True:
        try:
            await _reap_pending_generations()
            await _reap_pending_payments()
        except asyncio.CancelledError:
            log.info("reaper cancelled")
            raise
        except Exception as e:
            log.exception("reaper iteration error: %s", e)
        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
