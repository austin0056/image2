"""支付路由：充值发起 / 异步通知 / 浏览器跳回 / 订单查询。"""
from __future__ import annotations

import logging
import secrets
import time
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel, Field

from . import db, zpay
from .config import settings
from .deps import current_user

router = APIRouter()
log = logging.getLogger("payment")


# ---------- 请求模型 ----------

class RechargeBody(BaseModel):
    access_key: str = Field(..., min_length=8)
    amount_yuan: float = Field(..., gt=0)


# ---------- 工具 ----------

def _ensure_zpay_configured() -> None:
    if not (settings.zpay_pid and settings.zpay_key and settings.public_base_url):
        raise HTTPException(
            status_code=503,
            detail="支付暂未配置（缺 ZPAY_PID / ZPAY_KEY / PUBLIC_BASE_URL）",
        )


def _gen_out_trade_no(user_id: int) -> str:
    """格式 r_{uid}_{ts13}_{rand4}，最长 32 位以内。"""
    ts = int(time.time() * 1000)
    rand = secrets.token_hex(2)  # 4 chars
    return f"r_{user_id}_{ts}_{rand}"[:32]


# ---------- /api/recharge：发起 ----------

@router.post("/api/recharge")
async def create_recharge(body: RechargeBody) -> dict:
    _ensure_zpay_configured()

    user = await db.get_user_by_key(body.access_key)
    if not user:
        raise HTTPException(401, "access key 无效")

    cents = int(round(body.amount_yuan * 100))
    if cents < settings.recharge_min_cents:
        raise HTTPException(400, f"金额不能小于 ¥{settings.recharge_min_cents/100:.2f}")
    if cents > settings.recharge_max_cents:
        raise HTTPException(400, f"金额不能大于 ¥{settings.recharge_max_cents/100:.2f}")

    out_trade_no = _gen_out_trade_no(user["id"])
    await db.create_payment(user["id"], out_trade_no, cents, "alipay")

    pay_url = zpay.build_pay_url(
        pid=settings.zpay_pid,
        key=settings.zpay_key,
        name="image2 余额充值",
        money=zpay.cents_to_yuan_str(cents),
        out_trade_no=out_trade_no,
        notify_url=f"{settings.public_base_url}/api/payment/notify",
        return_url=f"{settings.public_base_url}/api/payment/return",
        pay_type="alipay",
        base=settings.zpay_base,
    )
    return {
        "out_trade_no": out_trade_no,
        "amount_cents": cents,
        "pay_url": pay_url,
    }


# ---------- /api/payment/notify：服务端异步通知 ----------

def _notify_dict(req: Request) -> dict:
    """同时支持 GET query 和 POST form 两种回调。"""
    return {k: v for k, v in req.query_params.multi_items()}


@router.api_route("/api/payment/notify", methods=["GET", "POST"])
async def payment_notify(req: Request) -> PlainTextResponse:
    """zpay 异步通知。必须在 5 秒内返回纯文本 'success' 才算成功。"""
    params = _notify_dict(req)
    if not params:
        try:
            form = await req.form()
            params = {k: str(v) for k, v in form.items()}
        except Exception:
            params = {}

    raw = urlencode(params)
    log.info("zpay notify recv: %s", raw[:500])

    # 1) 验签
    if not zpay.verify_sign(params, settings.zpay_key):
        log.warning("zpay notify sign invalid")
        return PlainTextResponse("fail")

    # 2) 状态校验
    status = params.get("trade_status") or ""
    if status != "TRADE_SUCCESS":
        # 终态 (已关闭/已完成)：不结算，回 success 避免 zpay 重试
        if status in ("TRADE_CLOSED", "TRADE_FINISHED"):
            log.info("zpay notify terminal-not-success: %s status=%s",
                     params.get("out_trade_no", ""), status)
            return PlainTextResponse("success")
        # 临时/未知态：回 fail 让 zpay 按重试策略推送下一条
        log.warning("zpay notify non-success status=%s out_trade_no=%s",
                    status, params.get("out_trade_no", ""))
        return PlainTextResponse("fail")

    out_trade_no = params.get("out_trade_no", "")
    money = params.get("money", "")
    trade_no = params.get("trade_no", "")
    if not out_trade_no or not money:
        return PlainTextResponse("fail")

    # 3) 金额转分
    try:
        cents = int(round(float(money) * 100))
    except ValueError:
        return PlainTextResponse("fail")

    # 4) 结算（DB 内部幂等）
    result, user_id, balance = await db.settle_payment(
        out_trade_no=out_trade_no,
        expected_amount_cents=cents,
        trade_no=trade_no,
        notify_raw=raw,
    )
    if result == "not_found":
        log.warning("zpay notify: order not found %s", out_trade_no)
        return PlainTextResponse("fail")
    if result == "amount_mismatch":
        log.error("zpay notify: amount mismatch %s money=%s", out_trade_no, money)
        return PlainTextResponse("fail")
    log.info("zpay notify settled: %s -> %s, uid=%s, bal=%s",
             out_trade_no, result, user_id, balance)
    return PlainTextResponse("success")


# ---------- /api/payment/return：浏览器跳回 ----------

@router.get("/api/payment/return")
async def payment_return(req: Request):
    """支付完成后浏览器跳回。带 out_trade_no 参数转给前端轮询。"""
    out_trade_no = req.query_params.get("out_trade_no", "")
    target = "/static/user.html"
    if out_trade_no:
        target += f"?recharge={out_trade_no}"
    return RedirectResponse(url=target, status_code=302)


# ---------- /api/payment/{out_trade_no}：前端轮询 ----------

@router.get("/api/payment/{out_trade_no}")
async def get_payment_status(
    out_trade_no: str,
    user: dict = Depends(current_user),
) -> JSONResponse:
    p = await db.get_payment(out_trade_no)
    if not p:
        raise HTTPException(404, "订单不存在")
    if p.get("user_id") != user["id"]:
        raise HTTPException(403, "订单不属于你")

    return JSONResponse({
        "out_trade_no": p["out_trade_no"],
        "trade_no": p.get("trade_no") or "",
        "status": p["status"],
        "amount_cents": p["amount_cents"],
        "paid_at": p["paid_at"].isoformat() if p.get("paid_at") else None,
        "balance_cents": user["balance_cents"] if p["status"] != "paid" else None,
    })


# ---------- /api/recharge/presets：金额预设 ----------

@router.get("/api/recharge/presets")
async def recharge_presets() -> dict:
    return {
        "presets_yuan": list(settings.recharge_presets_yuan),
        "min_yuan": settings.recharge_min_cents / 100,
        "max_yuan": settings.recharge_max_cents / 100,
        "enabled": bool(settings.zpay_pid and settings.zpay_key and settings.public_base_url),
    }


# ---------- /api/payments：用户充值历史 ----------

@router.get("/api/payments")
async def my_payments(
    limit: int = 50,
    user: dict = Depends(current_user),
) -> dict:
    limit = max(1, min(limit, 200))
    rows = await db.list_user_payments(user["id"], limit=limit)
    return {
        "items": [
            {
                "id": r["id"],
                "out_trade_no": r["out_trade_no"],
                "trade_no": r["trade_no"] or "",
                "amount_cents": r["amount_cents"],
                "status": r["status"],
                "pay_type": r["pay_type"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "paid_at": r["paid_at"].isoformat() if r["paid_at"] else None,
            }
            for r in rows
        ]
    }


# ---------- /api/ledger：账户流水（充值 + 消费 + 退款） ----------

@router.get("/api/ledger")
async def my_ledger(
    limit: int = 50,
    type: str | None = None,  # noqa: A002 兼容前端 query 名
    user: dict = Depends(current_user),
) -> dict:
    """统一流水。type 可选 'recharge' 只看充值，'consume_refund' 只看消费/退款。"""
    limit = max(1, min(limit, 200))
    rows = await db.list_user_ledger(user["id"], limit=limit, type_filter=type)
    items = []
    for r in rows:
        kind = r["kind"]
        st = r["status"]
        amt = int(r["amount_cents"])
        # 计算有符号增减
        if kind == "recharge":
            delta = amt if st == "paid" else 0
        elif kind == "refund":
            delta = amt
        else:  # consume
            delta = -amt if st == "success" else (0 if st == "pending" else 0)
        # 标题
        if kind == "recharge":
            title = "余额充值"
            if st == "paid":
                sub = "已到账"
            elif st == "pending":
                sub = "待支付"
            elif st == "expired":
                sub = "已过期（30 分钟未付款）"
            else:
                sub = st
        elif kind == "refund":
            title = "失败退款"
            sub = (r.get("prompt") or "")[:60]
        else:  # consume
            gk = r.get("gen_kind") or "image"
            title = "图标生成" if gk == "icon" else "图片生成"
            if st == "success":
                sub = (r.get("prompt") or "")[:60]
            elif st == "pending":
                sub = "生成中…"
            else:
                sub = st
        items.append({
            "kind": kind,
            "status": st,
            "delta_cents": delta,
            "amount_cents": amt,
            "title": title,
            "sub": sub,
            "ref_id": r["ref_id"],
            "ref_no": r.get("ref_no") or "",
            "pay_type": r.get("pay_type") or "",
            "occur_at": r["occur_at"].isoformat() if r["occur_at"] else None,
            "paid_at": r["paid_at"].isoformat() if r.get("paid_at") else None,
        })
    return {"items": items}
