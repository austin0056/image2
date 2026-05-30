"""支付路由：充值发起 / 异步通知 / 浏览器跳回 / 订单查询。"""
from __future__ import annotations

import logging
import re
import secrets
import time
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from . import db, zpay
from .config import settings
from .deps import current_user

router = APIRouter()
log = logging.getLogger("payment")

# out_trade_no 格式由本服务生成，仅允许这种字符。回跳参数据此校验。
_OUT_TRADE_NO_RE = re.compile(r"^r_\d{1,12}_\d{13}_[0-9a-f]{4}$")


# ---------- 请求模型 ----------

class RechargeBody(BaseModel):
    # 接收字符串/数字均可，内部统一为 Decimal，避免浮点 0.01 漂移。
    amount_yuan: Decimal = Field(..., gt=0)

    @field_validator("amount_yuan", mode="before")
    @classmethod
    def _coerce(cls, v):
        try:
            return Decimal(str(v))
        except (InvalidOperation, TypeError):
            raise ValueError("金额格式不正确")

    @field_validator("amount_yuan")
    @classmethod
    def _max_2_decimals(cls, v: Decimal) -> Decimal:
        # 不允许超过 2 位小数（0.01 元最小粒度）
        if v != v.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP):
            raise ValueError("金额最多保留 2 位小数")
        return v


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
async def create_recharge(
    body: RechargeBody,
    user: dict = Depends(current_user),
) -> dict:
    _ensure_zpay_configured()

    cents = int((body.amount_yuan * 100).to_integral_value(rounding=ROUND_HALF_UP))
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

def _query_dict(req: Request) -> dict:
    """只取 query。zpay 的 GET 通知走这里；POST form 在调用方兜底解析。"""
    return {k: v for k, v in req.query_params.multi_items()}


@router.api_route("/api/payment/notify", methods=["GET", "POST"])
async def payment_notify(req: Request) -> PlainTextResponse:
    """zpay 异步通知。必须在 5 秒内返回纯文本 'success' 才算成功。"""
    params = _query_dict(req)
    if not params:
        try:
            form = await req.form()
            params = {k: str(v) for k, v in form.items()}
        except Exception as e:
            log.warning("zpay notify: form parse failed: %s", e)
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
    """支付完成后浏览器跳回。带 out_trade_no 参数转给前端轮询。

    out_trade_no 必须匹配本服务生成的格式，避免成为开放重定向参数注入面。
    """
    out_trade_no = req.query_params.get("out_trade_no", "")
    target = "/static/user.html"
    if out_trade_no and _OUT_TRADE_NO_RE.match(out_trade_no):
        target += "?" + urlencode({"recharge": out_trade_no})
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


# ---------- /api/ledger：账户流水（充值 + 消费 + 退款） ----------

@router.get("/api/ledger")
async def my_ledger(
    limit: int = 50,
    type: str | None = None,  # noqa: A002 兼容前端 query 名
    user: dict = Depends(current_user),
) -> dict:
    """统一流水。type 可选 'all' / 'recharge' / 'consume_refund'。

    返回：
      items: 按 occur_at 倒序的条目。后端只返结构化字段，展示由前端决定。
      summary: 当前筛选范围内的汇总（income / expense / pending_in 三个总额）。
      filter: 实际生效的 type。
      truncated: 是否达到 limit 上限。
    """
    limit = max(1, min(limit, 200))
    rows = await db.list_user_ledger(user["id"], limit=limit, type_filter=type)
    items = []
    income = 0
    expense = 0
    pending_in = 0
    for r in rows:
        kind = r["kind"]
        st = r["status"]
        amt = int(r["amount_cents"])
        # 方向与增减
        if kind == "recharge":
            if st == "paid":
                direction = "in"
                delta = amt
                income += amt
            elif st == "pending":
                direction = "none"
                delta = 0
                pending_in += amt
            else:  # expired / failed / unknown
                direction = "none"
                delta = 0
        elif kind == "refund":
            direction = "in"
            delta = amt
            income += amt
        else:  # consume
            if st == "success":
                direction = "out"
                delta = -amt
                expense += amt
            else:  # pending / failed (不应出现这里，failed 已被划到 refund)
                direction = "none"
                delta = 0
        items.append({
            "kind": kind,
            "status": st,
            "direction": direction,
            "amount_cents": amt,
            "delta_cents": delta,
            "ref_id": r["ref_id"],
            "ref_no": r.get("ref_no") or "",
            "pay_type": r.get("pay_type") or "",
            "prompt": (r.get("prompt") or "")[:200],
            "gen_kind": r.get("gen_kind") or "",
            "error": (r.get("error") or "")[:200],
            "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
            "occur_at": r["occur_at"].isoformat() if r["occur_at"] else None,
            "paid_at": r["paid_at"].isoformat() if r.get("paid_at") else None,
        })
    actual_filter = (type or "all").lower()
    if actual_filter not in ("all", "recharge", "consume_refund"):
        actual_filter = "all"
    return {
        "items": items,
        "summary": {
            "income_cents": income,
            "expense_cents": expense,
            "pending_in_cents": pending_in,
            "count": len(items),
        },
        "filter": actual_filter,
        "limit": limit,
    }
