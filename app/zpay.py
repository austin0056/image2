"""zpay (zpayz.cn) 易支付接入工具。

文档要点：
1. 参数按 ASCII 升序排序，sign / sign_type / 空值 不参与签名
2. 拼成 a=b&c=d 后追加商户 KEY，做 md5 小写 = sign
3. submit.php 是页面跳转支付，跳浏览器即可
"""
from __future__ import annotations

import hashlib
from typing import Iterable
from urllib.parse import urlencode


def _sign_string(params: dict, key: str) -> str:
    """构造签名前的明文串（已含 KEY）。便于排错。"""
    items = []
    for k in sorted(params.keys()):
        if k in ("sign", "sign_type"):
            continue
        v = params[k]
        if v is None or v == "":
            continue
        items.append(f"{k}={v}")
    return "&".join(items) + key


def md5_sign(params: dict, key: str) -> str:
    raw = _sign_string(params, key)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def verify_sign(params: dict, key: str) -> bool:
    """校验回调里的 sign 是否合法。大小写不敏感，但官方文档指定小写。"""
    given = (params.get("sign") or "").strip().lower()
    if not given:
        return False
    expected = md5_sign(params, key)
    return given == expected


def build_pay_url(
    *,
    pid: str,
    key: str,
    name: str,
    money: str,
    out_trade_no: str,
    notify_url: str,
    return_url: str,
    pay_type: str = "alipay",
    base: str = "https://zpayz.cn",
    extra_param: str | None = None,
) -> str:
    """构造 submit.php 跳转 URL。前端直接 window.location.href = 此 URL。"""
    params: dict = {
        "pid": pid,
        "name": name,
        "money": money,
        "out_trade_no": out_trade_no,
        "notify_url": notify_url,
        "return_url": return_url,
        "type": pay_type,
    }
    if extra_param:
        params["param"] = extra_param
    params["sign"] = md5_sign(params, key)
    params["sign_type"] = "MD5"
    return f"{base.rstrip('/')}/submit.php?" + urlencode(params)


def normalize_money(yuan: float) -> str:
    """zpay 金额最多 2 位小数。返回字符串避免浮点误差。"""
    cents = int(round(yuan * 100))
    if cents < 1:
        raise ValueError("金额必须大于 0")
    return f"{cents // 100}.{cents % 100:02d}"


def cents_to_yuan_str(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"
