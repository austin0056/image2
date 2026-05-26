"""鉴权依赖：用户 access key 与管理员 session。"""
from __future__ import annotations

from typing import Any

from fastapi import Cookie, Header, HTTPException, Query
from itsdangerous import BadSignature, SignatureExpired, TimestampSigner

from . import db
from .config import settings

_signer = TimestampSigner(settings.session_secret, salt="admin-session")
_SESSION_MAX_AGE = 7 * 24 * 3600


def make_admin_token() -> str:
    return _signer.sign(b"admin").decode("utf-8")


def verify_admin_token(token: str | None) -> bool:
    if not token:
        return False
    try:
        _signer.unsign(token, max_age=_SESSION_MAX_AGE)
        return True
    except (BadSignature, SignatureExpired):
        return False


async def require_admin(admin_session: str | None = Cookie(default=None)) -> None:
    if not verify_admin_token(admin_session):
        raise HTTPException(status_code=401, detail="未登录")


def _key_from_header(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


async def require_user(access_key: str) -> dict[str, Any]:
    """接收原始 access_key 字符串。仅在需要表单/路由参数不能使用头的场景使用。"""
    user = await db.get_user_by_key(access_key.strip())
    if not user:
        raise HTTPException(status_code=401, detail="access key 无效")
    return user


async def current_user(
    authorization: str | None = Header(default=None),
    access_key: str | None = Query(default=None),
) -> dict[str, Any]:
    """依赖注入用。优先 Bearer，其次老 query。主要用于 GET 类接口。"""
    key = _key_from_header(authorization) or (access_key.strip() if access_key else None)
    if not key:
        raise HTTPException(status_code=401, detail="未提供凭证")
    return await require_user(key)
