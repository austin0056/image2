"""鉴权依赖：用户 access key 与管理员 session。"""
from __future__ import annotations

from typing import Any

from fastapi import Cookie, HTTPException
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


async def require_user(access_key: str) -> dict[str, Any]:
    user = await db.get_user_by_key(access_key.strip())
    if not user:
        raise HTTPException(status_code=401, detail="access key 无效")
    return user
