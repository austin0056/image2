"""管理员侧 API。"""
from __future__ import annotations

import io
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import db, storage
from .config import settings
from .deps import make_admin_token, require_admin

router = APIRouter()


class LoginBody(BaseModel):
    password: str


class CreateUserBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)


class RenameBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)


class TopupBody(BaseModel):
    yuan: float = Field(..., gt=0, le=1_000_000)


@router.post("/api/admin/login")
async def admin_login(body: LoginBody, response: Response) -> dict[str, Any]:
    if body.password != settings.admin_password:
        raise HTTPException(401, "密码错误")
    token = make_admin_token()
    response.set_cookie(
        "admin_session",
        token,
        max_age=7 * 24 * 3600,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return {"ok": True}


@router.post("/api/admin/logout")
async def admin_logout(response: Response) -> dict[str, Any]:
    response.delete_cookie("admin_session", path="/")
    return {"ok": True}


@router.get("/api/admin/me", dependencies=[Depends(require_admin)])
async def admin_me() -> dict[str, Any]:
    return {"ok": True}


@router.get("/api/admin/stats", dependencies=[Depends(require_admin)])
async def admin_stats() -> dict[str, Any]:
    return await db.admin_stats()


@router.get("/api/admin/users", dependencies=[Depends(require_admin)])
async def admin_users() -> dict[str, Any]:
    rows = await db.list_users()
    items = [
        {
            "id": r["id"],
            "access_key": r["access_key"],
            "name": r["name"],
            "balance_cents": r["balance_cents"],
            "gen_count": r["gen_count"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in rows
    ]
    return {"items": items}


@router.post("/api/admin/users", dependencies=[Depends(require_admin)])
async def admin_create_user(body: CreateUserBody) -> dict[str, Any]:
    u = await db.create_user(body.name.strip())
    return {
        "id": u["id"],
        "access_key": u["access_key"],
        "name": u["name"],
        "balance_cents": u["balance_cents"],
    }


@router.patch("/api/admin/users/{user_id}", dependencies=[Depends(require_admin)])
async def admin_rename(user_id: int, body: RenameBody) -> dict[str, Any]:
    await db.update_user_name(user_id, body.name.strip())
    return {"ok": True}


@router.post("/api/admin/users/{user_id}/topup", dependencies=[Depends(require_admin)])
async def admin_topup(user_id: int, body: TopupBody) -> dict[str, Any]:
    cents = int(round(body.yuan * 100))
    if cents <= 0:
        raise HTTPException(400, "金额过小")
    new_balance = await db.topup_user(user_id, cents)
    return {"balance_cents": new_balance}


@router.delete("/api/admin/users/{user_id}", dependencies=[Depends(require_admin)])
async def admin_delete(user_id: int) -> dict[str, Any]:
    await db.delete_user(user_id)
    return {"ok": True}


@router.get("/api/admin/generations", dependencies=[Depends(require_admin)])
async def admin_generations(
    user_id: int | None = Query(default=None),
    status: str | None = Query(default=None),
) -> dict[str, Any]:
    rows = await db.admin_list_generations(user_id=user_id, status=status)
    items = []
    for r in rows:
        kind = r.get("kind") or "image"
        item = {
            "id": r["id"],
            "user_id": r["user_id"],
            "user_name": r["user_name"],
            "kind": kind,
            "prompt": r["prompt"],
            "size": r["size"],
            "has_ref": r["has_ref"],
            "status": r["status"],
            "error": r["error"],
            "cost_cents": r["cost_cents"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        if kind == "icon":
            item["result_url"] = f"/api/admin/files/svg/{r['id']}" if r.get("result_svg") else None
            item["ref_url"] = None
        else:
            item["result_url"] = f"/api/admin/files/result/{r['id']}" if r["result_key"] else None
            item["ref_url"] = f"/api/admin/files/ref/{r['id']}" if r["ref_key"] else None
        items.append(item)
    return {"items": items}


@router.get("/api/admin/files/{kind}/{generation_id}", dependencies=[Depends(require_admin)])
async def admin_get_file(kind: str, generation_id: int):
    if kind not in ("ref", "result", "svg"):
        raise HTTPException(404)
    gen = await db.get_generation(generation_id)
    if not gen:
        raise HTTPException(404)
    if kind == "svg":
        svg = gen.get("result_svg")
        if not svg:
            raise HTTPException(404)
        return StreamingResponse(io.BytesIO(svg.encode("utf-8")), media_type="image/svg+xml")
    key = gen["ref_key"] if kind == "ref" else gen["result_key"]
    if not key:
        raise HTTPException(404)
    body, ctype = await storage.fetch_object(key)
    return StreamingResponse(io.BytesIO(body), media_type=ctype)


@router.delete("/api/admin/generations/{generation_id}", dependencies=[Depends(require_admin)])
async def admin_delete_generation(generation_id: int) -> dict[str, Any]:
    deleted = await db.delete_generation(generation_id)
    if not deleted:
        raise HTTPException(404, "记录不存在")
    keys = [k for k in (deleted.get("ref_key"), deleted.get("result_key")) if k]
    if keys:
        await storage.delete_keys(keys)
    return {"ok": True}
