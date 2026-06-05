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


class ImageProviderSettingsBody(BaseModel):
    upstream_base: str = Field(..., min_length=1, max_length=500)
    upstream_model: str = Field(..., min_length=1, max_length=200)
    # 空字符串/None 表示不修改现有密钥。
    upstream_key: str | None = Field(default=None, max_length=5000)
    price_cents: int = Field(..., ge=0, le=1_000_000)            # 1K 单价
    price_2k_cents: int | None = Field(default=None, ge=0, le=1_000_000)
    price_4k_cents: int | None = Field(default=None, ge=0, le=1_000_000)


class ProviderSettingsBody(BaseModel):
    base: str = Field(..., min_length=1, max_length=500)
    model: str = Field(..., min_length=1, max_length=200)
    key: str | None = Field(default=None, max_length=5000)  # 空/None=不改
    api_style: str | None = Field(default=None, max_length=20)  # auto/chat/responses；None=不改


@router.post("/api/admin/login")
async def admin_login(body: LoginBody, response: Response) -> dict[str, Any]:
    # 去掉首尾空白：避免在 Zeabur 粘贴 ADMIN_PASSWORD 时带了行尾换行/空格导致永远对不上。
    if body.password.strip() != settings.admin_password.strip():
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
    base = await db.admin_stats()
    pay = await db.admin_payment_stats()
    base.update(pay)
    return base


@router.get("/api/admin/settings/image-provider", dependencies=[Depends(require_admin)])
async def admin_get_image_provider_settings() -> dict[str, Any]:
    return await db.get_image_provider_settings(reveal_key=False)


@router.patch("/api/admin/settings/image-provider", dependencies=[Depends(require_admin)])
async def admin_update_image_provider_settings(body: ImageProviderSettingsBody) -> dict[str, Any]:
    upstream_base = body.upstream_base.strip().rstrip("/")
    upstream_model = body.upstream_model.strip()
    if not upstream_base.startswith(("http://", "https://")):
        raise HTTPException(400, "API Base URL 必须以 http:// 或 https:// 开头")
    if not upstream_model:
        raise HTTPException(400, "模型名不能为空")
    key = body.upstream_key.strip() if body.upstream_key is not None else ""
    return await db.update_image_provider_settings(
        upstream_base=upstream_base,
        upstream_model=upstream_model,
        price_cents=body.price_cents,
        price_2k_cents=body.price_2k_cents,
        price_4k_cents=body.price_4k_cents,
        upstream_key=key or None,
    )


# ----- 图片生成 · 额外来源池（负载均衡 / 抗限流） -----

class ImageSourceBody(BaseModel):
    base: str = Field(..., min_length=1, max_length=500)
    model: str = Field(..., min_length=1, max_length=200)
    key: str = Field(..., min_length=1, max_length=5000)


class ImageSourcePatch(BaseModel):
    enabled: bool | None = None
    base: str | None = Field(default=None, max_length=500)
    model: str | None = Field(default=None, max_length=200)
    key: str | None = Field(default=None, max_length=5000)  # 空=保留原 key


@router.get("/api/admin/settings/image-sources", dependencies=[Depends(require_admin)])
async def admin_get_image_sources() -> dict[str, Any]:
    return {"items": await db.get_image_extra_sources(reveal_key=False)}


@router.post("/api/admin/settings/image-sources", dependencies=[Depends(require_admin)])
async def admin_add_image_source(body: ImageSourceBody) -> dict[str, Any]:
    base = body.base.strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise HTTPException(400, "API Base URL 必须以 http:// 或 https:// 开头")
    model, key = body.model.strip(), body.key.strip()
    if not model or not key:
        raise HTTPException(400, "模型与 API Key 不能为空")
    return {"items": await db.add_image_extra_source(base=base, model=model, key=key)}


@router.patch("/api/admin/settings/image-sources/{sid}", dependencies=[Depends(require_admin)])
async def admin_update_image_source(sid: str, body: ImageSourcePatch) -> dict[str, Any]:
    base = body.base.strip().rstrip("/") if body.base is not None else None
    if base and not base.startswith(("http://", "https://")):
        raise HTTPException(400, "API Base URL 必须以 http:// 或 https:// 开头")
    return {"items": await db.update_image_extra_source(
        sid,
        enabled=body.enabled,
        base=base,
        model=body.model.strip() if body.model is not None else None,
        key=body.key.strip() if body.key else None,
    )}


@router.delete("/api/admin/settings/image-sources/{sid}", dependencies=[Depends(require_admin)])
async def admin_delete_image_source(sid: str) -> dict[str, Any]:
    return {"items": await db.delete_image_extra_source(sid)}


# ----- 用户可切换的图片模型（GPT-Image-2 / Banana Pro 等） -----

class ImageModelBody(BaseModel):
    label: str = Field(..., min_length=1, max_length=60)
    base: str = Field(..., min_length=1, max_length=500)
    model: str = Field(..., min_length=1, max_length=200)
    key: str = Field(..., min_length=1, max_length=5000)
    price_cents: int = Field(..., ge=0, le=1_000_000)            # 1K 单价
    price_2k_cents: int | None = Field(default=None, ge=0, le=1_000_000)
    price_4k_cents: int | None = Field(default=None, ge=0, le=1_000_000)


class ImageModelPatch(BaseModel):
    label: str | None = Field(default=None, max_length=60)
    base: str | None = Field(default=None, max_length=500)
    model: str | None = Field(default=None, max_length=200)
    key: str | None = Field(default=None, max_length=5000)  # 空=保留
    price_cents: int | None = Field(default=None, ge=0, le=1_000_000)
    price_2k_cents: int | None = Field(default=None, ge=0, le=1_000_000)
    price_4k_cents: int | None = Field(default=None, ge=0, le=1_000_000)
    enabled: bool | None = None


@router.get("/api/admin/settings/image-models", dependencies=[Depends(require_admin)])
async def admin_get_image_models() -> dict[str, Any]:
    return {"items": await db.get_image_models(reveal_key=False)}


@router.post("/api/admin/settings/image-models", dependencies=[Depends(require_admin)])
async def admin_add_image_model(body: ImageModelBody) -> dict[str, Any]:
    base = body.base.strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise HTTPException(400, "API Base URL 必须以 http:// 或 https:// 开头")
    label, model, key = body.label.strip(), body.model.strip(), body.key.strip()
    if not (label and model and key):
        raise HTTPException(400, "名称 / 模型 / Key 不能为空")
    return {"items": await db.add_image_model(label=label, base=base, model=model, key=key,
                                              price_cents=body.price_cents,
                                              price_2k_cents=body.price_2k_cents,
                                              price_4k_cents=body.price_4k_cents)}


@router.patch("/api/admin/settings/image-models/{mid}", dependencies=[Depends(require_admin)])
async def admin_update_image_model(mid: str, body: ImageModelPatch) -> dict[str, Any]:
    base = body.base.strip().rstrip("/") if body.base is not None else None
    if base and not base.startswith(("http://", "https://")):
        raise HTTPException(400, "API Base URL 必须以 http:// 或 https:// 开头")
    return {"items": await db.update_image_model(
        mid,
        label=body.label.strip() if body.label is not None else None,
        base=base,
        model=body.model.strip() if body.model is not None else None,
        key=body.key.strip() if body.key else None,
        price_cents=body.price_cents,
        price_2k_cents=body.price_2k_cents,
        price_4k_cents=body.price_4k_cents,
        enabled=body.enabled,
    )}


@router.delete("/api/admin/settings/image-models/{mid}", dependencies=[Depends(require_admin)])
async def admin_delete_image_model(mid: str) -> dict[str, Any]:
    return {"items": await db.delete_image_model(mid)}


@router.get("/api/admin/settings/provider/{name}", dependencies=[Depends(require_admin)])
async def admin_get_provider(name: str) -> dict[str, Any]:
    if name not in db.PROVIDER_NAMES:
        raise HTTPException(404, "未知提供商")
    return await db.get_provider(name, reveal_key=False)


@router.patch("/api/admin/settings/provider/{name}", dependencies=[Depends(require_admin)])
async def admin_update_provider(name: str, body: ProviderSettingsBody) -> dict[str, Any]:
    if name not in db.PROVIDER_NAMES:
        raise HTTPException(404, "未知提供商")
    base = body.base.strip().rstrip("/")
    model = body.model.strip()
    if not base.startswith(("http://", "https://")):
        raise HTTPException(400, "API Base URL 必须以 http:// 或 https:// 开头")
    if not model:
        raise HTTPException(400, "模型名不能为空")
    key = body.key.strip() if body.key is not None else ""
    api_style = body.api_style.strip().lower() if body.api_style is not None else None
    if api_style is not None and api_style not in ("auto", "chat", "responses"):
        raise HTTPException(400, "api_style 必须是 auto / chat / responses")
    return await db.update_provider(name, base=base, model=model, key=key or None, api_style=api_style)


@router.get("/api/admin/payments", dependencies=[Depends(require_admin)])
async def admin_payments(
    limit: int = 100,
    status: str | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    limit = max(1, min(limit, 500))
    if status and status not in ("pending", "paid"):
        status = None
    rows = await db.list_all_payments(limit=limit, status=status, key_prefix=key)
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
                "user_id": r["user_id"],
                "user_key_prefix": r["user_key_prefix"] or "",
                "user_name": r["user_name"] or "",
            }
            for r in rows
        ]
    }


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
