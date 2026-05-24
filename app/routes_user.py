"""用户侧 API。"""
from __future__ import annotations

import io
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel

from . import db, storage, upstream
from .config import settings
from .deps import require_user

router = APIRouter()

ALLOWED_SIZES = {"auto", "1024x1024", "1024x1536", "1536x1024"}
MAX_REF_BYTES = 10 * 1024 * 1024  # 10MB


class KeyBody(BaseModel):
    access_key: str


@router.post("/api/me")
async def api_me(body: KeyBody) -> dict[str, Any]:
    user = await require_user(body.access_key)
    return {
        "name": user["name"],
        "balance_cents": user["balance_cents"],
        "price_cents": settings.price_cents,
    }


def _normalize_ref(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    # 限制最长边 2048，避免上传超大图
    max_side = 2048
    if max(img.size) > max_side:
        ratio = max_side / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


@router.post("/api/generate")
async def api_generate(
    access_key: str = Form(...),
    prompt: str = Form(...),
    size: str = Form("1024x1024"),
    ref: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    user = await require_user(access_key)
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(400, "prompt 不能为空")
    if size not in ALLOWED_SIZES:
        raise HTTPException(400, f"size 必须是 {sorted(ALLOWED_SIZES)} 之一")

    ref_bytes: bytes | None = None
    ref_key: str | None = None
    if ref is not None and ref.filename:
        raw = await ref.read()
        if not raw:
            ref = None
        elif len(raw) > MAX_REF_BYTES:
            raise HTTPException(400, "参考图超过 10MB")
        else:
            try:
                ref_bytes = _normalize_ref(raw)
            except Exception as e:
                raise HTTPException(400, f"参考图解析失败: {e}")
            ref_key = storage.make_key("refs")
            await storage.upload_bytes(ref_key, ref_bytes)

    has_ref = ref_bytes is not None

    gen_id, balance_after = await db.try_charge_and_create(
        user_id=user["id"],
        prompt=prompt,
        size=size,
        has_ref=has_ref,
        ref_key=ref_key,
        cost_cents=settings.price_cents,
    )
    if gen_id is None:
        raise HTTPException(402, "余额不足")

    try:
        if has_ref and ref_bytes is not None:
            png = await upstream.edit_image(prompt, size, ref_bytes)
        else:
            png = await upstream.generate_image(prompt, size)
        result_key = storage.make_key("results")
        await storage.upload_bytes(result_key, png)
        await db.mark_success(gen_id, result_key)
        return {
            "generation_id": gen_id,
            "result_url": f"/files/result/{gen_id}",
            "balance_cents": balance_after,
        }
    except upstream.UpstreamError as e:
        new_balance = await db.mark_failed_and_refund(
            gen_id, user["id"], settings.price_cents, str(e)
        )
        raise HTTPException(502, f"生成失败: {e}. 已退款。当前余额 {new_balance} 分")
    except Exception as e:
        new_balance = await db.mark_failed_and_refund(
            gen_id, user["id"], settings.price_cents, str(e)
        )
        raise HTTPException(500, f"内部错误: {e}. 已退款。当前余额 {new_balance} 分")


@router.get("/api/history")
async def api_history(access_key: str = Query(...)) -> dict[str, Any]:
    user = await require_user(access_key)
    rows = await db.list_history(user["id"], limit=30)
    out = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "prompt": r["prompt"],
                "size": r["size"],
                "has_ref": r["has_ref"],
                "status": r["status"],
                "error": r["error"],
                "cost_cents": r["cost_cents"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "result_url": f"/files/result/{r['id']}" if r["result_key"] else None,
                "ref_url": f"/files/ref/{r['id']}" if r["ref_key"] else None,
            }
        )
    return {"items": out}


@router.get("/files/{kind}/{generation_id}")
async def get_file(kind: str, generation_id: int, access_key: str = Query(...)):
    if kind not in ("ref", "result"):
        raise HTTPException(404)
    user = await require_user(access_key)
    gen = await db.get_generation(generation_id)
    if not gen or gen["user_id"] != user["id"]:
        raise HTTPException(404)
    key = gen["ref_key"] if kind == "ref" else gen["result_key"]
    if not key:
        raise HTTPException(404)
    body, ctype = await storage.fetch_object(key)
    return StreamingResponse(io.BytesIO(body), media_type=ctype)
