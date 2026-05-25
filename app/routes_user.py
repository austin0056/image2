"""用户侧 API。"""
from __future__ import annotations

import io
import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel

from . import db, storage, upstream, upstream_claude, upstream_recraft
from .config import settings
from .deps import require_user

log = logging.getLogger("image2.user")
router = APIRouter()

ALLOWED_QUALITY = {"auto", "low", "medium", "high"}
MAX_REF_BYTES = 10 * 1024 * 1024  # 10MB

# gpt-image-2 尺寸规则：长宽均为 16 的倍数；最短边 ≥ 256；最长边 ≤ 4096。
SIZE_MIN = 256
SIZE_MAX = 4096
SIZE_MULTIPLE = 16


def _validate_size(size: str) -> str:
    s = size.strip().lower()
    if s == "auto":
        return "auto"
    if "x" not in s:
        raise HTTPException(400, "size 格式必须为 WIDTHxHEIGHT，比如 1024x1024")
    try:
        w_str, h_str = s.split("x", 1)
        w, h = int(w_str), int(h_str)
    except ValueError:
        raise HTTPException(400, "size 解析失败")
    if w < SIZE_MIN or h < SIZE_MIN:
        raise HTTPException(400, f"尺寸过小，宽高都需 ≥ {SIZE_MIN}")
    if w > SIZE_MAX or h > SIZE_MAX:
        raise HTTPException(400, f"尺寸过大，宽高都需 ≤ {SIZE_MAX}")
    if w % SIZE_MULTIPLE or h % SIZE_MULTIPLE:
        raise HTTPException(400, f"尺寸必须是 {SIZE_MULTIPLE} 的倍数")
    return f"{w}x{h}"


class KeyBody(BaseModel):
    access_key: str


@router.post("/api/me")
async def api_me(body: KeyBody) -> dict[str, Any]:
    user = await require_user(body.access_key)
    return {
        "name": user["name"],
        "balance_cents": user["balance_cents"],
        "price_cents": settings.price_cents,
        "price_recraft_cents": settings.price_recraft_cents,
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
    quality: str = Form("high"),
    ref: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    user = await require_user(access_key)
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(400, "prompt 不能为空")
    size = _validate_size(size)
    if quality not in ALLOWED_QUALITY:
        raise HTTPException(400, f"quality 必须是 {sorted(ALLOWED_QUALITY)} 之一")

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
            png = await upstream.edit_image(prompt, size, ref_bytes, quality=quality)
        else:
            png = await upstream.generate_image(prompt, size, quality=quality)
        result_key = storage.make_key("results")
        await storage.upload_bytes(result_key, png)
        await db.mark_success(gen_id, result_key)
        return {
            "generation_id": gen_id,
            "result_url": f"/files/result/{gen_id}",
            "balance_cents": balance_after,
        }
    except upstream.UpstreamError as e:
        log.warning("generate 上游失败 user=%s gen=%s: %s", user["id"], gen_id, e)
        new_balance = await db.mark_failed_and_refund(
            gen_id, user["id"], settings.price_cents, str(e)
        )
        raise HTTPException(502, f"生成失败: {e}. 已退款。当前余额 {new_balance} 分")
    except Exception as e:
        log.exception("generate 内部异常 user=%s gen=%s", user["id"], gen_id)
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
        kind = r.get("kind") or "image"
        item = {
            "id": r["id"],
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
            item["result_url"] = f"/icons/{r['id']}.svg" if r["result_svg"] else None
            item["ref_url"] = None
        else:
            item["result_url"] = f"/files/result/{r['id']}" if r["result_key"] else None
            item["ref_url"] = f"/files/ref/{r['id']}" if r["ref_key"] else None
        out.append(item)
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


@router.delete("/api/generations/{generation_id}")
async def api_delete_generation(generation_id: int, access_key: str = Query(...)) -> dict[str, Any]:
    user = await require_user(access_key)
    deleted = await db.delete_generation(generation_id, user_id=user["id"])
    if not deleted:
        raise HTTPException(404, "记录不存在或不属于你")
    keys = [k for k in (deleted.get("ref_key"), deleted.get("result_key")) if k]
    if keys:
        await storage.delete_keys(keys)
    return {"ok": True}


# ---------- 图标生成 ----------

ALLOWED_ICON_LIBRARIES = {
    "auto", "lucide", "heroicons-outline", "heroicons-solid",
    "phosphor", "tabler", "feather", "material", "duotone",
}
ALLOWED_ENGINES = {"claude", "recraft"}
RECRAFT_STYLES = set(upstream_recraft.RECRAFT_STYLES.keys())


@router.post("/api/generate-icon")
async def api_generate_icon(
    access_key: str = Form(...),
    prompt: str = Form(...),
    library: str = Form("lucide"),
    color: str = Form(""),                    # 兼容旧字段
    color_primary: str = Form(""),
    color_secondary: str = Form(""),
    stroke_width: float | None = Form(default=None),
    engine: str = Form("claude"),
    recraft_style: str = Form(upstream_recraft.DEFAULT_STYLE),
) -> dict[str, Any]:
    user = await require_user(access_key)
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(400, "prompt 不能为空")
    if engine not in ALLOWED_ENGINES:
        raise HTTPException(400, f"engine 必须是 {sorted(ALLOWED_ENGINES)} 之一")
    if engine == "claude" and library not in ALLOWED_ICON_LIBRARIES:
        raise HTTPException(400, f"library 必须是 {sorted(ALLOWED_ICON_LIBRARIES)} 之一")
    if engine == "recraft" and recraft_style not in RECRAFT_STYLES:
        raise HTTPException(400, f"recraft_style 不合法")
    primary = (color_primary or color or "").strip()[:32]
    secondary = (color_secondary or "").strip()[:32]
    if stroke_width is not None and not (0.5 <= stroke_width <= 8.0):
        raise HTTPException(400, "stroke_width 范围 0.5–8")

    # 价格与标签按引擎区分
    if engine == "recraft":
        cost = settings.price_recraft_cents
        style_label = f"recraft {recraft_style}"
    else:
        cost = settings.price_cents
        style_label = library
    if primary:
        style_label += f" P{primary}"
    if secondary:
        style_label += f" S{secondary}"
    if stroke_width and engine == "claude":
        style_label += f" sw{stroke_width}"

    gen_id, balance_after = await db.try_charge_and_create(
        user_id=user["id"],
        prompt=prompt,
        size=style_label[:64],
        has_ref=False,
        ref_key=None,
        cost_cents=cost,
        kind="icon",
    )
    if gen_id is None:
        raise HTTPException(402, "余额不足")

    try:
        if engine == "recraft":
            svg, meta = await upstream_recraft.generate_vector(
                prompt,
                style=recraft_style,
                color_primary=primary,
                color_secondary=secondary,
            )
            warnings: list[str] = []
            samples: list[str] = []
            engine_meta = meta
        else:
            svg, warnings, samples = await upstream_claude.generate_icon_svg(
                prompt,
                library=library,
                color_primary=primary,
                color_secondary=secondary,
                stroke_width=stroke_width,
            )
            engine_meta = {}
        await db.mark_success_svg(gen_id, svg)
        return {
            "generation_id": gen_id,
            "result_url": f"/icons/{gen_id}.svg",
            "balance_cents": balance_after,
            "warnings": warnings,
            "samples": samples,
            "engine": engine,
            "engine_meta": engine_meta,
        }
    except (upstream_claude.ClaudeError, upstream_recraft.RecraftError) as e:
        log.warning(
            "generate-icon 上游失败 user=%s gen=%s engine=%s: %s",
            user["id"], gen_id, engine, e,
        )
        new_balance = await db.mark_failed_and_refund(
            gen_id, user["id"], cost, str(e),
        )
        raise HTTPException(502, f"生成失败：{e}. 已退款。当前余额 {new_balance} 分")
    except Exception as e:
        log.exception("generate-icon 内部异常 user=%s gen=%s", user["id"], gen_id)
        new_balance = await db.mark_failed_and_refund(
            gen_id, user["id"], cost, str(e)
        )
        raise HTTPException(500, f"内部错误：{e}. 已退款。当前余额 {new_balance} 分")


@router.get("/icons/{generation_id}.svg")
async def get_icon_svg(generation_id: int, access_key: str = Query(...)):
    user = await require_user(access_key)
    gen = await db.get_generation(generation_id)
    if not gen or gen["user_id"] != user["id"] or not gen.get("result_svg"):
        raise HTTPException(404)
    return StreamingResponse(
        io.BytesIO(gen["result_svg"].encode("utf-8")),
        media_type="image/svg+xml",
    )
