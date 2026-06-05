"""对话式出图响应解析的回归测试（无需 pytest）：python test_image_parser.py

覆盖各家中转/原生 API 的图片返回形态。每加一种新形态（比如某天又冒出个新字段），
在这里补一条用例，避免再次 “出图响应里没找到可用图片”。
"""
import asyncio
import base64
import io
import os

for _k, _v in dict(
    DATABASE_URL="postgres://x", S3_ENDPOINT="http://x", S3_BUCKET="x",
    S3_ACCESS_KEY="x", S3_SECRET_KEY="x", UPSTREAM_KEY="x", ADMIN_PASSWORD="x",
).items():
    os.environ.setdefault(_k, _v)

from PIL import Image  # noqa: E402

from app import upstream as U  # noqa: E402

_buf = io.BytesIO()
Image.new("RGB", (8, 8), (120, 30, 200)).save(_buf, format="PNG")
PNG = _buf.getvalue()
B64 = base64.b64encode(PNG).decode()
DATAURI = "data:image/png;base64," + B64

_failures = 0


def check(name, cond):
    global _failures
    print(("OK   " if cond else "FAIL ") + name)
    if not cond:
        _failures += 1


def collected(p):
    return U._collect_image_urls(p)


def decodes(p):
    return asyncio.run(U._decode_chat_image(None, p))


# --- Gemini 原生 generateContent（用户实际命中的形态）---
check("gemini camelCase", collected({"candidates": [{"avgLogprobs": -1.3, "content": {"parts": [
    {"text": "Image generated successfully."},
    {"inlineData": {"mimeType": "image/png", "data": B64}}]}}]}) == [DATAURI])
check("gemini camelCase decodes", Image.open(io.BytesIO(decodes({"candidates": [{"content": {"parts": [
    {"inlineData": {"mimeType": "image/png", "data": B64}}]}}]}))).size == (8, 8))
check("gemini snake_case", collected({"candidates": [{"content": {"parts": [
    {"inline_data": {"mime_type": "image/png", "data": B64}}]}}]}) == [DATAURI])
check("gemini content-as-list", collected({"candidates": [{"content": [
    {"inlineData": {"data": B64}}]}]}) == [DATAURI])
check("gemini wrapped response{}", collected({"response": {"candidates": [{"content": {"parts": [
    {"inlineData": {"data": B64}}]}}]}}) == [DATAURI])
check("gemini fileData fileUri", collected({"candidates": [{"content": {"parts": [
    {"fileData": {"fileUri": "https://x/img.png"}}]}}]}) == ["https://x/img.png"])
check("gemini text-embedded data-uri", collected({"candidates": [{"content": {"parts": [
    {"text": "result: %s done" % DATAURI}]}}]}) == [DATAURI])

# --- OpenAI 兼容形态（回归保护）---
check("openai message.images", collected({"choices": [{"message": {"images": [
    {"image_url": {"url": DATAURI}}]}}]}) == [DATAURI])
check("openai content[] inline_data", collected({"choices": [{"message": {"content": [
    {"type": "text", "text": "hi"}, {"inlineData": {"data": B64}}]}}]}) == [DATAURI])
check("openai content string markdown", collected({"choices": [{"message": {
    "content": "here ![](%s) done" % DATAURI}}]}) == [DATAURI])
check("top-level data[] b64_json", collected({"data": [{"b64_json": B64}]}) == [DATAURI])

# --- 兜底：拿不到图时干净报错 ---
check("blocked -> []", collected({"candidates": [{"content": {"parts": [{"text": "blocked"}]}}]}) == [])
try:
    decodes({"candidates": [{"content": {"parts": [{"text": "blocked"}]}}]})
    check("blocked decode raises", False)
except U.UpstreamError:
    check("blocked decode raises", True)

print("\n%s" % ("ALL PASS" if _failures == 0 else "%d FAILED" % _failures))
raise SystemExit(1 if _failures else 0)
