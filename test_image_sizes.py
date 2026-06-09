"""分辨率档位尺寸的回归测试（无需 pytest）：python test_image_sizes.py

守住上游对出图尺寸的硬约束，避免再次 400：
  - size max side must be <= 3840px
  - size total pixels must be between 655360 and 8294400
每个档位 × 每种比例算出的尺寸都必须落在区间内，且比例不严重失真。
"""
import os

for _k, _v in dict(
    DATABASE_URL="postgres://x", S3_ENDPOINT="http://x", S3_BUCKET="x",
    S3_ACCESS_KEY="x", S3_SECRET_KEY="x", UPSTREAM_KEY="x", ADMIN_PASSWORD="x",
).items():
    os.environ.setdefault(_k, _v)

from app import db  # noqa: E402
from app import upstream  # noqa: E402

MAX_SIDE = 3840
MIN_PIXELS = 655_360
MAX_PIXELS = 8_294_400

failures = 0


def check(name, cond):
    global failures
    print(("OK   " if cond else "FAIL ") + name)
    if not cond:
        failures += 1


for tier in db.IMAGE_TIERS:
    for aspect in db.IMAGE_ASPECTS:
        w, h = map(int, db.tier_size(tier, aspect).split("x"))
        area = w * h
        aw, ah = db._TIER_ASPECT[aspect]
        drift = abs((w / h) - (aw / ah)) / (aw / ah)
        ok = (max(w, h) <= MAX_SIDE
              and MIN_PIXELS <= area <= MAX_PIXELS
              and w % 16 == 0 and h % 16 == 0
              and drift <= 0.06)
        check(f"tier_size {tier} {aspect:>5} -> {w}x{h} (area={area}, drift={drift*100:.1f}%)", ok)

# 真正发给上游的尺寸 = images_api_size(tier_size(...))，必须同样落在区间内
# （历史 bug：images_api_size 把长边 ×1.5，3840→5760，触发 size max side > 3840px）
for tier in db.IMAGE_TIERS:
    for aspect in db.IMAGE_ASPECTS:
        api = upstream.images_api_size(db.tier_size(tier, aspect))
        w, h = map(int, api.split("x"))
        area = w * h
        ok = max(w, h) <= MAX_SIDE and area <= MAX_PIXELS and w % 16 == 0 and h % 16 == 0
        check(f"api_size  {tier} {aspect:>5} -> {api} (area={area})", ok)

# 关键档位应命中标准尺寸
check("4K 16:9 == 3840x2160 (UHD)", db.tier_size("4k", "16:9") == "3840x2160")
check("4K 1:1 area == cap (2880x2880)", db.tier_size("4k", "1:1") == "2880x2880")
check("2K 1:1 == 2048x2048", db.tier_size("2k", "1:1") == "2048x2048")
check("1K 1:1 == 1024x1024", db.tier_size("1k", "1:1") == "1024x1024")
# 兜底：未知档位/比例回落
check("unknown tier -> 1k", db.normalize_tier("8k") == "1k")
check("unknown aspect -> 1:1", db.normalize_aspect("5:1") == "1:1")

print("\n%s" % ("ALL PASS" if failures == 0 else "%d FAILED" % failures))
raise SystemExit(1 if failures else 0)
