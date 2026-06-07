#!/usr/bin/env python3
"""一次性把本地 draw.io webapp 目录上传到 MinIO 的 drawio/ 前缀（用线上同一套 S3 凭证）。

获取 draw.io webapp 静态文件（任选其一）：
  A) git clone --depth 1 -b v24.7.17 https://github.com/jgraph/drawio
     → webapp 根目录在 ./drawio/src/main/webapp
  B) 从 https://github.com/jgraph/drawio/releases 下载源码包，解压后取 src/main/webapp

用法：
  设好环境变量（与 Zeabur 线上一致）：S3_ENDPOINT / S3_ACCESS_KEY / S3_SECRET_KEY / S3_BUCKET / S3_REGION
  python scripts/upload_drawio.py <webapp_dir>
例：
  S3_ENDPOINT=... S3_ACCESS_KEY=... S3_SECRET_KEY=... S3_BUCKET=images \
    python scripts/upload_drawio.py ./drawio/src/main/webapp
"""
from __future__ import annotations

import mimetypes
import os
import sys
from pathlib import Path

import boto3
from botocore.client import Config as BotoConfig

# 显式补全 webapp 常见类型，避免 mimetypes 在某些系统上猜错（尤其 js/wasm/woff2）。
EXTRA = {
    ".js": "application/javascript", ".mjs": "application/javascript",
    ".css": "text/css", ".html": "text/html; charset=utf-8",
    ".json": "application/json", ".map": "application/json",
    ".svg": "image/svg+xml", ".xml": "application/xml",
    ".wasm": "application/wasm",
    ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf", ".eot": "application/vnd.ms-fontobject",
    ".png": "image/png", ".gif": "image/gif", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8", ".md": "text/plain; charset=utf-8",
    ".appcache": "text/cache-manifest",
}


def ctype(p: Path) -> str:
    e = p.suffix.lower()
    return EXTRA.get(e) or mimetypes.guess_type(str(p))[0] or "application/octet-stream"


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/upload_drawio.py <drawio_webapp_dir>")
        sys.exit(2)
    src = Path(sys.argv[1]).resolve()
    if not (src / "index.html").exists():
        print(f"!! {src} 下找不到 index.html —— 确认这是 draw.io webapp 根目录（src/main/webapp）")
        sys.exit(2)

    cli = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    bucket = os.environ.get("S3_BUCKET", "images")

    n = 0
    for f in src.rglob("*"):
        if f.is_dir():
            continue
        rel = f.relative_to(src).as_posix()
        cli.put_object(Bucket=bucket, Key=f"drawio/{rel}", Body=f.read_bytes(), ContentType=ctype(f))
        n += 1
        if n % 100 == 0:
            print(f"  ...{n} files")
    print(f"done: uploaded {n} files -> s3://{bucket}/drawio/")
    print("now open  https://image2.moon9.cloud/drawio/?embed=1&proto=json  to verify")


if __name__ == "__main__":
    main()
