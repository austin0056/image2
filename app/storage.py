"""MinIO / S3 兼容存储封装。提供建桶、上传、流式下载。"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Iterator

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from .config import settings


def _client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def ensure_bucket() -> None:
    """启动时确保 bucket 存在。"""
    cli = _client()
    try:
        cli.head_bucket(Bucket=settings.s3_bucket)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchBucket", "NotFound"):
            cli.create_bucket(Bucket=settings.s3_bucket)
        else:
            raise


def _today_prefix() -> str:
    d = datetime.now(timezone.utc)
    return f"{d.year:04d}/{d.month:02d}/{d.day:02d}"


def make_key(kind: str) -> str:
    """生成 refs/results 下的对象 key。"""
    assert kind in ("refs", "results")
    return f"{kind}/{_today_prefix()}/{uuid.uuid4().hex}.png"


async def upload_bytes(key: str, data: bytes, content_type: str = "image/png") -> None:
    def _do() -> None:
        _client().put_object(
            Bucket=settings.s3_bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )

    await asyncio.to_thread(_do)


async def fetch_object(key: str) -> tuple[bytes, str]:
    """下载对象，返回 (body, content_type)。"""
    def _do() -> tuple[bytes, str]:
        obj = _client().get_object(Bucket=settings.s3_bucket, Key=key)
        body = obj["Body"].read()
        return body, obj.get("ContentType") or "image/png"

    return await asyncio.to_thread(_do)
