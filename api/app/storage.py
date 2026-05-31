"""Storage abstraction. Backends: local disk, S3-compatible (R2, MinIO, AWS).

The interface is intentionally narrow:
    - put(key, data, content_type) -> public URL
    - get(key) -> bytes
    - exists(key) -> bool
    - public_url(key) -> str

Keys look like ``uploads/<id>.jpg`` or ``outputs/<id>.jpg``.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .settings import Settings, get_settings


class Storage(ABC):
    @abstractmethod
    async def put(self, key: str, data: bytes, content_type: str) -> str: ...

    @abstractmethod
    async def get(self, key: str) -> bytes: ...

    @abstractmethod
    async def exists(self, key: str) -> bool: ...

    @abstractmethod
    def public_url(self, key: str) -> str: ...


class LocalStorage(Storage):
    def __init__(self, root: Path, public_base_url: str) -> None:
        self.root = root
        self.public_base_url = public_base_url.rstrip("/")
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Reject path traversal
        if ".." in Path(key).parts:
            raise ValueError("invalid key")
        return self.root / key

    async def put(self, key: str, data: bytes, content_type: str) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, data)
        return self.public_url(key)

    async def get(self, key: str) -> bytes:
        return await asyncio.to_thread(self._path(key).read_bytes)

    async def exists(self, key: str) -> bool:
        return await asyncio.to_thread(self._path(key).exists)

    def public_url(self, key: str) -> str:
        return f"{self.public_base_url}/static/{key}"


class S3Storage(Storage):
    def __init__(self, settings: Settings) -> None:
        import boto3

        self.bucket = settings.s3_bucket
        self.public_base = (settings.s3_public_base_url or "").rstrip("/")
        self.presigned_ttl = settings.s3_presigned_url_ttl_seconds
        self._client = boto3.client(
            "s3",
            region_name=settings.s3_region or None,
            endpoint_url=settings.s3_endpoint_url or None,
            aws_access_key_id=settings.s3_access_key_id or None,
            aws_secret_access_key=settings.s3_secret_access_key or None,
        )

    async def put(self, key: str, data: bytes, content_type: str) -> str:
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            CacheControl="public, max-age=31536000, immutable",
        )
        return self.public_url(key)

    async def get(self, key: str) -> bytes:
        def _get() -> bytes:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()

        return await asyncio.to_thread(_get)

    async def exists(self, key: str) -> bool:
        def _head() -> bool:
            try:
                self._client.head_object(Bucket=self.bucket, Key=key)
                return True
            except Exception:  # noqa: BLE001
                return False

        return await asyncio.to_thread(_head)

    def public_url(self, key: str) -> str:
        if self.public_base:
            return f"{self.public_base}/{key}"
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self.presigned_ttl,
        )


_storage: Optional[Storage] = None


def get_storage() -> Storage:
    global _storage
    if _storage is not None:
        return _storage
    settings = get_settings()
    if settings.storage_backend == "s3":
        if not settings.s3_bucket:
            raise RuntimeError("STORAGE_BACKEND=s3 requires S3_BUCKET")
        _storage = S3Storage(settings)
    else:
        if settings.app_env == "production" and not settings.allow_public_local_storage:
            raise RuntimeError(
                "Local static storage is public and disabled in production. "
                "Use STORAGE_BACKEND=s3 or set ALLOW_PUBLIC_LOCAL_STORAGE=true."
            )
        _storage = LocalStorage(settings.storage_dir, settings.public_base_url)
    return _storage


def reset_storage_for_tests() -> None:
    global _storage
    _storage = None
