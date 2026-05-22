from __future__ import annotations

from typing import Any

import httpx
import pytest

from app import queue as queue_module
from app.main import app


class _FakePool:
    async def enqueue_job(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
async def _ensure_db_initialized():
    """ASGITransport doesn't run lifespan in httpx 0.27, so initialize the DB explicitly."""
    from app.db import init_db

    await init_db()
    yield


@pytest.fixture(autouse=True)
def _stub_queue(monkeypatch):
    pool = _FakePool()

    async def _get_pool():
        return pool

    monkeypatch.setattr(queue_module, "get_pool", _get_pool)
    yield


@pytest.mark.asyncio
async def test_health_and_backgrounds():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["ok"] is True

        r = await client.get("/api/backgrounds")
        assert r.status_code == 200
        items = r.json()["backgrounds"]
        assert any(b["id"] == "studio-white" for b in items)


@pytest.mark.asyncio
async def test_process_creates_job(car_jpeg_bytes: bytes):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("car.jpg", car_jpeg_bytes, "image/jpeg")}
        data = {"background": "studio-white", "harmonize": "false", "relight": "false"}
        r = await client.post("/api/process", files=files, data=data)
        assert r.status_code == 200, r.text
        body = r.json()
        assert "job_id" in body
        assert body["status"] == "pending"


@pytest.mark.asyncio
async def test_process_rejects_unknown_background(car_jpeg_bytes: bytes):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("car.jpg", car_jpeg_bytes, "image/jpeg")}
        data = {"background": "does-not-exist"}
        r = await client.post("/api/process", files=files, data=data)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "validation_error"
