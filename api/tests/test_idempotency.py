"""Idempotency / duplicate-job tests for POST /api/process.

These tests verify the four cases called out in the task spec:

    a) same image + same params while pending  -> same job_id
    b) same image + same params while running  -> same job_id
    c) same image + same params after succeeded -> same job_id (cached)
    d) failed jobs do not block retry           -> new job_id
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest

from app import queue as queue_module
from app.db import get_sessionmaker, init_db
from app.main import app
from app.models import JobStatus
from app.repositories import JobRepository


class _FakePool:
    async def enqueue_job(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
async def _ensure_db_initialized():
    await init_db()
    yield


@pytest.fixture(autouse=True)
def _stub_queue(monkeypatch):
    """Replace Redis pool and prevent inline processing so jobs stay in their
    chosen state for the duration of the test."""
    pool = _FakePool()

    async def _get_pool():
        return pool

    async def _enqueue_job(_job_id: str) -> None:
        # Do nothing. We want the job to remain pending.
        return None

    async def _enqueue_or_inline(_job_id: str, _bg) -> None:
        return None

    monkeypatch.setattr(queue_module, "get_pool", _get_pool)
    monkeypatch.setattr(queue_module, "enqueue_job", _enqueue_job)

    # Block the BackgroundTasks fallback path too.
    from app import main as main_module

    monkeypatch.setattr(main_module, "enqueue_job_or_inline", _enqueue_or_inline)
    yield


async def _post_process(client: httpx.AsyncClient, body: bytes, **overrides: str) -> dict:
    files = {"file": ("car.jpg", body, "image/jpeg")}
    data = {
        "background": "studio-white",
        "harmonize": "false",
        "preserve_car": "false",
        "relight": "false",
        "plate_blur": "false",
        "upscale": "false",
    }
    data.update(overrides)
    r = await client.post("/api/process", files=files, data=data)
    assert r.status_code == 200, r.text
    return r.json()


async def _set_status(job_id: str, status: JobStatus) -> None:
    sm = get_sessionmaker()
    async with sm() as session:
        repo = JobRepository(session)
        job = await repo.get(job_id)
        job.status = status
        await session.commit()


@pytest.mark.asyncio
async def test_duplicate_while_pending_returns_same_job(car_jpeg_bytes: bytes):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await _post_process(client, car_jpeg_bytes)
        second = await _post_process(client, car_jpeg_bytes)

        assert first["job_id"] == second["job_id"]
        assert first["status"] == "pending"
        assert second["status"] == "pending"
        assert second["cached"] is True
        # An in-flight (pending) job is "reused", not a cached succeeded result.
        assert second["reused"] is True


@pytest.mark.asyncio
async def test_duplicate_while_running_returns_same_job(car_jpeg_bytes: bytes):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await _post_process(client, car_jpeg_bytes, background="studio-grey")
        await _set_status(first["job_id"], JobStatus.running)

        second = await _post_process(client, car_jpeg_bytes, background="studio-grey")
        assert first["job_id"] == second["job_id"]
        assert second["status"] == "running"
        assert second["cached"] is True
        assert second["reused"] is True


@pytest.mark.asyncio
async def test_duplicate_after_succeeded_returns_cached(car_jpeg_bytes: bytes):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await _post_process(client, car_jpeg_bytes, background="studio-charcoal")
        await _set_status(first["job_id"], JobStatus.succeeded)

        second = await _post_process(client, car_jpeg_bytes, background="studio-charcoal")
        assert first["job_id"] == second["job_id"]
        assert second["status"] == "succeeded"
        assert second["cached"] is True
        # Succeeded result -> cached, not "reused in-flight".
        assert second["reused"] is False


@pytest.mark.asyncio
async def test_failed_job_does_not_block_retry(car_jpeg_bytes: bytes):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await _post_process(client, car_jpeg_bytes, background="studio-warm")
        await _set_status(first["job_id"], JobStatus.failed)

        second = await _post_process(client, car_jpeg_bytes, background="studio-warm")
        assert second["job_id"] != first["job_id"]
        assert second["cached"] is False
        assert second["reused"] is False


@pytest.mark.asyncio
async def test_cancelled_job_does_not_block_retry(car_jpeg_bytes: bytes):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await _post_process(client, car_jpeg_bytes, background="studio-blueprint")
        await _set_status(first["job_id"], JobStatus.cancelled)

        second = await _post_process(client, car_jpeg_bytes, background="studio-blueprint")
        assert second["job_id"] != first["job_id"]
        assert second["cached"] is False
        assert second["reused"] is False


@pytest.mark.asyncio
async def test_different_params_creates_new_job(car_jpeg_bytes: bytes):
    """Sanity: only same input + same params should be deduped."""
    # Slight byte-level change so we get a fresh input_hash that hasn't been
    # used by other tests in this session.
    unique_body = car_jpeg_bytes + b"\x00"
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await _post_process(client, unique_body, background="studio-white")
        # Different background -> different params hash -> different job.
        second = await _post_process(client, unique_body, background="studio-grey")
        assert first["job_id"] != second["job_id"]
        assert second["cached"] is False
