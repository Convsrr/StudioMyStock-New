from __future__ import annotations

import os
from typing import Any

import httpx
import pytest
from PIL import Image

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
    from app.db import init_db

    await init_db()
    yield


@pytest.fixture(autouse=True)
def _stub_queue(monkeypatch):
    pool = _FakePool()

    async def _get_pool():
        return pool

    async def _enqueue_or_inline(_job_id: str, _bg) -> None:
        return None

    monkeypatch.setattr(queue_module, "get_pool", _get_pool)
    from app import main as main_module

    monkeypatch.setattr(main_module, "enqueue_job_or_inline", _enqueue_or_inline)
    yield


async def _post_process(
    client: httpx.AsyncClient,
    body: bytes,
    *,
    watermark: tuple[str, bytes, str] | None = None,
) -> dict:
    files: dict[str, Any] = {"file": ("car.jpg", body, "image/jpeg")}
    if watermark is not None:
        files["watermark"] = watermark
    data = {
        "background": "studio-white",
        "harmonize": "false",
        "preserve_car": "false",
        "relight": "false",
        "plate_blur": "false",
        "upscale": "false",
    }
    r = await client.post("/api/process", files=files, data=data)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.asyncio
async def test_watermark_bytes_are_part_of_idempotency(car_jpeg_bytes: bytes):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await _post_process(
            client,
            car_jpeg_bytes,
            watermark=("wm.png", b"watermark-one", "image/png"),
        )
        second = await _post_process(
            client,
            car_jpeg_bytes,
            watermark=("wm.png", b"watermark-two", "image/png"),
        )
        assert first["job_id"] != second["job_id"]
        assert second["cached"] is False


@pytest.mark.asyncio
async def test_same_watermark_reuses_existing_job(car_jpeg_bytes: bytes):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await _post_process(
            client,
            car_jpeg_bytes,
            watermark=("wm.png", b"same-watermark", "image/png"),
        )
        second = await _post_process(
            client,
            car_jpeg_bytes,
            watermark=("wm.png", b"same-watermark", "image/png"),
        )
        assert first["job_id"] == second["job_id"]
        assert second["cached"] is True


@pytest.mark.asyncio
async def test_no_watermark_and_watermark_are_different_jobs(car_jpeg_bytes: bytes):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await _post_process(client, car_jpeg_bytes)
        second = await _post_process(
            client,
            car_jpeg_bytes,
            watermark=("wm.png", b"new-watermark", "image/png"),
        )
        assert first["job_id"] != second["job_id"]
        assert second["cached"] is False


@pytest.mark.asyncio
async def test_extra_prompt_is_normalized_for_hash(car_jpeg_bytes: bytes):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("car.jpg", car_jpeg_bytes, "image/jpeg")}
        data = {"background": "studio-white", "harmonize": "false", "extra_prompt": "  soft   light  "}
        first = await client.post("/api/process", files=files, data=data)
        assert first.status_code == 200, first.text

        files = {"file": ("car.jpg", car_jpeg_bytes, "image/jpeg")}
        data = {"background": "studio-white", "harmonize": "false", "extra_prompt": "soft light"}
        second = await client.post("/api/process", files=files, data=data)
        assert second.status_code == 200, second.text
        assert first.json()["job_id"] == second.json()["job_id"]


def test_worker_timeout_uses_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "123")
    from app.settings import get_settings

    get_settings.cache_clear()
    import importlib

    import app.worker as worker

    importlib.reload(worker)
    assert worker.WorkerSettings.job_timeout == 123
    get_settings.cache_clear()


def test_background_cache_key_changes_when_asset_mtime_changes(monkeypatch, tmp_path) -> None:
    from app import backgrounds
    from app.settings import get_settings

    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    get_settings.cache_clear()

    asset = tmp_path / "studio-white.jpg"
    img = Image.new("RGB", (10, 10), "white")
    img.save(asset)
    monkeypatch.setattr(backgrounds, "ASSETS_DIR", tmp_path)

    first = backgrounds.get_background("studio-white", (8, 8))
    assert first.size == (8, 8)
    first_cache_files = set((cache_dir / "backgrounds").glob("studio-white_8x8_*.jpg"))
    assert len(first_cache_files) == 1

    img = Image.new("RGB", (10, 10), "gray")
    img.save(asset)
    os.utime(asset, ns=(asset.stat().st_atime_ns + 10_000, asset.stat().st_mtime_ns + 10_000))
    second = backgrounds.get_background("studio-white", (8, 8))
    assert second.size == (8, 8)
    second_cache_files = set((cache_dir / "backgrounds").glob("studio-white_8x8_*.jpg"))
    assert len(second_cache_files) >= 2
    get_settings.cache_clear()
