"""Arq queue integration."""
from __future__ import annotations

from typing import Optional

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from .settings import get_settings


def _redis_settings() -> RedisSettings:
    settings = RedisSettings.from_dsn(get_settings().redis_url)
    # Fail fast so the inline fallback kicks in immediately when Redis isn't running.
    settings.conn_retries = 1
    settings.conn_timeout = 1
    return settings


_pool: Optional[ArqRedis] = None


async def get_pool() -> ArqRedis:
    global _pool
    if _pool is None:
        _pool = await create_pool(_redis_settings())
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def enqueue_job(job_id: str) -> None:
    pool = await get_pool()
    await pool.enqueue_job("process_job", job_id, _job_id=f"job:{job_id}")
