"""Arq worker entrypoint.

Run with:
    arq app.worker.WorkerSettings
"""
from __future__ import annotations

from arq.connections import RedisSettings

from .db import get_sessionmaker, init_db
from .errors import StudioError
from .logging_setup import configure_logging, get_logger
from .models import JobStatus
from .pipeline import PipelineParams, run_pipeline
from .repositories import JobRepository
from .settings import get_settings
from .storage import get_storage

log = get_logger(__name__)


async def process_job(ctx: dict, job_id: str) -> dict:
    """Worker entrypoint: load the job, run the pipeline, save the result."""
    storage = get_storage()
    sm = get_sessionmaker()

    async with sm() as session:
        repo = JobRepository(session)
        job = await repo.get(job_id)

        if job.status in {JobStatus.succeeded, JobStatus.cancelled}:
            log.info("worker.skip", job_id=job_id, status=job.status.value)
            return {"skipped": True}

        await repo.mark_running(job_id)

    # Load input bytes from storage outside the DB session
    try:
        input_bytes = await storage.get(job.input_key)
        watermark_bytes: bytes | None = None
        if job.watermark_key:
            try:
                watermark_bytes = await storage.get(job.watermark_key)
            except Exception as exc:  # noqa: BLE001
                log.warning("worker.watermark.missing", error=str(exc))

        params = PipelineParams(
            background_id=job.background_id,
            harmonize=job.harmonize,
            preserve_car=job.preserve_car,
            relight=job.relight,
            plate_blur=job.plate_blur,
            upscale=job.upscale,
            watermark_bytes=watermark_bytes,
            extra_prompt=job.extra_prompt,
        )
        result = await run_pipeline(input_bytes, params)
        output_key = f"outputs/{job_id}.jpg"
        public_url = await storage.put(output_key, result.image_bytes, result.content_type)

    except StudioError as exc:
        async with sm() as session:
            repo = JobRepository(session)
            await repo.mark_failed(job_id, exc.code, exc.message)
        log.warning("worker.job.failed", job_id=job_id, code=exc.code, error=exc.message)
        raise
    except Exception as exc:  # noqa: BLE001
        async with sm() as session:
            repo = JobRepository(session)
            await repo.mark_failed(job_id, "internal_error", str(exc))
        log.exception("worker.job.crashed", job_id=job_id)
        raise

    async with sm() as session:
        repo = JobRepository(session)
        await repo.mark_succeeded(job_id, output_key, public_url, result.timings_ms)

    return {"job_id": job_id, "url": public_url}


async def startup(ctx: dict) -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.app_env != "development")
    await init_db()
    log.info("worker.startup", env=settings.app_env)


async def shutdown(ctx: dict) -> None:
    log.info("worker.shutdown")


class WorkerSettings:
    functions = [process_job]
    on_startup = startup
    on_shutdown = shutdown
    job_timeout = get_settings().job_timeout_seconds
    max_tries = 3
    keep_result = 3600
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
