"""Persistence helpers for jobs."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .errors import NotFoundError
from .models import Job, JobStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_aware(dt: datetime | None) -> datetime | None:
    """SQLite drops tz info on round-trip. Re-attach UTC so we can subtract safely."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class JobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, **fields) -> Job:
        job = Job(**fields)
        self.session.add(job)
        await self.session.commit()
        await self.session.refresh(job)
        return job

    async def get(self, job_id: str) -> Job:
        job = await self.session.get(Job, job_id)
        if not job:
            raise NotFoundError(f"Job {job_id} not found")
        return job

    async def get_optional(self, job_id: str) -> Optional[Job]:
        return await self.session.get(Job, job_id)

    async def find_succeeded_by_hash(self, input_hash: str, params_hash: str) -> Optional[Job]:
        stmt = select(Job).where(
            Job.input_hash == input_hash,
            Job.params_hash == params_hash,
            Job.status == JobStatus.succeeded,
        ).limit(1)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def find_existing_by_hash(self, input_hash: str, params_hash: str) -> Optional[Job]:
        """Return any reusable job (pending, running, or succeeded) for this hash pair.

        Failed or cancelled jobs are intentionally excluded so retries are allowed.
        Prefers the most recently created reusable job.
        """
        stmt = (
            select(Job)
            .where(
                Job.input_hash == input_hash,
                Job.params_hash == params_hash,
                Job.status.in_(
                    [JobStatus.pending, JobStatus.running, JobStatus.succeeded]
                ),
            )
            .order_by(Job.created_at.desc())
            .limit(1)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def mark_running(self, job_id: str) -> None:
        job = await self.get(job_id)
        job.status = JobStatus.running
        job.started_at = _utcnow()
        job.attempts += 1
        await self.session.commit()

    async def mark_succeeded(
        self,
        job_id: str,
        output_key: str,
        result_url: str,
        timings_ms: dict[str, int],
    ) -> None:
        job = await self.get(job_id)
        job.status = JobStatus.succeeded
        job.finished_at = _utcnow()
        started = _to_aware(job.started_at)
        if started:
            job.duration_ms = int((job.finished_at - started).total_seconds() * 1000)
        job.output_key = output_key
        job.result_url = result_url
        job.stage_timings = timings_ms
        await self.session.commit()

    async def mark_failed(self, job_id: str, code: str, message: str) -> None:
        job = await self.get(job_id)
        job.status = JobStatus.failed
        job.finished_at = _utcnow()
        started = _to_aware(job.started_at)
        if started:
            job.duration_ms = int((job.finished_at - started).total_seconds() * 1000)
        job.error_code = code
        job.error_message = message[:1000]
        await self.session.commit()
