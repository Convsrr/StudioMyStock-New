"""Database models."""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base, TimestampMixin


class JobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class Job(Base, TimestampMixin):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    api_key_hash: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status"), default=JobStatus.pending, nullable=False, index=True
    )

    background_id: Mapped[str] = mapped_column(String(64), nullable=False)
    harmonize: Mapped[bool] = mapped_column(default=True, nullable=False)
    preserve_car: Mapped[bool] = mapped_column(default=True, nullable=False)
    relight: Mapped[bool] = mapped_column(default=False, nullable=False)
    plate_blur: Mapped[bool] = mapped_column(default=False, nullable=False)
    upscale: Mapped[bool] = mapped_column(default=False, nullable=False)
    watermark_key: Mapped[Optional[str]] = mapped_column(String(255))
    extra_prompt: Mapped[Optional[str]] = mapped_column(String(1024))

    input_key: Mapped[str] = mapped_column(String(255), nullable=False)
    output_key: Mapped[Optional[str]] = mapped_column(String(255))
    original_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    result_url: Mapped[Optional[str]] = mapped_column(String(1024))

    input_hash: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    params_hash: Mapped[Optional[str]] = mapped_column(String(64), index=True)

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)

    error_code: Mapped[Optional[str]] = mapped_column(String(64))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    stage_timings: Mapped[Optional[dict]] = mapped_column(JSON)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
