"""Pydantic schemas for API request/response payloads."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .models import JobStatus


class BackgroundOut(BaseModel):
    id: str
    name: str
    description: str


class BackgroundsResponse(BaseModel):
    backgrounds: list[BackgroundOut]


class JobCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus
    cached: bool = Field(
        default=False,
        description="True if served from idempotency cache (any reused job: pending, running, or succeeded)",
    )
    reused: bool = Field(
        default=False,
        description="True if an existing pending or running job was reused (vs cached succeeded result)",
    )


class JobOut(BaseModel):
    id: str
    status: JobStatus
    background_id: str
    harmonize: bool
    preserve_car: bool
    relight: bool
    plate_blur: bool
    upscale: bool
    original_url: str
    result_url: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None
    stage_timings: Optional[dict[str, int]] = None
    created_at: datetime
    updated_at: datetime
    finished_at: Optional[datetime] = None


class HealthResponse(BaseModel):
    ok: bool
    replicate_configured: bool
    storage_backend: str
    queue_connected: bool
