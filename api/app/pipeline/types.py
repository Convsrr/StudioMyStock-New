"""Shared pipeline types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PIL import Image


@dataclass
class PipelineContext:
    """Carries state across pipeline stages."""

    job_id: str
    background_id: str
    relight: bool = False
    plate_blur: bool = False
    upscale: bool = False
    watermark_bytes: Optional[bytes] = None

    # Set by stages
    source: Optional[Image.Image] = None  # working RGB
    cutout: Optional[Image.Image] = None  # RGBA car cutout (trimmed)
    composite: Optional[Image.Image] = None  # RGB composite
    final: Optional[Image.Image] = None  # RGB output

    timings_ms: dict[str, int] = field(default_factory=dict)
