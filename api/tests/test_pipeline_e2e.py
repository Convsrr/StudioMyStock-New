from __future__ import annotations

import io

import pytest
from PIL import Image

from app.pipeline import PipelineParams, run_pipeline


@pytest.mark.asyncio
async def test_pipeline_runs_end_to_end_without_replicate(car_jpeg_bytes: bytes) -> None:
    params = PipelineParams(background_id="studio-white", harmonize=False, relight=False, plate_blur=False, upscale=False)
    result = await run_pipeline(car_jpeg_bytes, params)
    assert result.content_type == "image/jpeg"
    assert result.image_bytes[:3] == b"\xff\xd8\xff"  # JPEG SOI
    assert result.width > 0 and result.height > 0
    # Spot-check decode
    img = Image.open(io.BytesIO(result.image_bytes))
    img.load()
    assert "decode" in result.timings_ms
    assert "compose" in result.timings_ms
