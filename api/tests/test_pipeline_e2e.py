from __future__ import annotations

import io

import pytest
from PIL import Image

from app.pipeline import PipelineParams, run_pipeline


@pytest.mark.asyncio
async def test_pipeline_runs_end_to_end_classical(car_jpeg_bytes: bytes) -> None:
    """Classical path: deterministic compose + shadow + reflection only."""
    params = PipelineParams(
        background_id="studio-white",
        harmonize=False,
        relight=False,
        plate_blur=False,
        upscale=False,
    )
    result = await run_pipeline(car_jpeg_bytes, params)
    assert result.content_type == "image/jpeg"
    assert result.image_bytes[:3] == b"\xff\xd8\xff"  # JPEG SOI
    assert result.width > 0 and result.height > 0
    img = Image.open(io.BytesIO(result.image_bytes))
    img.load()
    assert "decode" in result.timings_ms
    assert "compose" in result.timings_ms
    assert "shadow" in result.timings_ms
    assert "reflection" in result.timings_ms


@pytest.mark.asyncio
async def test_pipeline_runs_end_to_end_harmonize_stub(
    car_jpeg_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AI path with harmonize=True; conftest disables Replicate so harmonize
    just returns the deterministic composite. We still expect the post-AI
    stages (quality_guard, preserve_car) to run.

    The synthetic ``car_jpeg_bytes`` fixture is a flat-coloured image
    that legitimately looks like a studio shot (uniform grey borders),
    so it would trip the studio short-circuit. Disable the short-circuit
    here so we can verify the full pipeline path.
    """
    monkeypatch.setenv("ENABLE_STUDIO_SHORTCIRCUIT", "false")
    from app.settings import get_settings

    get_settings.cache_clear()

    params = PipelineParams(
        background_id="studio-grey",
        harmonize=True,
        preserve_car=True,
    )
    result = await run_pipeline(car_jpeg_bytes, params)
    assert result.content_type == "image/jpeg"
    assert result.image_bytes[:3] == b"\xff\xd8\xff"
    assert "harmonize" in result.timings_ms
    assert "quality_guard" in result.timings_ms
    assert "preserve_car" in result.timings_ms

    get_settings.cache_clear()
