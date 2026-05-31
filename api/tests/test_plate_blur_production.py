from __future__ import annotations

import pytest
from PIL import Image

from app.errors import PipelineError
from app.pipeline import plate_blur


@pytest.mark.asyncio
async def test_plate_blur_requires_detector_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("REPLICATE_PLATE_DETECTOR", "")
    from app.settings import get_settings

    get_settings.cache_clear()
    with pytest.raises(PipelineError, match="REPLICATE_PLATE_DETECTOR"):
        await plate_blur.blur_plates(Image.new("RGB", (100, 100), "white"))
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_plate_blur_detector_failure_is_visible_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("USE_REPLICATE", "true")
    monkeypatch.setenv("REPLICATE_API_TOKEN", "token")
    monkeypatch.setenv("REPLICATE_PLATE_DETECTOR", "detector:model")
    from app.settings import get_settings

    get_settings.cache_clear()

    async def _detector_fails(_image: Image.Image, _model_ref: str) -> Image.Image:
        raise RuntimeError("detector unavailable")

    monkeypatch.setattr(plate_blur, "_blur_via_replicate", _detector_fails)
    with pytest.raises(PipelineError, match="Production plate detector failed"):
        await plate_blur.blur_plates(Image.new("RGB", (100, 100), "white"))
    get_settings.cache_clear()
