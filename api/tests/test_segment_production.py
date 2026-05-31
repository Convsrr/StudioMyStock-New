from __future__ import annotations

import pytest
from PIL import Image

from app.errors import ExternalServiceError, PipelineError
from app.pipeline import segment


@pytest.mark.asyncio
async def test_auto_segmentation_falls_back_from_picsart_to_replicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PICSART_API_KEY", "picsart-key")
    monkeypatch.setenv("REPLICATE_API_TOKEN", "replicate-token")
    monkeypatch.setenv("USE_REPLICATE", "true")
    monkeypatch.setenv("SEGMENTATION_PROVIDER", "auto")
    from app.settings import get_settings

    get_settings.cache_clear()

    async def _picsart_fails(_source: Image.Image) -> Image.Image:
        raise ExternalServiceError("picsart unavailable")

    async def _replicate_succeeds(_source: Image.Image, _settings) -> Image.Image:
        return Image.new("RGBA", (4, 4), (255, 0, 0, 255))

    monkeypatch.setattr(segment.picsart_client, "remove_background", _picsart_fails)
    monkeypatch.setattr(segment, "_replicate_segment", _replicate_succeeds)

    out = await segment.segment_car(Image.new("RGB", (4, 4), "white"))
    assert out.mode == "RGBA"
    assert out.getpixel((0, 0)) == (255, 0, 0, 255)
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_unconfigured_segmentation_fails_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("USE_REPLICATE", "false")
    monkeypatch.setenv("REPLICATE_API_TOKEN", "")
    monkeypatch.setenv("PICSART_API_KEY", "")
    monkeypatch.setenv("SEGMENTATION_PROVIDER", "auto")
    from app.settings import get_settings

    get_settings.cache_clear()
    with pytest.raises(PipelineError):
        await segment.segment_car(Image.new("RGB", (4, 4), "white"))
    get_settings.cache_clear()
