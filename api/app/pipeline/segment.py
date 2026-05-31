"""Background removal. Pluggable provider: Picsart, Replicate, or local fallback."""
from __future__ import annotations

import io

from PIL import Image

from ..errors import ExternalServiceError, PipelineError
from ..logging_setup import get_logger
from ..settings import get_settings
from . import picsart_client, replicate_client

log = get_logger(__name__)


async def segment_car(source: Image.Image) -> Image.Image:
    """Return an RGBA cutout where alpha masks the car."""
    settings = get_settings()
    provider = settings.active_segmentation_provider
    log.info("segment.start", provider=provider, size=source.size)

    if provider == "picsart":
        try:
            return await picsart_client.remove_background(source)
        except ExternalServiceError as exc:
            if settings.segmentation_provider == "auto" and settings.replicate_enabled:
                log.warning("segment.picsart_failed.try_replicate", error=str(exc))
                return await _replicate_segment(source, settings)
            raise
        except Exception as exc:  # noqa: BLE001
            if settings.segmentation_provider == "auto" and settings.replicate_enabled:
                log.warning("segment.picsart_crashed.try_replicate", error=str(exc))
                return await _replicate_segment(source, settings)
            raise PipelineError(f"picsart segmentation failed: {exc}") from exc

    if provider == "replicate":
        return await _replicate_segment(source, settings)

    if settings.app_env == "production":
        raise PipelineError(
            "No segmentation provider is configured. Set PICSART_API_KEY or REPLICATE_API_TOKEN."
        )

    # 'none': local fallback for dev/CI. The output will look wrong but the
    # pipeline still completes so end-to-end tests can run without network.
    log.warning("segment.fallback.no_provider")
    return source.convert("RGBA")


async def _replicate_segment(source: Image.Image, settings) -> Image.Image:
    buf = io.BytesIO()
    source.save(buf, format="PNG")
    buf.seek(0)
    try:
        output = await replicate_client.run_model(
            settings.replicate_bg_remover, {"image": buf}
        )
    except ExternalServiceError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"replicate segmentation failed: {exc}") from exc

    url = output if isinstance(output, str) else output[0]
    cutout = await replicate_client.download_image(url)
    return cutout.convert("RGBA")
