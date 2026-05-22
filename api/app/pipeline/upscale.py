"""Upscaling via Real-ESRGAN. Optional and best-effort."""
from __future__ import annotations

import io

from PIL import Image

from ..logging_setup import get_logger
from ..settings import get_settings
from . import replicate_client

log = get_logger(__name__)


async def upscale(image: Image.Image, scale: int = 2) -> Image.Image:
    settings = get_settings()
    if not settings.replicate_enabled:
        return image

    target_max = settings.output_max_dim
    if max(image.size) >= target_max:
        return image

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    buf.seek(0)

    try:
        output = await replicate_client.run_model(
            settings.replicate_upscaler,
            {"image": buf, "scale": scale, "face_enhance": False},
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("upscale.skipped", error=str(exc))
        return image

    url = output if isinstance(output, str) else output[0]
    try:
        big = await replicate_client.download_image(url)
    except Exception as exc:  # noqa: BLE001
        log.warning("upscale.download_failed", error=str(exc))
        return image

    big = big.convert("RGB")
    if max(big.size) > target_max:
        big.thumbnail((target_max, target_max), Image.LANCZOS)
    return big
