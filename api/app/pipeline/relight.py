"""Optional relighting via Replicate IC-Light. Best-effort: failures are non-fatal."""
from __future__ import annotations

import io

from PIL import Image

from ..logging_setup import get_logger
from ..settings import get_settings
from . import replicate_client

log = get_logger(__name__)


_PROMPTS: dict[str, str] = {
    "studio-gray": "professional automotive studio lighting, soft key light, neutral fill, clean cyc",
    "studio-dark": "moody automotive studio, low key lighting, rim light on bodywork, dark cyc",
    "showroom": "warm dealership showroom lighting, overhead softboxes, polished floor",
    "sunset-road": "golden hour sunlight, warm rim light, long shadows, asphalt foreground",
    "forest": "soft overcast daylight, cool ambient, subtle dappled highlights",
}


async def relight(image: Image.Image, background_id: str) -> Image.Image:
    """Send the composite to IC-Light. Returns the original image on any failure."""
    settings = get_settings()
    if not settings.replicate_enabled:
        return image

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    buf.seek(0)

    prompt = _PROMPTS.get(background_id, "professional automotive lighting")
    try:
        output = await replicate_client.run_model(
            settings.replicate_relight,
            {
                "subject_image": buf,
                "prompt": prompt,
                "image_width": image.width,
                "image_height": image.height,
            },
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("relight.skipped", error=str(exc))
        return image

    url = output if isinstance(output, str) else output[0]
    try:
        out_img = await replicate_client.download_image(url)
    except Exception as exc:  # noqa: BLE001
        log.warning("relight.download_failed", error=str(exc))
        return image
    return out_img.convert("RGB")
