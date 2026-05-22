"""License plate blur. Optional. Uses a Replicate detector if configured."""
from __future__ import annotations

import io

from PIL import Image, ImageFilter

from ..logging_setup import get_logger
from ..settings import get_settings
from . import replicate_client

log = get_logger(__name__)


async def blur_plates(image: Image.Image) -> Image.Image:
    """Detect plates and blur them. Returns the original on any failure or if not configured."""
    settings = get_settings()
    if not settings.replicate_enabled or not settings.replicate_plate_detector:
        return image

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    buf.seek(0)

    try:
        result = await replicate_client.run_model(
            settings.replicate_plate_detector, {"image": buf}
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("plate_blur.skipped", error=str(exc))
        return image

    boxes = _extract_boxes(result)
    if not boxes:
        return image

    out = image.copy()
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = _clamp(x1, y1, x2, y2, out.size)
        if x2 <= x1 or y2 <= y1:
            continue
        region = out.crop((x1, y1, x2, y2))
        region = region.filter(ImageFilter.GaussianBlur(radius=max((x2 - x1) // 8, 6)))
        out.paste(region, (x1, y1))
    return out


def _extract_boxes(result: object) -> list[tuple[int, int, int, int]]:
    """Best-effort extraction of bounding boxes from a detector's output JSON.

    Different detectors return different schemas. We try a few common ones.
    """
    if isinstance(result, list) and result and isinstance(result[0], dict):
        boxes = []
        for item in result:
            box = item.get("box") or item.get("bbox")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                boxes.append(tuple(int(v) for v in box))  # type: ignore[arg-type]
        return boxes
    return []


def _clamp(x1: int, y1: int, x2: int, y2: int, size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)
