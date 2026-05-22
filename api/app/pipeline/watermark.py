"""Optional PNG watermark overlay."""
from __future__ import annotations

import io

from PIL import Image


def apply_watermark(image: Image.Image, watermark_bytes: bytes, opacity: float = 0.7) -> Image.Image:
    """Place a watermark in the bottom-right corner."""
    base = image.convert("RGBA")
    try:
        wm = Image.open(io.BytesIO(watermark_bytes)).convert("RGBA")
    except Exception:  # noqa: BLE001
        return image

    target_w = int(base.width * 0.18)
    ratio = target_w / wm.width
    wm = wm.resize((target_w, max(int(wm.height * ratio), 1)), Image.LANCZOS)

    if opacity < 1.0:
        alpha = wm.split()[-1].point(lambda v: int(v * opacity))
        wm.putalpha(alpha)

    margin = int(base.width * 0.02)
    x = base.width - wm.width - margin
    y = base.height - wm.height - margin
    base.alpha_composite(wm, (x, y))
    return base.convert("RGB")
