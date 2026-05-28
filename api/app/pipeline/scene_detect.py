"""Detect whether the input is already on a clean studio-like background.

Why: when a dealer uploads a photo that is already on a plain wall, the
segment + compose + shadow path frequently makes things *worse* (a small
mask error that wouldn't matter in a forecourt photo gets dropped onto a
new background and becomes obvious).

If the input passes the studio test we can skip segmentation entirely
and let the harmonize pass do a light polish on the original photo.

The test must be very conservative; missing a true studio is fine,
mis-classifying a forecourt as a studio would be a regression.

Heuristic - all four conditions must hold:

    1. The image's outer 12% border has very low chroma variance
       (background is monochromatic-ish).
    2. The same border has low luma variance (no busy texture, no sky
       gradient).
    3. There are no high-contrast vertical / structural edges in the
       outer border (no buildings / shelves / windows).
    4. The background is bright enough that it could plausibly be a
       studio wall (excludes nighttime forecourts).

The function returns ``True`` only when *all* of these hold.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from ..logging_setup import get_logger

log = get_logger(__name__)


def looks_like_studio(
    image: Image.Image,
    *,
    border_ratio: float = 0.12,
    chroma_std_max: float = 6.0,
    luma_std_max: float = 14.0,
    edge_density_max: float = 0.04,
    min_brightness: float = 90.0,
) -> bool:
    """Return True if ``image`` already looks like a studio shot.

    Defaults are deliberately tight. Bumping them up will increase recall
    but risk false positives on forecourts with grey walls.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    W, H = image.size
    if W < 64 or H < 64:
        return False

    border_w = max(int(W * border_ratio), 8)
    border_h = max(int(H * border_ratio), 8)

    arr = np.asarray(image, dtype=np.float32)

    # Build a boolean border mask: pixels within ``border_w`` of the left
    # or right edge, or ``border_h`` of the top or bottom.
    mask = np.zeros((H, W), dtype=bool)
    mask[:border_h, :] = True
    mask[H - border_h :, :] = True
    mask[:, :border_w] = True
    mask[:, W - border_w :] = True

    border_pixels = arr[mask]
    if border_pixels.size == 0:
        return False

    # 1. Chroma std: how varied is colour, ignoring brightness?
    r, g, b = border_pixels[:, 0], border_pixels[:, 1], border_pixels[:, 2]
    chroma_a = r - g
    chroma_b = 0.5 * (r + g) - b
    chroma_std = float((np.std(chroma_a) + np.std(chroma_b)) / 2.0)

    # 2. Luma std: how varied is brightness?
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    luma_std = float(np.std(luma))
    luma_mean = float(np.mean(luma))

    # 3. Edge density in the border region.
    edges = np.asarray(
        image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.uint8
    )
    border_edges = edges[mask]
    edge_density = float((border_edges > 32).mean())

    decision = (
        chroma_std < chroma_std_max
        and luma_std < luma_std_max
        and edge_density < edge_density_max
        and luma_mean > min_brightness
    )

    log.info(
        "scene_detect.looks_like_studio",
        decision=decision,
        chroma_std=round(chroma_std, 2),
        luma_std=round(luma_std, 2),
        edge_density=round(edge_density, 4),
        luma_mean=round(luma_mean, 1),
    )
    return decision
