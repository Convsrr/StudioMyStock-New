"""Post-harmonize quality checks.

Qwen's harmonize pass is supposed to only adjust lighting/reflections; it
must not introduce a duplicate vehicle. Sometimes it does anyway (a ghost
car drawn next to the real one, an extended bumper, a mirrored silhouette
on the floor). This module provides a cheap deterministic guard that
flags those cases so the orchestrator can fall back to the pre-AI
composite.

Strategy: we trust changes inside (and a small halo around) the original
car region. Outside that safe region we fire on either of two signals:

    1. **New high-contrast edges.** The original guard. Catches sharp
       duplicates with crisp paint lines and trim.
    2. **New dark blob with vehicle-ish shape.** Catches soft / blurry
       duplicates that don't add many edges (a hazy ghost car, an
       extended bumper smeared into the floor). Looks for connected
       regions where the AI noticeably *darkened* a previously bright
       area, with size and aspect ratio in the car-shaped band.

If either signal exceeds its threshold we flag the result. Both
thresholds are tuned to be conservative on subtle lighting changes
(walls getting brighter, soft floor shading appearing) but to fire on
anything that reads as a second vehicle.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from ..logging_setup import get_logger

log = get_logger(__name__)


def _to_gray(img: Image.Image, max_dim: int = 512) -> np.ndarray:
    """Downscale to ``max_dim`` longest edge and return a grayscale uint8 array.

    The downscale keeps the check fast and removes JPEG-noise-level edge
    differences.
    """
    g = img.convert("L")
    if max(g.size) > max_dim:
        scale = max_dim / max(g.size)
        new_size = (max(int(g.width * scale), 1), max(int(g.height * scale), 1))
        g = g.resize(new_size, Image.LANCZOS)
    return np.asarray(g, dtype=np.uint8)


def _edge_magnitude(gray: np.ndarray) -> np.ndarray:
    """Approximate Sobel edge magnitude using PIL's FIND_EDGES + small blur."""
    img = Image.fromarray(gray, mode="L").filter(ImageFilter.FIND_EDGES)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    return np.asarray(img, dtype=np.float32)


def _safe_region_mask(
    shape: tuple[int, int],
    car_box: tuple[int, int, int, int],
    src_size: tuple[int, int],
    halo_ratio: float = 0.15,
) -> np.ndarray:
    """Boolean mask the size of the (downscaled) image. True where we trust
    AI changes (inside car_box + halo); False elsewhere.

    The halo is intentionally smaller than before (15%) so a duplicate
    drawn shoulder-to-shoulder with the real car still falls in the
    unsafe region. The reflection/shadow zone directly under the car is
    handled separately by the dark-blob detector's vehicle-shape filter.
    """
    out_h, out_w = shape
    src_w, src_h = src_size
    x, y, w, h = car_box

    # Scale car_box from source coords to downscaled coords.
    sx = out_w / max(src_w, 1)
    sy = out_h / max(src_h, 1)
    bx0 = int(round(x * sx))
    by0 = int(round(y * sy))
    bx1 = int(round((x + w) * sx))
    by1 = int(round((y + h) * sy))

    # Expand by halo_ratio of the box dimensions.
    pad_x = int((bx1 - bx0) * halo_ratio)
    pad_y = int((by1 - by0) * halo_ratio)
    bx0 = max(bx0 - pad_x, 0)
    by0 = max(by0 - pad_y, 0)
    bx1 = min(bx1 + pad_x, out_w)
    # Extend the safe region a bit further down to cover the deterministic
    # shadow / reflection / chassis AO band. AI is allowed to lightly
    # polish those without firing the guard.
    box_h = by1 - by0
    by1 = min(by1 + int(box_h * 0.20), out_h)

    mask = np.zeros((out_h, out_w), dtype=bool)
    if bx1 > bx0 and by1 > by0:
        mask[by0:by1, bx0:bx1] = True
    return mask


def _connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """4-connectivity components on a small binary mask.

    Returns ``(x0, y0, x1, y1, area)`` per component (x1, y1 exclusive).
    Iterative flood-fill so it can't blow the recursion limit on large
    blobs.
    """
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out: list[tuple[int, int, int, int, int]] = []

    for y in range(H):
        row = mask[y]
        for x in range(W):
            if not row[x] or visited[y, x]:
                continue
            stack = [(x, y)]
            x0 = x1 = x
            y0 = y1 = y
            area = 0
            while stack:
                cx, cy = stack.pop()
                if cx < 0 or cy < 0 or cx >= W or cy >= H:
                    continue
                if visited[cy, cx] or not mask[cy, cx]:
                    continue
                visited[cy, cx] = True
                area += 1
                if cx < x0:
                    x0 = cx
                if cx > x1:
                    x1 = cx
                if cy < y0:
                    y0 = cy
                if cy > y1:
                    y1 = cy
                stack.append((cx + 1, cy))
                stack.append((cx - 1, cy))
                stack.append((cx, cy + 1))
                stack.append((cx, cy - 1))
            out.append((x0, y0, x1 + 1, y1 + 1, area))
    return out


def detect_possible_duplicate_vehicle(
    before_ai: Image.Image,
    after_ai: Image.Image,
    car_box: tuple[int, int, int, int],
    *,
    edge_increase_threshold: float = 18.0,
    blob_area_ratio_threshold: float = 0.012,
    luma_drop_threshold: float = 22.0,
    luma_blob_min_area_ratio: float = 0.005,
    luma_blob_max_area_ratio: float = 0.40,
    halo_ratio: float = 0.15,
    reflection_band_height_ratio: float = 0.22,
) -> bool:
    """Return True if the AI pass appears to have introduced a duplicate
    vehicle outside the original car region.

    Two signals; either one fires the guard:

    - ``edge_increase_threshold`` / ``blob_area_ratio_threshold``:
      pixel-level "new edges in the unsafe region". Catches sharp
      duplicates.
    - ``luma_drop_threshold`` / ``luma_blob_min_area_ratio`` /
      ``luma_blob_max_area_ratio``: connected-component check on
      "regions the AI noticeably darkened that are roughly car-sized".
      Catches soft / blurry duplicates and bumper-extension smears.

    Reflection-band exemption: blobs that sit predominantly inside the
    floor reflection band (``reflection_band_height_ratio`` of the
    car_box height immediately below the car) are skipped because that
    is exactly where the deterministic reflection stage paints its
    output and where Qwen is allowed to lightly polish it.
    """
    if before_ai.size == (0, 0) or after_ai.size == (0, 0):
        return False

    if after_ai.size != before_ai.size:
        after_ai = after_ai.resize(before_ai.size, Image.LANCZOS)

    before_gray = _to_gray(before_ai)
    after_gray = _to_gray(after_ai)
    if before_gray.shape != after_gray.shape:
        return False

    safe = _safe_region_mask(before_gray.shape, car_box, before_ai.size, halo_ratio)
    unsafe = ~safe
    unsafe_area = int(unsafe.sum())
    if unsafe_area < 64:
        return False

    # Reflection band in downscaled coords. Anything whose centroid sits
    # inside this band gets exempted from the dark-blob signal.
    reflection_band = _reflection_band(
        before_gray.shape, car_box, before_ai.size, reflection_band_height_ratio,
    )

    flagged_edge, edge_metrics = _edge_signal(
        before_gray, after_gray, unsafe, edge_increase_threshold,
        blob_area_ratio_threshold,
    )
    flagged_luma, luma_metrics = _luma_signal(
        before_gray, after_gray, unsafe, luma_drop_threshold,
        luma_blob_min_area_ratio, luma_blob_max_area_ratio,
        reflection_band=reflection_band,
    )

    flagged = flagged_edge or flagged_luma

    log_method = log.warning if flagged else log.info
    log_method(
        "quality_guard.result",
        flagged=flagged,
        flagged_edge=flagged_edge,
        flagged_luma=flagged_luma,
        edge_metrics=edge_metrics,
        luma_metrics=luma_metrics,
    )

    return flagged


def _reflection_band(
    shape: tuple[int, int],
    car_box: tuple[int, int, int, int],
    src_size: tuple[int, int],
    height_ratio: float,
) -> tuple[int, int, int, int]:
    """Return the (x0, y0, x1, y1) of the floor reflection band in
    downscaled coords.

    The band sits immediately below the car_box, spans the car's width
    plus a small margin, and is ``height_ratio`` of the car's height
    tall. Any duplicate-car-shaped blob whose centroid falls inside
    this band is exempt from the guard because that's the deterministic
    reflection's territory.
    """
    out_h, out_w = shape
    src_w, src_h = src_size
    x, y, w, h = car_box

    sx = out_w / max(src_w, 1)
    sy = out_h / max(src_h, 1)
    bx0 = max(int(round(x * sx)) - int(round(w * sx * 0.08)), 0)
    bx1 = min(int(round((x + w) * sx)) + int(round(w * sx * 0.08)), out_w)
    by_top = int(round((y + h) * sy))
    band_h = max(int(round(h * sy * height_ratio)), 4)
    by_bottom = min(by_top + band_h, out_h)
    return bx0, by_top, bx1, by_bottom


def _edge_signal(
    before_gray: np.ndarray,
    after_gray: np.ndarray,
    unsafe: np.ndarray,
    edge_increase_threshold: float,
    blob_area_ratio_threshold: float,
) -> tuple[bool, dict]:
    """Original edge-based signal.

    Returns ``(flagged, metrics)``.
    """
    before_edges = _edge_magnitude(before_gray)
    after_edges = _edge_magnitude(after_gray)

    delta = (after_edges - before_edges).clip(min=0.0)
    new_edge_pixels = delta > edge_increase_threshold
    new_edge_in_unsafe = new_edge_pixels & unsafe
    unsafe_area = int(unsafe.sum())
    blob_area = int(new_edge_in_unsafe.sum())
    blob_ratio = blob_area / max(unsafe_area, 1)

    return blob_ratio > blob_area_ratio_threshold, {
        "blob_ratio": round(blob_ratio, 4),
        "threshold": blob_area_ratio_threshold,
        "blob_area": blob_area,
        "unsafe_area": unsafe_area,
    }


def _luma_signal(
    before_gray: np.ndarray,
    after_gray: np.ndarray,
    unsafe: np.ndarray,
    luma_drop_threshold: float,
    min_area_ratio: float,
    max_area_ratio: float,
    *,
    reflection_band: tuple[int, int, int, int] | None = None,
) -> tuple[bool, dict]:
    """Connected-component check on regions the AI noticeably darkened.

    A genuine soft duplicate car shows up as a *connected region* where
    the post-AI image is meaningfully darker than the pre-AI image and
    the region's bounding box has a vehicle-ish aspect ratio.

    We fire the guard if any single dark blob in the unsafe region:

        - covers between ``min_area_ratio`` and ``max_area_ratio`` of
          the unsafe area (filters out tiny noise and image-wide
          changes),
        - has an aspect ratio between 0.7 and 5.0 (vehicles, bumpers,
          extended silhouettes),
        - has reasonable fill (area / bbox >= 0.30) so we don't flag a
          long thin shadow line,
        - has a centroid that falls *outside* the deterministic floor
          reflection band (``reflection_band``) so we don't mistake the
          reflection for a duplicate.
    """
    delta = before_gray.astype(np.int16) - after_gray.astype(np.int16)
    darkened = (delta > luma_drop_threshold) & unsafe
    if darkened.sum() == 0:
        return False, {"reason": "no_dark_pixels"}

    components = _connected_components(darkened)
    if not components:
        return False, {"reason": "no_components"}

    unsafe_area = int(unsafe.sum())
    largest_metrics: dict | None = None

    for x0, y0, x1, y1, area in components:
        w = x1 - x0
        h = y1 - y0
        if w <= 0 or h <= 0:
            continue
        area_ratio = area / max(unsafe_area, 1)
        aspect = w / max(h, 1)
        fill = area / float(w * h)
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        in_reflection = False
        if reflection_band is not None:
            rx0, ry0, rx1, ry1 = reflection_band
            in_reflection = rx0 <= cx < rx1 and ry0 <= cy < ry1
        metrics = {
            "area": int(area),
            "area_ratio": round(area_ratio, 4),
            "w": w,
            "h": h,
            "aspect": round(aspect, 2),
            "fill": round(fill, 2),
            "in_reflection": in_reflection,
        }
        if largest_metrics is None or area > largest_metrics["area"]:
            largest_metrics = metrics
        if in_reflection:
            # Reflection band is owned by the deterministic stage; AI
            # is permitted to lightly polish it.
            continue
        if (
            area_ratio >= min_area_ratio
            and area_ratio <= max_area_ratio
            and 0.7 <= aspect <= 5.0
            and fill >= 0.30
        ):
            return True, {"vehicle_blob": metrics}

    return False, {"largest": largest_metrics or {"reason": "all_below_threshold"}}
