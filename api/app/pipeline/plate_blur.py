"""License plate blur. Optional.

Two-tier detection:

    1. If ``REPLICATE_PLATE_DETECTOR`` is configured, call it (most reliable).
    2. Otherwise fall back to a local heuristic detector that looks for
       bright, plate-shaped regions with dense horizontal edges in the
       lower half of the car box.

The heuristic is conservative on purpose: dealers would rather see an
unblurred plate than a blurred badge or wheel hub. When in doubt, do
nothing and return the original image.

Pass ``car_box`` from the orchestrator to constrain the search and avoid
flagging text or rectangles in the studio background (price tags, exit
signs, etc).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

from ..logging_setup import get_logger
from ..settings import get_settings
from . import replicate_client

log = get_logger(__name__)


# Heuristic detector tuning.
#
# Plates in the working canvas (1920x1280) are typically 80-260 px wide
# and 20-80 px tall. The aspect ratio of EU/UK plates sits around
# 4.5-5.0:1, US plates around 2:1. We accept 1.8:1..6.0:1 to cover both,
# with extra penalty in scoring for ratios further from 4.5.
_MIN_PLATE_W_RATIO = 0.04   # >=4% of canvas width
_MAX_PLATE_W_RATIO = 0.30   # <=30% of canvas width
_MIN_ASPECT = 1.8
_MAX_ASPECT = 6.0
_BRIGHTNESS_THRESHOLD = 165   # 0-255; plates are usually >180
_DARK_PLATE_THRESHOLD = 75    # for black UK rear plates etc
_EDGE_DENSITY_THRESHOLD = 0.18


async def blur_plates(
    image: Image.Image,
    car_box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """Detect plates and blur them.

    Returns the original image on any failure or if nothing plausible is
    detected. ``car_box`` (x, y, w, h) constrains the search; if omitted
    the entire image is searched (more false-positive-prone).
    """
    settings = get_settings()

    # Tier 1: cloud detector (preferred)
    if settings.replicate_enabled and settings.replicate_plate_detector:
        try:
            return await _blur_via_replicate(image, settings.replicate_plate_detector)
        except Exception as exc:  # noqa: BLE001
            log.warning("plate_blur.replicate.failed_falling_back", error=str(exc))

    # Tier 2: local heuristic
    try:
        return _blur_via_heuristic(image, car_box)
    except Exception as exc:  # noqa: BLE001
        log.warning("plate_blur.heuristic.failed", error=str(exc))
        return image


# --- Tier 1: Replicate detector --------------------------------------------


async def _blur_via_replicate(image: Image.Image, model_ref: str) -> Image.Image:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    buf.seek(0)

    result = await replicate_client.run_model(model_ref, {"image": buf})
    boxes = _extract_boxes(result)
    if not boxes:
        return image
    return _apply_blur(image, boxes)


def _extract_boxes(result: object) -> list[tuple[int, int, int, int]]:
    """Best-effort extraction of bounding boxes from a detector's output JSON."""
    if isinstance(result, list) and result and isinstance(result[0], dict):
        boxes: list[tuple[int, int, int, int]] = []
        for item in result:
            box = item.get("box") or item.get("bbox")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                boxes.append(tuple(int(v) for v in box))  # type: ignore[arg-type]
        return boxes
    return []


# --- Tier 2: local heuristic detector --------------------------------------


def _blur_via_heuristic(
    image: Image.Image,
    car_box: tuple[int, int, int, int] | None,
) -> Image.Image:
    """Find plate-like rectangles using brightness + horizontal-edge density."""
    rgb = image.convert("RGB")
    W, H = rgb.size

    # Search region: lower half of the car_box, or the whole image with a
    # bottom-half bias.
    if car_box is not None:
        cx, cy, cw, ch = car_box
        sx0 = max(cx, 0)
        sy0 = max(cy + ch // 2, 0)  # plates are below the car's vertical mid
        sx1 = min(cx + cw, W)
        sy1 = min(cy + ch + 4, H)   # a few pixels past the bumper for clearance
    else:
        sx0, sy0 = 0, H // 2
        sx1, sy1 = W, H

    if sx1 - sx0 < 32 or sy1 - sy0 < 16:
        return image

    region = rgb.crop((sx0, sy0, sx1, sy1))
    rw, rh = region.size

    gray = np.asarray(region.convert("L"), dtype=np.uint8)

    # Two candidate masks: bright (typical EU front / US plates) and dark
    # (rear UK yellow-on-black sometimes; very dirty plates). The dark
    # candidate is much rarer; we still try.
    bright = gray >= _BRIGHTNESS_THRESHOLD
    dark = gray <= _DARK_PLATE_THRESHOLD

    # Horizontal edge map: a plate has dense vertical strokes from digits.
    # Sobel-x via PIL FIND_EDGES + a column-collapse.
    edge_img = region.convert("L").filter(ImageFilter.FIND_EDGES)
    edges = np.asarray(edge_img, dtype=np.uint8) > 50

    candidates: list[tuple[int, int, int, int, float]] = []
    for mask, kind in ((bright, "bright"), (dark, "dark")):
        candidates.extend(_score_components(mask, edges, kind, rw, rh, W))

    if not candidates:
        log.info("plate_blur.heuristic.no_candidates")
        return image

    # Take the top ~2 by score; multiple plates per image are rare but
    # possible (front + rear visible).
    candidates.sort(key=lambda c: -c[4])
    top = candidates[:2]
    boxes_canvas = [
        (sx0 + x0, sy0 + y0, sx0 + x1, sy0 + y1) for x0, y0, x1, y1, _ in top
    ]
    log.info("plate_blur.heuristic.detected", boxes=boxes_canvas)
    return _apply_blur(image, boxes_canvas)


def _score_components(
    mask: np.ndarray,
    edges: np.ndarray,
    kind: str,
    rw: int,
    rh: int,
    canvas_w: int,
) -> list[tuple[int, int, int, int, float]]:
    """Score connected-component candidates from a binary mask.

    Returns (x0, y0, x1, y1, score) tuples in *region-local* coords.
    """
    components = _connected_components(mask)
    out: list[tuple[int, int, int, int, float]] = []
    for x0, y0, x1, y1, area in components:
        w = x1 - x0
        h = y1 - y0
        if w < 8 or h < 4:
            continue
        aspect = w / max(h, 1)
        if aspect < _MIN_ASPECT or aspect > _MAX_ASPECT:
            continue
        w_ratio = w / max(canvas_w, 1)
        if w_ratio < _MIN_PLATE_W_RATIO or w_ratio > _MAX_PLATE_W_RATIO:
            continue

        # Edge density inside the candidate region.
        comp_edges = edges[y0:y1, x0:x1]
        if comp_edges.size == 0:
            continue
        edge_density = float(comp_edges.mean())
        if edge_density < _EDGE_DENSITY_THRESHOLD:
            continue

        # Score: prefer aspect ratio close to 4.5, decent edge density,
        # higher fill ratio, smaller relative width (full-canvas-width
        # candidates are usually walls or floor lines).
        aspect_penalty = abs(aspect - 4.5) / 4.5
        fill = area / float(w * h)
        score = (
            edge_density
            + 0.5 * fill
            - 0.6 * aspect_penalty
            - 0.4 * w_ratio
        )
        # Dark plates are rarer; nudge their score down slightly so a
        # competing bright candidate wins ties.
        if kind == "dark":
            score -= 0.1
        out.append((x0, y0, x1, y1, score))
    return out


def _connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """4-connectivity components on a binary mask.

    Returns ``(x0, y0, x1, y1, area)`` tuples. Plain Python; the masks we
    operate on here are tiny (search region only) so this is fast enough.
    """
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out: list[tuple[int, int, int, int, int]] = []

    # Iterative flood-fill with a stack.
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


# --- Shared blur utility ---------------------------------------------------


def _apply_blur(image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    out = image.convert("RGB").copy()
    W, H = out.size
    for x1, y1, x2, y2 in boxes:
        x1 = max(min(x1, W - 1), 0)
        y1 = max(min(y1, H - 1), 0)
        x2 = max(min(x2, W), x1 + 1)
        y2 = max(min(y2, H), y1 + 1)
        if x2 <= x1 or y2 <= y1:
            continue
        # Pad by ~10% so the blur covers any tight bounding box that
        # missed the very edge of the plate.
        bw, bh = x2 - x1, y2 - y1
        pad_x = max(int(bw * 0.10), 2)
        pad_y = max(int(bh * 0.10), 2)
        x1 = max(x1 - pad_x, 0)
        y1 = max(y1 - pad_y, 0)
        x2 = min(x2 + pad_x, W)
        y2 = min(y2 + pad_y, H)
        region = out.crop((x1, y1, x2, y2))
        radius = max((x2 - x1) // 6, 8)
        region = region.filter(ImageFilter.GaussianBlur(radius=radius))
        out.paste(region, (x1, y1))
    return out
