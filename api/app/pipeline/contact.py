"""Detect the wheel contact line of a car cutout.

The cutout's bbox bottom is rarely the actual ground-touch point: it might
be the lowest pixel of a low bumper, exhaust tip or alpha noise. We need
the y where the rubber meets the road.

Strategy (in order, first reliable signal wins):

    1. Two-blob bottom detector. Run a connected-components pass on the
       bottom 30% of the cutout's alpha and look for two roughly
       symmetric "wheel" blobs left and right of centre. If both are
       found, the contact y is the max of their bottoms. This handles
       3/4 angles, low riders and trucks.

    2. Row-width plateau. Scan upward from the bottom of the cutout
       and find the lowest y where the silhouette width is within 95%
       of its maximum. This is the original heuristic. It works on
       straight-on shots where both wheels are visible at the same y.

    3. Bottom of bbox, with a small margin. Last-resort fallback so
       the pipeline never fails to anchor the car.

The returned y is bounded to ``[H//2, H-1]`` to defeat pathological
cases where the silhouette happens to be widest near the roof (e.g. a
truck shot from above).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from ..logging_setup import get_logger

log = get_logger(__name__)


def find_wheel_contact_y(cutout: Image.Image, alpha_threshold: int = 32) -> int:
    """Return the y (in cutout coordinates) where the wheels touch the ground.

    The cutout is assumed to be trimmed to its bbox so y=0 is the top of
    the car and y=H-1 is the lowest pixel of any part of the car.
    """
    if cutout.mode != "RGBA":
        cutout = cutout.convert("RGBA")
    alpha = np.asarray(cutout.split()[-1], dtype=np.uint8)
    H, W = alpha.shape

    if H < 4 or W < 4:
        return H - 1

    binary = (alpha >= alpha_threshold).astype(np.uint8)
    if binary.sum() == 0:
        return H - 1

    # Strategy 1: two-blob wheel detector
    contact_two = _two_wheel_contact_y(binary)
    if contact_two is not None:
        log.info("contact.detected_two_wheels", y=contact_two, h=H)
        return _bound(contact_two, H)

    # Strategy 2: row-width plateau (original heuristic)
    contact_plateau = _row_width_plateau(binary)
    if contact_plateau is not None:
        log.info("contact.detected_row_plateau", y=contact_plateau, h=H)
        return _bound(contact_plateau, H)

    # Strategy 3: fall through to lowest opaque row
    rows = binary.any(axis=1)
    last_opaque = int(np.argmax(rows[::-1]))  # 0 means H-1 is opaque
    contact_fallback = H - 1 - last_opaque
    log.info("contact.fallback_bbox_bottom", y=contact_fallback, h=H)
    return _bound(contact_fallback, H)


def _bound(y: int, H: int) -> int:
    """Keep contact y in the lower half of the cutout."""
    return max(min(y, H - 1), H // 2)


def _row_width_plateau(binary: np.ndarray) -> int | None:
    """Original heuristic - find the lowest row near the silhouette's maximum width."""
    H, _ = binary.shape
    row_widths = binary.sum(axis=1)
    if not row_widths.any():
        return None

    max_width = int(row_widths.max())
    if max_width == 0:
        return None

    threshold = int(max_width * 0.95)
    bottom_third_start = int(H * 0.55)

    for y in range(H - 1, bottom_third_start - 1, -1):
        if row_widths[y] >= threshold:
            return y
    return None


def _two_wheel_contact_y(binary: np.ndarray) -> int | None:
    """Look for two wheel-shaped blobs in the bottom 30% of the silhouette.

    Returns the y of the highest-bottomed of the two blobs (whichever
    wheel is closer to the camera) so the car gets anchored to the
    closest tyre's contact line, which is what looks right.
    """
    H, W = binary.shape
    bottom_start = int(H * 0.70)
    if bottom_start >= H - 2:
        return None

    region = binary[bottom_start:H, :]
    if region.size == 0 or region.sum() == 0:
        return None

    # Trim columns where the bottom region is empty so a long bumper
    # doesn't merge the two wheels into one component.
    column_has_pixels = region.any(axis=0)
    # Erode horizontally a touch: a single-pixel-wide bridge of bumper
    # would otherwise connect both wheels.
    eroded = region.copy()
    eroded[:, 1:-1] = region[:, 1:-1] & region[:, :-2] & region[:, 2:]
    if eroded.sum() == 0:
        return None

    components = _connected_components(eroded)
    if len(components) < 2:
        return None

    # Sort components by area, take the two largest, then sort those by x.
    components.sort(key=lambda c: -c[4])
    top_two = components[:2]
    top_two.sort(key=lambda c: c[0])
    left, right = top_two

    # Sanity: the two blobs should be roughly the same size and on
    # opposite sides of centre.
    lcx = (left[0] + left[2]) / 2.0
    rcx = (right[0] + right[2]) / 2.0
    if not (lcx < W * 0.55 and rcx > W * 0.45):
        return None
    if rcx - lcx < W * 0.20:
        return None
    larea = left[4]
    rarea = right[4]
    smaller, larger = sorted((larea, rarea))
    if smaller < larger * 0.25:
        # Very asymmetric: probably one wheel + a bumper edge, not two wheels
        return None

    # Translate component-y back to cutout coords.
    contact_local = max(left[3], right[3])  # max(y1) - bottom of the two blobs
    return bottom_start + contact_local


def _connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """4-connectivity components on a small binary mask. Returns
    ``(x0, y0, x1, y1, area)`` per component (x1, y1 exclusive).
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
