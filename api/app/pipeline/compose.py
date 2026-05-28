"""Place a car cutout onto a background canvas.

Smart placement:
    - Detects the wheel contact y of the cutout (where the tyres meet the ground)
    - Reads the floor-line y of the background preset
    - Picks a landscape, square, or portrait canvas based on the input
      aspect ratio so phone photos don't end up tiny in a wide frame
    - Scales the car to a sensible canvas fraction (different ratios for
      landscape vs portrait so the car doesn't crash into the cyc seam)
    - Anchors so the wheel contact line sits exactly on the floor line

Returns (canvas_rgba, car_box_x_y_w_h, contact_y_in_canvas).
"""
from __future__ import annotations

from typing import Optional

from PIL import Image

from .. import backgrounds
from .contact import find_wheel_contact_y


# Canvas presets. The pipeline picks one of these by closest aspect to
# the input image. Sizes are tuned so each preset still uses the same
# horizontal field of studio "behind" the car.
_LANDSCAPE = (1920, 1280)   # 3:2 - default for forecourt photos
_SQUARE = (1600, 1600)
_PORTRAIT = (1280, 1920)


def _pick_canvas_size(
    source_size: tuple[int, int],
    requested: Optional[tuple[int, int]] = None,
) -> tuple[int, int]:
    """Pick a canvas size that matches the input's orientation.

    If the caller passes ``requested`` we honour it (used by tests).
    """
    if requested is not None:
        return requested
    sw, sh = source_size
    if sh <= 0:
        return _LANDSCAPE
    aspect = sw / sh
    # Bands chosen so a 4:3 phone shot (1.33) still uses landscape, but
    # anything noticeably taller than wide goes portrait.
    if aspect >= 1.20:
        return _LANDSCAPE
    if aspect <= 0.83:
        return _PORTRAIT
    return _SQUARE


def compose(
    cutout: Image.Image,
    background_id: str,
    canvas_size: Optional[tuple[int, int]] = None,
    *,
    source_size: Optional[tuple[int, int]] = None,
) -> tuple[Image.Image, tuple[int, int, int, int], int]:
    """Place ``cutout`` on the studio background.

    Either ``canvas_size`` or ``source_size`` may be passed. If both are
    omitted we fall back to the cutout's own size (so existing tests
    continue to work) and from that pick a canvas band.

    If ``cutout`` carries a ``raw_alpha`` attribute (set by
    ``refine_mask.refine``), the wheel-contact detection runs on the
    raw alpha instead of the post-fill alpha. This lets the detector
    see the gap between the wheels even when interior-hole-filling
    has closed it.
    """
    if canvas_size is None:
        # Prefer source_size when available (the orchestrator passes it
        # in so we honour the *original* photo orientation, not the
        # cropped cutout's). Otherwise infer from the cutout.
        canvas_size = _pick_canvas_size(source_size or cutout.size)

    target_w, target_h = canvas_size
    preset = backgrounds.get_preset(background_id)
    bg = backgrounds.get_background(background_id, (target_w, target_h)).convert("RGBA")

    # Scale the car to fit. Portrait canvases need the car proportionally
    # smaller in width or it crashes into the cyc seam; landscape can be
    # bigger.
    is_portrait = target_h > target_w
    if is_portrait:
        car_max_w = int(target_w * 0.86)
        car_max_h = int(target_h * 0.55)
    else:
        car_max_w = int(target_w * 0.78)
        car_max_h = int(target_h * 0.70)

    car = cutout.copy()
    car.thumbnail((car_max_w, car_max_h), Image.LANCZOS)

    # Find where the wheels meet the ground in the *scaled* cutout. If the
    # refine_mask stage attached a pre-fill ``raw_alpha`` we use it - it
    # preserves the gap between the wheels which the two-blob detector
    # depends on.
    raw_alpha_src = getattr(cutout, "raw_alpha", None)
    if raw_alpha_src is not None:
        # Resize the raw alpha by the same factor as the car was thumbnailed.
        scaled_raw = raw_alpha_src.resize(car.size, Image.LANCZOS)
        contact_y_local = find_wheel_contact_y(_alpha_to_rgba(scaled_raw))
    else:
        contact_y_local = find_wheel_contact_y(car)

    floor_y_canvas = int(target_h * preset.floor_y_ratio)

    # x: center horizontally
    x = (target_w - car.width) // 2
    # y: align contact_y_local of the cutout with floor_y_canvas
    y = floor_y_canvas - contact_y_local

    # Clamp so the car never goes above the top edge or below the canvas
    y = max(min(y, target_h - 1), -car.height // 4)

    canvas = bg.copy()
    canvas.alpha_composite(car, (x, y))
    contact_y_canvas = y + contact_y_local
    return canvas, (x, y, car.width, car.height), contact_y_canvas


def _alpha_to_rgba(alpha: Image.Image) -> Image.Image:
    """Wrap an L-mode alpha as an RGBA image so ``find_wheel_contact_y``
    (which expects RGBA) can read it. The colour channels are zero; only
    alpha is meaningful.
    """
    if alpha.mode != "L":
        alpha = alpha.convert("L")
    rgba = Image.new("RGBA", alpha.size, (0, 0, 0, 0))
    rgba.putalpha(alpha)
    return rgba
