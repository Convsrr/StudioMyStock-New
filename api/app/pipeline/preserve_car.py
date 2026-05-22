"""Preserve the original car identity after AI harmonization.

Qwen Image Edit creates a beautifully lit scene but can subtly alter the car
(license plate digits, trim, badges, wheel pattern). For automotive listings
this is unacceptable — dealerships need the exact car back, just in a nicer
setting.

This stage takes the harmonized image (great background + lighting + shadow)
and pastes the original car cutout back on top using the alpha mask from the
segmentation step. The result is:

    - background, ground, ground shadow, environment reflections: from Qwen
    - car body, paint, plate, badges, wheels: pixel-identical to the input

The car receives a small global color/exposure tweak so it doesn't look pasted
onto the new lighting.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from ..logging_setup import get_logger

log = get_logger(__name__)


def _color_match_to_scene(
    car_rgba: Image.Image,
    scene_rgb: Image.Image,
    car_box: tuple[int, int, int, int],
    strength: float = 0.25,
) -> Image.Image:
    """Pull the car's exposure/cast slightly toward the harmonized scene.

    Computes the mean luma of the scene region around where the car will sit
    (excluding the car region itself) and shifts the car luma a fraction of
    the way toward it. Keeps chroma intact so the paint colour doesn't drift.
    """
    if car_rgba.mode != "RGBA":
        car_rgba = car_rgba.convert("RGBA")

    car_arr = np.asarray(car_rgba, dtype=np.float32)
    car_alpha = car_arr[..., 3] / 255.0
    if car_alpha.sum() < 1e-3:
        return car_rgba

    scene_arr = np.asarray(scene_rgb.convert("RGB"), dtype=np.float32)

    # Sample scene luma around the car box (a 30%-wide ring around the box).
    H, W, _ = scene_arr.shape
    x, y, w, h = car_box
    pad_x = int(w * 0.4)
    pad_y = int(h * 0.4)
    rx0, ry0 = max(x - pad_x, 0), max(y - pad_y, 0)
    rx1, ry1 = min(x + w + pad_x, W), min(y + h + pad_y, H)
    region = scene_arr[ry0:ry1, rx0:rx1]
    if region.size == 0:
        return car_rgba

    scene_luma = (0.299 * region[..., 0] + 0.587 * region[..., 1] + 0.114 * region[..., 2]).mean()

    car_rgb = car_arr[..., :3]
    car_luma = (0.299 * car_rgb[..., 0] + 0.587 * car_rgb[..., 1] + 0.114 * car_rgb[..., 2])
    car_mean = (car_luma * car_alpha).sum() / max(car_alpha.sum(), 1e-3)
    if car_mean < 1e-3:
        return car_rgba

    # Multiplicative gain so dark areas stay relatively dark
    gain = 1.0 + strength * (scene_luma / car_mean - 1.0)
    gain = float(np.clip(gain, 0.85, 1.15))  # be conservative; never blow out paint
    adjusted_rgb = np.clip(car_rgb * gain, 0, 255)

    out = np.dstack([adjusted_rgb, car_arr[..., 3]]).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def _feathered_alpha(cutout: Image.Image, feather: int = 1) -> Image.Image:
    """Slightly soften the alpha edge so the re-paste blends with the AI shadow."""
    if cutout.mode != "RGBA":
        cutout = cutout.convert("RGBA")
    if feather <= 0:
        return cutout
    r, g, b, a = cutout.split()
    a = a.filter(ImageFilter.GaussianBlur(radius=feather))
    return Image.merge("RGBA", (r, g, b, a))


def preserve_car(
    harmonized: Image.Image,
    cutout: Image.Image,
    car_box: tuple[int, int, int, int],
    color_match_strength: float = 0.25,
) -> Image.Image:
    """Composite the original car back onto the harmonized scene."""
    if harmonized.mode != "RGB":
        harmonized = harmonized.convert("RGB")

    x, y, w, h = car_box
    if cutout.size != (w, h):
        cutout = cutout.resize((w, h), Image.LANCZOS)

    # Subtle global lighting match before re-paste
    car = _color_match_to_scene(cutout, harmonized, car_box, strength=color_match_strength)
    car = _feathered_alpha(car, feather=1)

    canvas = harmonized.convert("RGBA")
    canvas.alpha_composite(car, (x, y))
    return canvas.convert("RGB")
