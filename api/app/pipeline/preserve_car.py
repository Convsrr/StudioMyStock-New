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

Safe mode (``safe_base`` parameter):
    When ``safe_base`` is provided we use the deterministic pre-harmonize
    composite as the background base and pull only a global tone/lighting
    reference from Qwen's output. This guarantees that any "ghost car",
    duplicate silhouette, or hallucinated extra vehicle Qwen may have drawn
    outside the original car mask can never survive into the final image
    because the floor and background pixels come from our own composite.
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


def _global_tone_shift(
    base_rgb: Image.Image,
    reference_rgb: Image.Image,
    car_box: tuple[int, int, int, int],
    strength: float = 0.5,
) -> Image.Image:
    """Apply a subtle global tone/lighting shift derived from a reference image.

    Compares the average colour outside the car bounding box between the
    ``base_rgb`` (deterministic composite) and the ``reference_rgb`` (Qwen
    output) and nudges the base a fraction of the way toward the reference.
    Hard-clamped per-channel gain so we never produce strong artefacts even
    if Qwen returned something wildly different.
    """
    base = np.asarray(base_rgb.convert("RGB"), dtype=np.float32)
    ref = np.asarray(reference_rgb.convert("RGB"), dtype=np.float32)
    if base.shape != ref.shape:
        return base_rgb

    H, W, _ = base.shape
    x, y, w, h = car_box

    # Build a mask of "scene only" pixels (outside the car box).
    mask = np.ones((H, W), dtype=bool)
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + w, W)
    y1 = min(y + h, H)
    mask[y0:y1, x0:x1] = False
    if mask.sum() < 16:
        return base_rgb

    base_mean = base[mask].reshape(-1, 3).mean(axis=0)
    ref_mean = ref[mask].reshape(-1, 3).mean(axis=0)
    base_mean = np.clip(base_mean, 1.0, None)

    gain = 1.0 + strength * (ref_mean / base_mean - 1.0)
    gain = np.clip(gain, 0.92, 1.08)  # very conservative
    adjusted = np.clip(base * gain, 0, 255).astype(np.uint8)
    return Image.fromarray(adjusted, mode="RGB")


def preserve_car(
    harmonized: Image.Image,
    cutout: Image.Image,
    car_box: tuple[int, int, int, int],
    color_match_strength: float = 0.25,
    safe_base: Image.Image | None = None,
) -> Image.Image:
    """Composite the original car back onto the (harmonized) scene.

    Parameters
    ----------
    harmonized:
        Qwen's harmonized output. Used for tone reference and (in non-safe
        mode) as the actual background base.
    cutout:
        Original car RGBA cutout from segmentation.
    car_box:
        ``(x, y, w, h)`` placement of ``cutout`` on the canvas.
    color_match_strength:
        How aggressively to nudge the car's exposure toward the scene.
    safe_base:
        Optional. The deterministic pre-harmonize composite (RGB). When
        provided, this image is used as the background base instead of
        ``harmonized``. A subtle global tone shift derived from ``harmonized``
        is applied so we still benefit from Qwen's lighting cues, but any
        duplicate car / ghost silhouette Qwen may have drawn outside the
        original car region cannot leak into the final result.
    """
    harmonized_rgb = harmonized.convert("RGB") if harmonized.mode != "RGB" else harmonized

    x, y, w, h = car_box
    if cutout.size != (w, h):
        cutout = cutout.resize((w, h), Image.LANCZOS)

    if safe_base is not None:
        base = safe_base.convert("RGB") if safe_base.mode != "RGB" else safe_base
        if base.size != harmonized_rgb.size:
            # Resize the deterministic composite to match the harmonized canvas
            # rather than the other way round; the cutout/car_box are sized
            # against the deterministic canvas.
            base = base.resize(harmonized_rgb.size, Image.LANCZOS)
        # Subtle global tone shift from Qwen's output so the deterministic
        # background still picks up lighting cues without copying any pixels.
        base = _global_tone_shift(base, harmonized_rgb, car_box, strength=0.5)
        # Match the car against the deterministic-but-toned scene.
        car = _color_match_to_scene(cutout, base, car_box, strength=color_match_strength)
        car = _feathered_alpha(car, feather=1)
        canvas = base.convert("RGBA")
        canvas.alpha_composite(car, (x, y))
        return canvas.convert("RGB")

    # Default behaviour: use Qwen's output as the background base.
    car = _color_match_to_scene(cutout, harmonized_rgb, car_box, strength=color_match_strength)
    car = _feathered_alpha(car, feather=1)

    canvas = harmonized_rgb.convert("RGBA")
    canvas.alpha_composite(car, (x, y))
    return canvas.convert("RGB")
