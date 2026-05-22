"""Build a 'scene-only' image to feed Qwen.

Why: when Qwen sees a car in the input, it sometimes duplicates it or adds a
mirror-image vehicle. Since we re-paste the original car after harmonization,
we don't want Qwen to see (or generate) any car at all — we just want it to
build us the scene.

This stage takes the rough composite + the car cutout alpha, masks the car
region with the average background colour, and lightly feathers the boundary
so Qwen has a soft area to fill in lighting/shadow.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def make_scene_input(
    composite_rgba: Image.Image,
    cutout_alpha: Image.Image,
    car_box: tuple[int, int, int, int],
) -> Image.Image:
    """Return an RGB image with the car region replaced by background paint.

    The replacement colour is the median of the surrounding scene so the patch
    blends naturally and Qwen has a believable starting point.
    """
    if composite_rgba.mode != "RGBA":
        composite_rgba = composite_rgba.convert("RGBA")

    rgb = composite_rgba.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8).copy()
    H, W, _ = arr.shape

    x, y, w, h = car_box

    # Build full-canvas alpha for the car
    car_mask = np.zeros((H, W), dtype=np.uint8)
    a_arr = np.asarray(cutout_alpha.resize((w, h), Image.LANCZOS), dtype=np.uint8)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 > x0 and y1 > y0:
        car_mask[y0:y1, x0:x1] = a_arr[y0 - y : y1 - y, x0 - x : x1 - x]

    # Median scene colour (everywhere the car alpha is below 8/255)
    scene_pixels = arr[car_mask < 8]
    if scene_pixels.size:
        fill = np.median(scene_pixels.reshape(-1, 3), axis=0).astype(np.uint8)
    else:
        fill = np.array([220, 220, 222], dtype=np.uint8)

    # Hard fill where the car was
    paint_mask = car_mask >= 32
    arr[paint_mask] = fill

    # Soft feather the boundary so the model has a graduated region
    feathered = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=8))
    feather_mask = (
        Image.fromarray(car_mask, mode="L")
        .filter(ImageFilter.GaussianBlur(radius=20))
    )
    out = Image.composite(feathered, Image.fromarray(arr), feather_mask)
    return out.convert("RGB")
