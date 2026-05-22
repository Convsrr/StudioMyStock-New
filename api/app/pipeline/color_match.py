"""Match the car's color cast to the new background.

A Reinhard-style mean/std color transfer in LAB space, applied gently and only
to the masked car region. This catches the "obvious composite" look where the
car's white balance is from a different light than the background.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _to_lab(arr: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to a fast pseudo-LAB. We avoid OpenCV to keep deps slim."""
    # Approximate via simple opponent-color transform. Good enough for color
    # transfer; not colorimetric.
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    L = 0.299 * r + 0.587 * g + 0.114 * b
    A = r - g
    B = 0.5 * (r + g) - b
    return np.stack([L, A, B], axis=-1)


def _from_lab(lab: np.ndarray) -> np.ndarray:
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    g = L - 0.587 / (0.587 + 0.299) * A  # rough inverse, intentionally soft
    r = A + g
    b = (r + g) - 2 * B
    out = np.stack([r, g, b], axis=-1)
    return np.clip(out, 0.0, 1.0)


def color_match(
    composite_rgba: Image.Image,
    car_box: tuple[int, int, int, int],
    cutout_alpha: Image.Image,
    strength: float = 0.35,
) -> Image.Image:
    """Apply gentle color transfer from background statistics onto the masked car."""
    if composite_rgba.mode != "RGBA":
        composite_rgba = composite_rgba.convert("RGBA")

    rgb = np.asarray(composite_rgba.convert("RGB"), dtype=np.float32) / 255.0
    H, W, _ = rgb.shape

    # Build full-canvas alpha for the car
    car_mask = np.zeros((H, W), dtype=np.float32)
    x, y, w, h = car_box
    a = np.asarray(cutout_alpha.resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return composite_rgba
    car_mask[y0:y1, x0:x1] = a[y0 - y : y1 - y, x0 - x : x1 - x]

    bg_mask = 1.0 - car_mask
    if bg_mask.sum() < 1e-3 or car_mask.sum() < 1e-3:
        return composite_rgba

    lab = _to_lab(rgb)

    def _stats(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = mask[..., None]
        total = m.sum() + 1e-6
        mean = (lab * m).sum(axis=(0, 1)) / total
        var = ((lab - mean) ** 2 * m).sum(axis=(0, 1)) / total
        std = np.sqrt(var) + 1e-6
        return mean, std

    bg_mean, bg_std = _stats(bg_mask)
    car_mean, car_std = _stats(car_mask)

    # Move only L (luma) and slightly toward background. Leave chroma mostly alone
    # so the paint color stays recognizable.
    transferred = lab.copy()
    transferred[..., 0] = (transferred[..., 0] - car_mean[0]) * (bg_std[0] / car_std[0]) + bg_mean[0]

    blended_lab = lab * (1 - strength * car_mask[..., None]) + transferred * (strength * car_mask[..., None])
    out_rgb = _from_lab(blended_lab)
    out = (out_rgb * 255.0).astype(np.uint8)

    rgba = np.dstack([out, np.full((H, W), 255, dtype=np.uint8)])
    return Image.fromarray(rgba, mode="RGBA")
