"""Synthesize a perspective contact shadow under the car.

We have:
    - The car's alpha silhouette
    - The exact wheel contact line (in canvas coordinates)
    - The light direction from the background preset

Approach:
    1. Take the car alpha (an RGBA cutout placed at car_box on the canvas)
    2. Build a "ground projection" of the silhouette: vertically squash so the
       shadow lies on the floor plane instead of standing up like the car
    3. Skew horizontally based on light direction
    4. Heavy gaussian blur with a falloff toward the edges
    5. Composite UNDER the car at moderate opacity

The result is a real-shaped shadow that matches the car's actual outline and
the chosen light. No more sedan-icon silhouette.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


_LIGHT_OFFSETS = {
    # (horizontal skew in pixels per unit height, sign of x offset)
    "top":         (0.00,  0.0),
    "top-left":    (0.18,  0.6),   # shadow falls to the right
    "top-right":   (-0.18, -0.6),  # shadow falls to the left
}


def build_shadow_alpha(
    canvas_size: tuple[int, int],
    cutout: Image.Image,
    car_box: tuple[int, int, int, int],
    contact_y_canvas: int,
    light_direction: str = "top",
    intensity: float = 0.55,
    spread: float = 0.45,
) -> Image.Image:
    """Return a canvas-sized L mask describing where the contact shadow should
    darken the floor (255 = full strength, 0 = no shadow). Does not composite
    onto a canvas.

    The returned mask has the car silhouette zeroed so it can be applied as a
    pure floor darkening without touching the car body.
    """
    if cutout.mode != "RGBA":
        cutout = cutout.convert("RGBA")
    x, y, w, h = car_box
    if cutout.size != (w, h):
        cutout = cutout.resize((w, h), Image.LANCZOS)

    alpha = np.asarray(cutout.split()[-1], dtype=np.uint8)
    H_a, W_a = alpha.shape

    # Cap the shadow source to the bottom of the silhouette
    car_height_above_contact = max(contact_y_canvas - y, 1)
    upper_cutoff = max(int(car_height_above_contact * 0.6), 1)
    weighted = alpha.astype(np.float32)
    fade = np.linspace(0.3, 1.0, num=upper_cutoff, dtype=np.float32)
    weighted[:upper_cutoff] *= fade[:, None]

    shadow_alpha = np.clip(weighted * intensity, 0, 255).astype(np.uint8)
    shadow_l = Image.fromarray(shadow_alpha, mode="L")

    # Vertical squash + horizontal extension
    shadow_h = max(int(H_a * spread), 8)
    shadow_w = int(W_a * 1.05)
    squashed = shadow_l.resize((shadow_w, shadow_h), Image.LANCZOS)

    # Light-direction skew
    skew_per_y, x_offset_ratio = _LIGHT_OFFSETS.get(light_direction, _LIGHT_OFFSETS["top"])
    if abs(skew_per_y) > 1e-3:
        skew_total = int(skew_per_y * shadow_h)
        out_w = shadow_w + abs(skew_total)
        out_l = Image.new("L", (out_w, shadow_h), 0)
        for row in range(shadow_h):
            shift = int(skew_per_y * row)
            base_x = max(skew_total, 0) - shift if skew_per_y > 0 else -min(skew_total, 0) - shift
            out_l.paste(squashed.crop((0, row, shadow_w, row + 1)), (base_x, row))
        squashed = out_l
        shadow_w = out_w

    blur_radius = max(min(W_a, H_a) // 25, 8)
    squashed = squashed.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    arr = np.asarray(squashed, dtype=np.float32)
    falloff = np.linspace(1.0, 0.25, num=arr.shape[0], dtype=np.float32)[:, None]
    arr = arr * falloff
    squashed = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")

    shadow_x = x - (shadow_w - w) // 2 + int(w * x_offset_ratio * 0.05)
    shadow_y = contact_y_canvas - squashed.height // 6

    canvas_w, canvas_h = canvas_size
    full = Image.new("L", (canvas_w, canvas_h), 0)
    full.paste(squashed, (shadow_x, shadow_y))

    # Zero out anywhere the car covers so the shadow doesn't darken the car.
    car_alpha_full = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    car_alpha_local = np.asarray(cutout.split()[-1], dtype=np.uint8)
    cx0, cy0 = max(x, 0), max(y, 0)
    cx1, cy1 = min(x + w, canvas_w), min(y + h, canvas_h)
    if cx1 > cx0 and cy1 > cy0:
        car_alpha_full[cy0:cy1, cx0:cx1] = car_alpha_local[cy0 - y:cy1 - y, cx0 - x:cx1 - x]

    full_arr = np.asarray(full, dtype=np.uint8).copy()
    full_arr = np.where(car_alpha_full > 16, 0, full_arr)
    return Image.fromarray(full_arr, mode="L")


def add_shadow(
    canvas: Image.Image,
    cutout: Image.Image,
    car_box: tuple[int, int, int, int],
    contact_y_canvas: int,
    light_direction: str = "top",
    intensity: float = 0.55,
    spread: float = 0.45,
) -> Image.Image:
    """Composite a perspective shadow under the car onto an existing canvas.

    Used in the classical (non-Qwen) path. For the Qwen path we use
    ``build_shadow_alpha`` and apply the shadow multiplicatively in
    preserve_car so it lands on the harmonized floor.
    """
    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")

    shadow_l = build_shadow_alpha(
        canvas.size,
        cutout,
        car_box,
        contact_y_canvas,
        light_direction=light_direction,
        intensity=intensity,
        spread=spread,
    )
    shadow_rgba = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_rgba.putalpha(shadow_l)
    out = canvas.copy()
    out.alpha_composite(shadow_rgba)
    return out
