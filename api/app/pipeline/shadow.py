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


def add_shadow(
    canvas: Image.Image,
    cutout: Image.Image,
    car_box: tuple[int, int, int, int],
    contact_y_canvas: int,
    light_direction: str = "top",
    intensity: float = 0.55,
    spread: float = 0.45,
) -> Image.Image:
    """Composite a perspective shadow under the car.

    canvas: RGBA scene with car already placed at car_box
    cutout: RGBA car cutout at car_box's size
    car_box: (x, y, w, h) of where the car sits on canvas
    contact_y_canvas: y on canvas where wheels meet the ground
    """
    if cutout.mode != "RGBA":
        cutout = cutout.convert("RGBA")
    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")

    x, y, w, h = car_box
    if cutout.size != (w, h):
        cutout = cutout.resize((w, h), Image.LANCZOS)

    alpha = np.asarray(cutout.split()[-1], dtype=np.uint8)

    # Cap the shadow source to the bottom 50% of the silhouette: the upper body
    # of the car shouldn't contribute as much to the ground shadow because the
    # car is tall but the shadow lies flat.
    H_a, W_a = alpha.shape
    car_height_above_contact = max(contact_y_canvas - y, 1)
    upper_cutoff = max(int(car_height_above_contact * 0.6), 1)
    weighted = alpha.astype(np.float32)
    fade = np.linspace(0.3, 1.0, num=upper_cutoff, dtype=np.float32)
    weighted[:upper_cutoff] *= fade[:, None]

    shadow_alpha = np.clip(weighted * intensity, 0, 255).astype(np.uint8)
    shadow_l = Image.fromarray(shadow_alpha, mode="L")

    # Vertical squash: a car that's H_car tall should cast a shadow ~H_car * spread
    shadow_h = max(int(H_a * spread), 8)
    shadow_w = int(W_a * 1.05)  # slight horizontal extension
    squashed = shadow_l.resize((shadow_w, shadow_h), Image.LANCZOS)

    # Light-direction skew: shear the shadow horizontally
    skew_per_y, x_offset_ratio = _LIGHT_OFFSETS.get(light_direction, _LIGHT_OFFSETS["top"])
    if abs(skew_per_y) > 1e-3:
        # Affine transform: x' = x + skew * y
        skew_total = int(skew_per_y * shadow_h)
        # We need a wider canvas for the skew result
        out_w = shadow_w + abs(skew_total)
        out_l = Image.new("L", (out_w, shadow_h), 0)
        for row in range(shadow_h):
            shift = int(skew_per_y * row)
            base_x = max(skew_total, 0) - shift if skew_per_y > 0 else -min(skew_total, 0) - shift
            out_l.paste(squashed.crop((0, row, shadow_w, row + 1)), (base_x, row))
        squashed = out_l
        shadow_w = out_w

    # Heavy blur for soft shadow edge
    blur_radius = max(min(W_a, H_a) // 25, 8)
    squashed = squashed.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Vertical falloff: shadow gets lighter further from the contact line
    arr = np.asarray(squashed, dtype=np.float32)
    falloff = np.linspace(1.0, 0.25, num=arr.shape[0], dtype=np.float32)[:, None]
    arr = arr * falloff
    squashed = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")

    # Build a black RGBA layer with the shadow as alpha
    shadow_rgba = Image.new("RGBA", squashed.size, (0, 0, 0, 0))
    shadow_rgba.putalpha(squashed)

    # Position: contact line on canvas should align with the top of the shadow.
    # Shadow extends downward from there. Apply small x offset for off-center light.
    shadow_x = x - (shadow_w - w) // 2 + int(w * x_offset_ratio * 0.05)
    shadow_y = contact_y_canvas - shadow_rgba.height // 6

    out = canvas.copy()
    # Paste the shadow underneath the existing car composite. We can't truly
    # "paint behind" so we composite shadow onto a fresh background copy and
    # then re-paste the car on top.
    from .. import backgrounds  # local import to avoid cycles
    bg = backgrounds.get_background("studio-white", canvas.size).convert("RGBA")  # we don't actually use this
    # Rebuild order: take a copy of canvas with the car region cleared back to bg,
    # paste shadow, then paste car on top. But we don't have "canvas without car"
    # easily. Workaround: paste shadow on top of a fresh canvas-sized transparent
    # image, then composite that under the car region of canvas using the inverse
    # of the car's alpha channel as a mask.

    # Simpler and correct approach: paint shadow onto a transparent layer, then
    # composite layer UNDER the car using PIL's Image.alpha_composite ordering.
    # We achieve "under" by:
    #   start with the canvas BEFORE the car was placed (we don't have it)
    # So instead, let's just paste the shadow on top but with the car's alpha
    # subtracted. That keeps the car visible and the shadow visible only where
    # the car isn't.
    out_arr = np.asarray(out).copy()
    shadow_canvas = Image.new("RGBA", out.size, (0, 0, 0, 0))
    shadow_canvas.alpha_composite(shadow_rgba, (shadow_x, shadow_y))
    s_arr = np.array(shadow_canvas, dtype=np.uint8)

    # Subtract car alpha from shadow alpha so the shadow doesn't darken the car
    car_alpha_full = np.zeros(out.size[::-1], dtype=np.uint8)
    car_alpha_local = np.asarray(cutout.split()[-1], dtype=np.uint8)
    cw, ch = w, h
    cx0, cy0 = max(x, 0), max(y, 0)
    cx1, cy1 = min(x + cw, out.size[0]), min(y + ch, out.size[1])
    if cx1 > cx0 and cy1 > cy0:
        car_alpha_full[cy0:cy1, cx0:cx1] = car_alpha_local[cy0 - y:cy1 - y, cx0 - x:cx1 - x]
    s_arr[..., 3] = np.where(car_alpha_full > 16, 0, s_arr[..., 3])
    shadow_canvas = Image.fromarray(s_arr, mode="RGBA")

    # Now composite the shadow OVER the canvas. Because we masked out the car,
    # shadow only appears on the floor / wall regions.
    out.alpha_composite(shadow_canvas)
    return out
