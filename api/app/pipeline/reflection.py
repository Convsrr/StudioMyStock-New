"""Deterministic floor reflection and chassis ambient occlusion under the car.

Two effects, both applied here because they share the wheel-contact line
as their anchor:

1. **Floor reflection** - a faint, downward-fading mirror of the car body
   so the car reads as resting on a polished or semi-matte floor instead
   of floating on it. The reflection is *tinted* toward the local floor
   colour sampled from the canvas, so a white car on a dark floor doesn't
   produce a glowing white smear.

2. **Chassis ambient occlusion** - a soft elliptical darkening directly
   under the silhouette near the contact line. Real cars cast a darker
   "bowl" of shadow under the chassis where the body blocks all ambient
   sky light. This sells the ground attachment more than the contact
   shadow alone, especially on bright studio floors.

Both are deterministic. Qwen never sees them; Qwen's job downstream is
only to lightly polish the lighting.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..logging_setup import get_logger

log = get_logger(__name__)


def add_reflection(
    composite: Image.Image,
    cutout: Image.Image,
    car_box: tuple[int, int, int, int],
    contact_y_canvas: int,
    *,
    opacity: float = 0.18,
    blur_radius: int = 10,
    fade_height_ratio: float = 0.45,
    floor_tint_strength: float = 0.55,
    add_chassis_ao: bool = True,
    ao_strength: float = 0.45,
) -> Image.Image:
    """Composite a faint floor reflection (and optional chassis AO) under the car.

    Parameters
    ----------
    composite:
        RGBA scene with the car already placed at ``car_box``.
    cutout:
        Original RGBA car cutout, sized to ``(w, h)`` of ``car_box`` (or any
        size; it will be resized).
    car_box:
        ``(x, y, w, h)`` placement of ``cutout`` on the canvas.
    contact_y_canvas:
        y-coordinate on the canvas where the wheels meet the floor.
    opacity:
        Maximum opacity of the reflection at the contact line. 0 = invisible,
        1 = fully opaque. Realistic studio values sit in 0.1 - 0.25.
    blur_radius:
        Gaussian blur radius applied to the flipped car so individual details
        smear into a soft floor reflection.
    fade_height_ratio:
        How far below the contact line the reflection extends, as a fraction
        of the car's height. 0.45 means the reflection fades to zero ~45% of
        the car's height below the wheels.
    floor_tint_strength:
        How strongly to bias the reflection toward the local floor colour
        sampled from the canvas. 0 = use the car's own colours, 1 = use the
        floor colour entirely. 0.5..0.7 reads as a real floor reflection.
    add_chassis_ao:
        Whether to add the chassis ambient-occlusion blob.
    ao_strength:
        Maximum darkening from the chassis AO (0..1). 0.45 gives a
        believable shadow without crushing detail.

    Returns
    -------
    A new RGBA image the same size as ``composite``.
    """
    if composite.mode != "RGBA":
        composite = composite.convert("RGBA")
    if cutout.mode != "RGBA":
        cutout = cutout.convert("RGBA")

    canvas_w, canvas_h = composite.size
    x, y, w, h = car_box
    if w <= 0 or h <= 0:
        return composite

    # Sanity-clamp inputs.
    opacity = float(np.clip(opacity, 0.0, 1.0))
    blur_radius = max(int(blur_radius), 0)
    fade_height_ratio = float(np.clip(fade_height_ratio, 0.05, 1.0))
    floor_tint_strength = float(np.clip(floor_tint_strength, 0.0, 1.0))
    ao_strength = float(np.clip(ao_strength, 0.0, 1.0))

    if cutout.size != (w, h):
        cutout = cutout.resize((w, h), Image.LANCZOS)

    out = composite.copy()

    # Chassis AO first so the reflection sits *on top of* the darkened
    # floor. This matches reality: the under-car shadow is in the floor
    # itself, the reflection is the bouncing light from the car body.
    if add_chassis_ao and ao_strength > 0:
        out = _apply_chassis_ao(
            out, cutout, car_box, contact_y_canvas, strength=ao_strength,
        )

    if opacity > 0:
        # Sample floor colour from the canvas just below the contact line,
        # left and right of the car so we don't pick up the car shadow.
        floor_rgb = _sample_floor_colour(out, car_box, contact_y_canvas)

        # 1. Flip vertically so the reflection mirrors the car around the
        #    contact line.
        flipped = cutout.transpose(Image.FLIP_TOP_BOTTOM)

        # 2. Heavy blur so the reflection becomes a soft colour smear, not
        #    a second car silhouette.
        if blur_radius > 0:
            r, g, b, a = flipped.split()
            rgb = Image.merge("RGB", (r, g, b)).filter(
                ImageFilter.GaussianBlur(radius=blur_radius)
            )
            a = a.filter(ImageFilter.GaussianBlur(radius=max(blur_radius // 2, 1)))
            flipped = Image.merge("RGBA", (*rgb.split(), a))

        # 3. Tint the colours toward the floor so the reflection reads as
        #    a floor reflection, not a glowing duplicate car. We blend in
        #    LAB-ish opponent colour space so luminance is mostly
        #    preserved but chroma collapses toward the floor's hue.
        if floor_tint_strength > 0 and floor_rgb is not None:
            flipped = _tint_toward(flipped, floor_rgb, strength=floor_tint_strength)

        # 4. Vertical fade with quadratic falloff.
        fade_h_px = max(int(h * fade_height_ratio), 1)
        if fade_h_px < flipped.height:
            flipped = flipped.crop((0, 0, flipped.width, fade_h_px))

        fh = flipped.height
        fade = np.linspace(1.0, 0.0, num=fh, dtype=np.float32) ** 2
        a_arr = np.asarray(flipped.split()[-1], dtype=np.float32) / 255.0
        a_arr *= fade[:, None]
        a_arr *= opacity
        new_alpha = np.clip(a_arr * 255.0, 0, 255).astype(np.uint8)

        r, g, b, _ = flipped.split()
        flipped = Image.merge(
            "RGBA", (r, g, b, Image.fromarray(new_alpha, mode="L"))
        )

        paste_x = x
        paste_y = contact_y_canvas
        if paste_y < canvas_h:
            out.alpha_composite(flipped, (paste_x, paste_y))

    return out


def _sample_floor_colour(
    canvas: Image.Image,
    car_box: tuple[int, int, int, int],
    contact_y_canvas: int,
) -> tuple[int, int, int] | None:
    """Sample a representative floor colour from the canvas.

    Looks just below the contact line on both sides of the car so we
    avoid sampling the car body or its shadow. Returns ``None`` if there
    is not enough room to sample.
    """
    x, _, w, _ = car_box
    canvas_w, canvas_h = canvas.size
    sample_y0 = contact_y_canvas + 4
    sample_y1 = min(contact_y_canvas + 24, canvas_h)
    if sample_y1 - sample_y0 < 4:
        return None

    rgb = canvas.convert("RGB")

    samples: list[np.ndarray] = []
    # Left of the car
    lx0 = max(0, x - max(w // 4, 16))
    lx1 = max(lx0 + 4, x - 4)
    if lx1 > lx0 and lx1 <= canvas_w:
        samples.append(
            np.asarray(rgb.crop((lx0, sample_y0, lx1, sample_y1)), dtype=np.float32)
            .reshape(-1, 3)
        )
    # Right of the car
    rx0 = min(canvas_w - 4, x + w + 4)
    rx1 = min(canvas_w, rx0 + max(w // 4, 16))
    if rx1 > rx0:
        samples.append(
            np.asarray(rgb.crop((rx0, sample_y0, rx1, sample_y1)), dtype=np.float32)
            .reshape(-1, 3)
        )
    if not samples:
        return None
    arr = np.concatenate(samples, axis=0)
    if arr.size == 0:
        return None
    mean = np.mean(arr, axis=0)
    return (int(mean[0]), int(mean[1]), int(mean[2]))


def _tint_toward(
    image: Image.Image,
    target_rgb: tuple[int, int, int],
    strength: float,
) -> Image.Image:
    """Blend an RGBA image's colour toward ``target_rgb`` while keeping its
    luminance and alpha.

    This is the trick that makes a white-car reflection on a charcoal
    floor look right: hold luminance, collapse chroma toward the floor's
    own colour.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    arr = np.asarray(image, dtype=np.float32)
    rgb = arr[..., :3]
    alpha = arr[..., 3:4]

    luma = (
        0.299 * rgb[..., 0:1]
        + 0.587 * rgb[..., 1:2]
        + 0.114 * rgb[..., 2:3]
    )

    target = np.array(target_rgb, dtype=np.float32)
    target_luma = 0.299 * target[0] + 0.587 * target[1] + 0.114 * target[2]
    if target_luma < 1.0:
        target_luma = 1.0
    # Tint = the target's hue, scaled per-pixel so each pixel keeps its
    # original luma.
    scaled_target = target * (luma / target_luma)
    scaled_target = np.clip(scaled_target, 0, 255)

    out_rgb = rgb * (1.0 - strength) + scaled_target * strength
    out = np.concatenate([out_rgb, alpha], axis=-1)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def _apply_chassis_ao(
    canvas: Image.Image,
    cutout: Image.Image,
    car_box: tuple[int, int, int, int],
    contact_y_canvas: int,
    *,
    strength: float,
) -> Image.Image:
    """Darken a soft ellipse on the floor directly under the car silhouette.

    Approximates the ambient occlusion bowl that real cars cast where the
    body blocks ambient sky light from reaching the floor. Strictly
    below the contact line so it never darkens the car body.
    """
    x, _, w, _ = car_box
    canvas_w, canvas_h = canvas.size

    # Ellipse roughly the silhouette's bottom width, half its height.
    # Anchored *just below* the contact line.
    ao_w = int(w * 1.05)
    ao_h = max(int(w * 0.18), 10)

    ao_x = x + (w - ao_w) // 2
    ao_y = contact_y_canvas + 2  # top edge of ellipse just past contact

    # Build the AO blob in a small dedicated layer for cheap blurring.
    pad = max(ao_h, 16)
    layer_w = max(ao_w + pad * 2, 32)
    layer_h = max(ao_h + pad * 2, 32)
    layer = Image.new("L", (layer_w, layer_h), 0)
    draw = ImageDraw.Draw(layer)
    draw.ellipse((pad, pad, pad + ao_w, pad + ao_h), fill=int(255 * strength))
    blurred = layer.filter(ImageFilter.GaussianBlur(radius=max(ao_h // 3, 6)))

    out = canvas.copy()
    if out.mode != "RGBA":
        out = out.convert("RGBA")

    # Build a canvas-sized darkening alpha.
    full_alpha = Image.new("L", (canvas_w, canvas_h), 0)
    paste_x = ao_x - pad
    paste_y = ao_y - pad
    full_alpha.paste(blurred, (paste_x, paste_y))

    # Clip to strictly below the contact line. Blur has a Gaussian tail
    # that would otherwise darken the car body above the contact line.
    full_alpha_arr = np.asarray(full_alpha, dtype=np.uint8).copy()
    if 0 <= contact_y_canvas < canvas_h:
        full_alpha_arr[: contact_y_canvas + 1, :] = 0
    a = full_alpha_arr.astype(np.float32) / 255.0

    arr = np.asarray(out, dtype=np.float32)
    rgb = arr[..., :3] * (1.0 - a[..., None])
    arr[..., :3] = np.clip(rgb, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")
