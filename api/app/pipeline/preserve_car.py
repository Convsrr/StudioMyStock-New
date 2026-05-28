"""Preserve dealer-accurate car identity after the AI harmonize pass.

The compositor places the original car cutout on the studio background and
the AI pass polishes the lighting. This stage guarantees the car the
dealer uploaded is the car the dealer gets back.

We composite the original cutout pixels back over the harmonized output
using the segmentation alpha. A small amount of the harmonized lighting is
allowed to bleed through (governed by ``car_identity_strength``) so the
car body picks up some of the studio reflections; the default of 0.92
keeps 92% of the original car pixels and 8% of Qwen's lighting touch -
enough for paint to look studio-lit, not so much that plates or badges
drift.

Adaptive identity (chrome / glass / dark paint):
    Real automotive photography lives or dies on chrome and glass
    reflections. A flat 0.92 identity blend across the whole car keeps
    every pixel dealer-accurate but also keeps the chrome looking flat
    and the windows looking opaque. We classify each pixel inside the
    cutout into one of three bands and apply different identity weights:

        - badge / plate / paint (default): full identity_strength
        - bright specular (chrome, headlights, polished trim):
          identity scaled down toward 0.78 so a small amount of Qwen's
          studio reflections show through
        - glass (low-saturation mid-luma, filtered to where windows
          actually live in the silhouette): identity scaled down toward
          0.75 so reflections in the glass come through

    The floors are deliberately on the conservative side. We learned
    the hard way that 0.55 / 0.65 let too much of Qwen through on
    silver / metallic cars, where huge body regions classify as
    "bright specular". When in doubt, keep the dealer's car.

    Plates and badges are protected because they sit at well-known
    luminance / saturation levels (dark digits on a bright field, or
    textured chrome script) which both fall outside the "bright
    specular" classifier band when the *plate region as a whole* is
    considered. To be safe we also clamp identity to a higher floor
    inside the bottom-rear band of the cutout where rear plates live.

Edges are softly feathered (1-2 px) to avoid a hard cutout halo without
producing a glow.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from ..logging_setup import get_logger

log = get_logger(__name__)


def _feathered_alpha(cutout: Image.Image, feather: int = 1) -> Image.Image:
    """Slightly soften the alpha edge so the re-paste blends with the AI shadow."""
    if cutout.mode != "RGBA":
        cutout = cutout.convert("RGBA")
    if feather <= 0:
        return cutout
    r, g, b, a = cutout.split()
    a = a.filter(ImageFilter.GaussianBlur(radius=feather))
    return Image.merge("RGBA", (r, g, b, a))


def _adaptive_identity_map(
    cutout_rgb: np.ndarray,
    cutout_alpha: np.ndarray,
    base_identity: float,
    *,
    chrome_floor: float = 0.78,
    glass_floor: float = 0.75,
) -> np.ndarray:
    """Return a per-pixel identity-strength map shaped like ``cutout_alpha``.

    Inputs are float arrays in 0..1 range for RGB and alpha. The returned
    array is also float in 0..1 and is meant to multiply with ``alpha``
    when composing.
    """
    H, W = cutout_alpha.shape
    out = np.full((H, W), base_identity, dtype=np.float32)

    # Convert RGB to HSV-ish quickly: max - min for saturation, max for value.
    cmax = cutout_rgb.max(axis=-1)
    cmin = cutout_rgb.min(axis=-1)
    value = cmax  # 0..1
    chroma = cmax - cmin  # 0..1
    sat = np.where(value > 1e-3, chroma / np.maximum(value, 1e-3), 0.0)

    inside = cutout_alpha > 0.5

    # Per-row y position (0..1) broadcast across the width so we can use
    # full (H, W) boolean masks below without shape mismatches.
    y_norm = np.broadcast_to(
        np.arange(H, dtype=np.float32)[:, None] / max(H - 1, 1),
        (H, W),
    )

    # Bright specular: very high value, low chroma. Chrome / headlights /
    # polished trim. We only relax identity here.
    bright_specular = inside & (value > 0.85) & (sat < 0.12)
    if bright_specular.any():
        out[bright_specular] = chrome_floor

    # Glass: mid-to-low value, low saturation, located in the upper half
    # of the silhouette (windows aren't on the floor). The "upper half"
    # gate excludes the rear plate area and the front bumper, both of
    # which we want to keep at full identity.
    upper_silhouette = y_norm < 0.55
    glass = (
        inside
        & upper_silhouette
        & (value > 0.18)
        & (value < 0.75)
        & (sat < 0.18)
    )
    if glass.any():
        glass_only = glass & ~bright_specular
        out[glass_only] = glass_floor

    # Plate / badge protection: bottom 18% of the cutout (where plates
    # and badge clusters live on most cars) is held at base_identity.
    bottom_band = y_norm > 0.82
    out[bottom_band] = np.maximum(out[bottom_band], base_identity)

    # Smooth the identity map a touch so we don't see a sharp border
    # between bands inside the car.
    smooth = Image.fromarray((out * 255.0).astype(np.uint8), mode="L").filter(
        ImageFilter.GaussianBlur(radius=2.0)
    )
    return np.asarray(smooth, dtype=np.float32) / 255.0


def preserve_car(
    harmonized: Image.Image,
    cutout: Image.Image,
    car_box: tuple[int, int, int, int],
    *,
    car_identity_strength: float = 0.92,
    feather_px: int = 1,
    adaptive: bool = True,
) -> Image.Image:
    """Composite the original car back onto the harmonized scene.

    Parameters
    ----------
    harmonized:
        The scene returned by the AI harmonize pass. Used as the canvas.
    cutout:
        Original car RGBA cutout from segmentation, sized to ``(w, h)``
        of ``car_box`` (will be resized if needed).
    car_box:
        ``(x, y, w, h)`` placement of ``cutout`` on the canvas.
    car_identity_strength:
        Base blend weight for the original car pixels vs Qwen's harmonized
        version of the car region. ``1.0`` keeps the cutout pixel-perfect;
        ``0.0`` discards them entirely. Default ``0.92`` keeps dealer
        identity strong while letting a faint amount of Qwen's lighting
        play across the car body. Clamped to ``[0.5, 1.0]``.
    feather_px:
        Alpha edge softening in pixels.
    adaptive:
        If True, classify chrome / glass / paint inside the cutout and
        let more of Qwen's harmonized lighting through on chrome and
        glass while keeping plates and badges at full identity. If
        False, use a flat identity blend across the whole cutout.

    Returns
    -------
    A new RGB image the same size as ``harmonized``.
    """
    harmonized_rgb = (
        harmonized.convert("RGB") if harmonized.mode != "RGB" else harmonized
    )
    canvas_size = harmonized_rgb.size
    canvas_w, canvas_h = canvas_size

    x, y, w, h = car_box
    if w <= 0 or h <= 0:
        return harmonized_rgb

    if cutout.size != (w, h):
        cutout = cutout.resize((w, h), Image.LANCZOS)

    cutout = _feathered_alpha(cutout, feather=feather_px)

    identity = float(np.clip(car_identity_strength, 0.5, 1.0))

    if identity >= 0.999 and not adaptive:
        # Fast path: pure paste.
        canvas = harmonized_rgb.convert("RGBA")
        canvas.alpha_composite(cutout, (x, y))
        return canvas.convert("RGB")

    # Blended path: linearly mix the cutout's RGB with whatever Qwen has
    # under it, weighted by identity. The cutout's alpha still determines
    # *where* the blend applies (no leaking onto the floor / walls).
    canvas_arr = np.asarray(harmonized_rgb, dtype=np.float32)
    cut_arr = np.asarray(cutout, dtype=np.float32)
    cut_rgb = cut_arr[..., :3]
    cut_alpha = cut_arr[..., 3] / 255.0  # 0..1

    # Per-pixel identity (adaptive) or flat (legacy).
    if adaptive:
        identity_map_local = _adaptive_identity_map(
            cut_rgb / 255.0, cut_alpha, identity,
        )
    else:
        identity_map_local = np.full_like(cut_alpha, identity)

    # Place the cutout into a canvas-sized buffer for vectorised math.
    full_rgb = np.zeros_like(canvas_arr)
    full_alpha = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    full_identity = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    cx0, cy0 = max(x, 0), max(y, 0)
    cx1, cy1 = min(x + w, canvas_w), min(y + h, canvas_h)
    if cx1 <= cx0 or cy1 <= cy0:
        return harmonized_rgb

    src_x0, src_y0 = cx0 - x, cy0 - y
    src_x1, src_y1 = src_x0 + (cx1 - cx0), src_y0 + (cy1 - cy0)

    full_rgb[cy0:cy1, cx0:cx1] = cut_rgb[src_y0:src_y1, src_x0:src_x1]
    full_alpha[cy0:cy1, cx0:cx1] = cut_alpha[src_y0:src_y1, src_x0:src_x1]
    full_identity[cy0:cy1, cx0:cx1] = identity_map_local[src_y0:src_y1, src_x0:src_x1]

    # Effective per-pixel weight: alpha * identity. Outside the cutout
    # alpha is 0, so canvas_arr is preserved untouched. Inside the cutout
    # we blend identity*cutout + (1-identity)*canvas, all under the alpha
    # mask.
    weight = (full_alpha * full_identity)[..., None]
    out = full_rgb * weight + canvas_arr * (1.0 - weight)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")
