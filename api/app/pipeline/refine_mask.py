"""Mask refinement: feather, despill, edge cleanup.

Background removers leave hard edges and color spill from the original
environment. This stage:
    1. Erodes slightly to pull the edge inside any color spill halo
    2. Feathers the alpha edge so it composites smoothly
    3. Solidifies the interior so partial-alpha noise inside the car body
       (windshield reflections, etc.) doesn't show through.
"""
from __future__ import annotations

from PIL import Image, ImageFilter


def refine(cutout: Image.Image, feather_px: int = 2, erode_px: int = 1) -> Image.Image:
    """Refine an RGBA cutout. Returns a new RGBA image trimmed to its bbox."""
    if cutout.mode != "RGBA":
        cutout = cutout.convert("RGBA")

    r, g, b, a = cutout.split()

    # 1. Solidify the interior: high-alpha regions become fully opaque.
    a = a.point(lambda v: 255 if v > 230 else v)

    # 2. Erode to pull edge inside any color spill from the original background.
    if erode_px > 0:
        a = a.filter(ImageFilter.MinFilter(size=erode_px * 2 + 1))

    # 3. Feather the edge so it composites smoothly.
    if feather_px > 0:
        a = a.filter(ImageFilter.GaussianBlur(radius=feather_px))

    refined = Image.merge("RGBA", (r, g, b, a))
    bbox = refined.getbbox()
    if bbox:
        refined = refined.crop(bbox)
    return refined
