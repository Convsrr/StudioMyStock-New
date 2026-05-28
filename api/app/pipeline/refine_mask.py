"""Mask refinement: feather, despill, hole-fill, edge cleanup.

Background removers leave hard edges and color spill from the original
environment. They also frequently mark interior regions as semi-transparent
or fully transparent — license plate faces (often the same value as a dark
sky), dark interior windows, hollows under the bumper, and so on. When those
holes survive into the final composite, Qwen sees a partial car and helpfully
"completes" it by drawing a duplicate plate, an extra bumper edge, etc.

This stage:
    1. Erodes slightly to pull the edge inside any color spill halo
    2. Solidifies the interior so partial-alpha noise inside the car body
       (windshield reflections, plate faces, etc.) doesn't show through.
    3. Fills any fully transparent holes that are completely surrounded by
       opaque pixels (these are always segmentation errors, never genuine
       see-through regions).
    4. Feathers the alpha edge so it composites smoothly.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _fill_interior_holes(alpha: Image.Image, opaque_threshold: int = 200) -> Image.Image:
    """Force any transparent region fully enclosed by the silhouette to opaque.

    Strategy: build a binary "is-opaque" view of the alpha. Flood the
    transparent region from *every* border pixel that is still
    transparent at flood time (subsequent passes on the same region are
    cheap because earlier floods mark the pixels with 128, so the loop
    short-circuits). Anything not reached is fully enclosed by the
    silhouette and is forced to 255.

    Critically, we seed from every border pixel, not just one per edge.
    A car cutout that touches the bottom corners of its bbox (long bumper,
    or wheel arches at the very edge of the trim) creates multiple
    disconnected transparent regions on the same border, and missing any
    one of them used to fill the gap between the wheels.
    """
    from PIL import ImageDraw

    arr = np.asarray(alpha, dtype=np.uint8)
    H, W = arr.shape

    # Two-tone working image: 0 = transparent, 255 = opaque. Anything
    # in between is treated as opaque so flood doesn't bleed through soft
    # edges.
    binary = np.where(arr >= opaque_threshold, 255, 0).astype(np.uint8)

    # Image.fromarray on a numpy array marks the result as read-only,
    # which makes ImageDraw.floodfill a silent no-op. Use frombytes for
    # a writable image.
    work = Image.frombytes("L", (W, H), binary.tobytes())

    # Flood from every transparent border pixel. The flood marks visited
    # pixels with 128 so re-checking is O(1) per border pixel.
    def _maybe_flood(x: int, y: int) -> None:
        if work.getpixel((x, y)) == 0:
            ImageDraw.floodfill(work, (x, y), 128)

    for x in range(W):
        _maybe_flood(x, 0)
        _maybe_flood(x, H - 1)
    for y in range(H):
        _maybe_flood(0, y)
        _maybe_flood(W - 1, y)

    flooded = np.asarray(work, dtype=np.uint8)
    # Holes: pixels still at 0 (transparent and not reachable from any border).
    holes = flooded == 0
    if not holes.any():
        return alpha

    out = arr.copy()
    out[holes] = 255
    return Image.fromarray(out, mode="L")


def refine(cutout: Image.Image, feather_px: int = 2, erode_px: int = 1) -> Image.Image:
    """Refine an RGBA cutout. Returns a new RGBA image trimmed to its bbox.

    The returned image carries an extra attribute, ``raw_alpha``: the
    pre-fill, pre-feather alpha mask, also trimmed to the same bbox.
    Downstream stages that need to detect the gap between the wheels
    (compose / contact) read this instead of the post-fill alpha so a
    correctly identified interior fill doesn't fool the contact detector.
    """
    if cutout.mode != "RGBA":
        cutout = cutout.convert("RGBA")

    r, g, b, a = cutout.split()

    # 1. Solidify the interior: high-alpha regions become fully opaque.
    a = a.point(lambda v: 255 if v > 230 else v)

    # 2. Erode to pull edge inside any color spill from the original background.
    if erode_px > 0:
        a = a.filter(ImageFilter.MinFilter(size=erode_px * 2 + 1))

    # Snapshot the alpha *before* the interior-fill pass so contact
    # detection can see the gap between the wheels even when the fill
    # closes it (e.g. a low rider where the gap doesn't reach the bbox
    # bottom).
    raw_alpha = a.copy()

    # 3. Fill any transparent holes that are fully enclosed by opaque pixels.
    a = _fill_interior_holes(a)

    # 4. Feather the edge so it composites smoothly.
    if feather_px > 0:
        a = a.filter(ImageFilter.GaussianBlur(radius=feather_px))

    refined = Image.merge("RGBA", (r, g, b, a))
    bbox = refined.getbbox()
    if bbox:
        refined = refined.crop(bbox)
        raw_alpha = raw_alpha.crop(bbox)

    # Attach the raw alpha so compose / contact can pick it up. PIL Image
    # objects accept arbitrary attributes; we use a clearly-named one to
    # avoid colliding with anything Pillow uses internally.
    refined.raw_alpha = raw_alpha  # type: ignore[attr-defined]
    return refined
