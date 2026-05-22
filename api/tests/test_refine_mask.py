from __future__ import annotations

from PIL import Image

from app.pipeline import refine_mask


def test_refine_returns_rgba_and_trims() -> None:
    img = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    # Opaque rect in the middle
    for x in range(150, 250):
        for y in range(150, 250):
            img.putpixel((x, y), (200, 50, 50, 255))

    out = refine_mask.refine(img, feather_px=2, erode_px=1)
    assert out.mode == "RGBA"
    # Should be trimmed close to the rect
    assert out.width <= 110
    assert out.height <= 110
