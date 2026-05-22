from __future__ import annotations

from PIL import Image

from app.pipeline import color_match


def test_color_match_returns_rgba_canvas() -> None:
    canvas = Image.new("RGBA", (400, 300), (100, 100, 100, 255))
    cutout_alpha = Image.new("L", (200, 150), 0)
    for x in range(40, 160):
        for y in range(20, 130):
            cutout_alpha.putpixel((x, y), 255)
    out = color_match.color_match(canvas, (100, 75, 200, 150), cutout_alpha, strength=0.4)
    assert out.size == canvas.size
    assert out.mode == "RGBA"
