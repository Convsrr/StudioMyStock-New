from __future__ import annotations

from PIL import Image

from app.pipeline import compose


def test_compose_places_on_canvas() -> None:
    cutout = Image.new("RGBA", (300, 200), (200, 50, 50, 255))
    canvas, box, contact_y = compose.compose(cutout, "studio-white", canvas_size=(800, 600))
    assert canvas.size == (800, 600)
    x, y, w, h = box
    assert 0 <= x < 800
    assert 0 <= y + h <= 600 + 50  # allow slight overflow we clamp to
    assert 0 < w <= 800
    assert 0 < h <= 600
    assert 0 < contact_y <= 600
