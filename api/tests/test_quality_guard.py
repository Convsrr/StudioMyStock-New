from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app.pipeline import quality_guard


def _make_studio_with_car(size: tuple[int, int] = (800, 600)) -> tuple[Image.Image, tuple[int, int, int, int]]:
    img = Image.new("RGB", size, (220, 220, 222))
    draw = ImageDraw.Draw(img)
    car_box = (250, 280, 300, 180)
    x, y, w, h = car_box
    # Car body
    draw.rounded_rectangle((x, y, x + w, y + h), radius=24, fill=(40, 60, 180))
    # Wheels
    draw.ellipse((x + 30, y + h - 60, x + 95, y + h - 5), fill=(15, 15, 15))
    draw.ellipse((x + w - 95, y + h - 60, x + w - 30, y + h - 5), fill=(15, 15, 15))
    # A few details that contribute edges
    draw.line((x + 10, y + 70, x + w - 10, y + 70), fill=(20, 20, 20), width=3)
    return img, car_box


def test_guard_passes_on_subtle_lighting_change() -> None:
    before, car_box = _make_studio_with_car()
    # Simulate a soft AI lighting pass: gentle vignette + slight contrast bump.
    after_arr = np.asarray(before, dtype=np.float32)
    H, W, _ = after_arr.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2, W / 2
    radial = np.sqrt(((xx - cx) / W) ** 2 + ((yy - cy) / H) ** 2)
    light = (1.0 - 0.15 * radial)[..., None]  # gentle vignette ~0.85..1.0
    after_arr = np.clip(after_arr * light, 0, 255).astype(np.uint8)
    after = Image.fromarray(after_arr).filter(ImageFilter.GaussianBlur(radius=0.5))

    flagged = quality_guard.detect_possible_duplicate_vehicle(before, after, car_box)
    assert flagged is False, "subtle lighting change should not be flagged"


def test_guard_catches_synthetic_duplicate_car() -> None:
    before, car_box = _make_studio_with_car()
    after = before.copy()
    draw = ImageDraw.Draw(after)
    # A second "car" drawn well outside the original car_box + halo.
    draw.rounded_rectangle((10, 30, 220, 180), radius=24, fill=(40, 60, 180))
    draw.ellipse((30, 130, 90, 180), fill=(15, 15, 15))
    draw.ellipse((150, 130, 210, 180), fill=(15, 15, 15))
    # Add some structural lines so edge detection catches it.
    draw.line((20, 60, 210, 60), fill=(20, 20, 20), width=3)
    draw.line((20, 100, 210, 100), fill=(20, 20, 20), width=3)

    flagged = quality_guard.detect_possible_duplicate_vehicle(before, after, car_box)
    assert flagged is True, "obvious duplicate vehicle should be flagged"


def test_guard_is_size_robust() -> None:
    before, car_box = _make_studio_with_car()
    # AI returns a slightly differently sized image. Guard must handle it.
    after = before.resize((before.width - 8, before.height - 8))
    flagged = quality_guard.detect_possible_duplicate_vehicle(before, after, car_box)
    # Subtle resize should not trigger.
    assert flagged is False


def test_guard_handles_zero_unsafe_area() -> None:
    """If car_box covers the whole canvas there is no unsafe region."""
    before = Image.new("RGB", (200, 200), (200, 200, 200))
    after = Image.new("RGB", (200, 200), (180, 180, 180))
    flagged = quality_guard.detect_possible_duplicate_vehicle(
        before, after, (0, 0, 200, 200),
    )
    assert flagged is False


def test_guard_ignores_reflection_band_blob() -> None:
    """A wide darkened blob that sits in the floor reflection band
    (immediately below the car) must NOT trip the duplicate guard.

    This is the deterministic reflection territory; both the reflection
    stage and Qwen's polish are allowed to operate there.
    """
    before, car_box = _make_studio_with_car()
    after_arr = np.asarray(before, dtype=np.int16).copy()
    x, y, w, h = car_box

    # Darken a wide horizontal strip directly under the car. Wide and
    # short — exactly the shape a real floor reflection takes.
    band_y0 = y + h + 4
    band_y1 = band_y0 + 30
    band_x0 = x + 4
    band_x1 = x + w - 4
    after_arr[band_y0:band_y1, band_x0:band_x1] -= 35
    after_arr = np.clip(after_arr, 0, 255).astype(np.uint8)
    after = Image.fromarray(after_arr)

    flagged = quality_guard.detect_possible_duplicate_vehicle(
        before, after, car_box,
    )
    assert flagged is False, "reflection-band blob should be exempt"


def test_guard_still_catches_duplicate_above_reflection_band() -> None:
    """A car-shaped dark blob OFF to the side of the reflection band
    must still trip the guard.
    """
    before, car_box = _make_studio_with_car()
    after_arr = np.asarray(before, dtype=np.int16).copy()
    # A dark rectangular blob with car-ish aspect ratio, well off to the
    # side of and above the reflection band.
    after_arr[40:170, 30:230] -= 80
    after_arr = np.clip(after_arr, 0, 255).astype(np.uint8)
    after = Image.fromarray(after_arr)

    flagged = quality_guard.detect_possible_duplicate_vehicle(
        before, after, car_box,
    )
    assert flagged is True, "side-region car-shaped blob should still fire"
