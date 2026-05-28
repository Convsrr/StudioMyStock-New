from __future__ import annotations

import numpy as np
from PIL import Image

from app.pipeline import reflection


def _make_canvas_and_cutout() -> tuple[Image.Image, Image.Image, tuple[int, int, int, int], int]:
    """Helper: solid grey canvas (800x600) with a red car-shaped cutout
    placed near the centre. Returns (canvas, cutout, car_box, contact_y).
    """
    canvas = Image.new("RGBA", (800, 600), (220, 220, 220, 255))

    # Car cutout: a 300x180 red rectangle with rounded-ish wheels at the bottom.
    cutout = Image.new("RGBA", (300, 180), (0, 0, 0, 0))
    arr = np.zeros((180, 300, 4), dtype=np.uint8)
    arr[20:170, 10:290] = (200, 30, 30, 255)  # body
    arr[140:175, 30:90] = (20, 20, 20, 255)   # left wheel
    arr[140:175, 210:270] = (20, 20, 20, 255)  # right wheel
    cutout = Image.fromarray(arr, mode="RGBA")

    car_box = (250, 320, 300, 180)
    canvas.alpha_composite(cutout, (car_box[0], car_box[1]))
    contact_y = car_box[1] + car_box[3] - 5  # bottom of wheels
    return canvas, cutout, car_box, contact_y


def test_reflection_does_not_change_image_size() -> None:
    canvas, cutout, car_box, contact_y = _make_canvas_and_cutout()
    out = reflection.add_reflection(canvas, cutout, car_box, contact_y)
    assert out.size == canvas.size


def test_reflection_only_appears_below_contact_line() -> None:
    canvas, cutout, car_box, contact_y = _make_canvas_and_cutout()
    out = reflection.add_reflection(
        canvas, cutout, car_box, contact_y,
        opacity=0.5,  # exaggerate so changes are easy to detect
        blur_radius=8,
    )

    before = np.asarray(canvas, dtype=np.int16)
    after = np.asarray(out, dtype=np.int16)
    diff = np.abs(after - before).sum(axis=2)

    # Above the contact line, the reflection should not have touched anything.
    above = diff[: contact_y - 1, :]
    # Tiny anti-alias smearing at the boundary is OK; require the bulk is
    # untouched.
    assert above.max() <= 5, f"reflection bled above contact line, max diff {above.max()}"

    # Below the contact line, we should see at least *some* change.
    x, _, w, _ = car_box
    below = diff[contact_y + 1 : contact_y + 100, x : x + w]
    assert below.sum() > 0, "expected reflection content below contact line"


def test_reflection_returns_rgba() -> None:
    canvas, cutout, car_box, contact_y = _make_canvas_and_cutout()
    out = reflection.add_reflection(canvas, cutout, car_box, contact_y)
    assert out.mode == "RGBA"


def test_reflection_zero_opacity_is_no_op() -> None:
    canvas, cutout, car_box, contact_y = _make_canvas_and_cutout()
    # Disable chassis AO too so we test the literal "nothing happens"
    # path; chassis AO is opacity-independent and has its own toggle.
    out = reflection.add_reflection(
        canvas, cutout, car_box, contact_y,
        opacity=0.0, add_chassis_ao=False,
    )
    diff = np.abs(
        np.asarray(out, dtype=np.int16) - np.asarray(canvas, dtype=np.int16)
    ).sum()
    assert diff == 0


def test_reflection_chassis_ao_only_below_contact_line() -> None:
    """Chassis AO must darken below the contact line and never above it."""
    canvas, cutout, car_box, contact_y = _make_canvas_and_cutout()
    out = reflection.add_reflection(
        canvas, cutout, car_box, contact_y,
        opacity=0.0,           # disable mirror reflection
        add_chassis_ao=True,
        ao_strength=0.6,
    )
    before = np.asarray(canvas, dtype=np.int16)
    after = np.asarray(out, dtype=np.int16)
    diff = np.abs(after - before).sum(axis=2)

    # Strictly nothing changes above the contact line.
    above = diff[: contact_y + 1, :]
    assert above.max() == 0, f"chassis AO bled above contact line, max diff {above.max()}"

    # Visible darkening below the contact line, in the band right under
    # the car, somewhere in the AO ellipse.
    x, _, w, _ = car_box
    band = diff[contact_y + 2 : contact_y + 30, x : x + w]
    assert band.sum() > 0, "expected chassis AO darkening below contact line"


def test_reflection_does_not_create_full_second_car() -> None:
    """The reflection should be much fainter than the car itself."""
    canvas, cutout, car_box, contact_y = _make_canvas_and_cutout()
    out = reflection.add_reflection(
        canvas, cutout, car_box, contact_y,
        opacity=0.18, blur_radius=10,
    )
    arr = np.asarray(out, dtype=np.int16)

    # Sample the car region (above contact line).
    x, y, w, h = car_box
    car_strip = arr[y + 30 : y + 100, x + 50 : x + w - 50, :3]
    car_redness = (car_strip[..., 0].mean() - car_strip[..., 1].mean())

    # Sample the reflection region (just below contact line) - same x range.
    refl_strip = arr[contact_y + 5 : contact_y + 50, x + 50 : x + w - 50, :3]
    refl_redness = (refl_strip[..., 0].mean() - refl_strip[..., 1].mean())

    # The reflection region must be far less red than the car.
    assert refl_redness < car_redness * 0.6, (
        f"reflection looks too much like the car: car_red={car_redness:.1f}, "
        f"refl_red={refl_redness:.1f}"
    )
