from __future__ import annotations

import numpy as np
from PIL import Image

from app.pipeline import preserve_car


def _make_scene_and_cutout(canvas_size=(800, 600), car_size=(300, 180)):
    """A grey 'harmonized' scene + a solid red car cutout."""
    scene = Image.new("RGB", canvas_size, (180, 180, 180))
    car = np.zeros((car_size[1], car_size[0], 4), dtype=np.uint8)
    car[10:-10, 10:-10] = (220, 30, 30, 255)
    cutout = Image.fromarray(car, mode="RGBA")
    car_box = (
        (canvas_size[0] - car_size[0]) // 2,
        (canvas_size[1] - car_size[1]) // 2,
        car_size[0],
        car_size[1],
    )
    return scene, cutout, car_box


def test_preserve_car_returns_same_size() -> None:
    scene, cutout, car_box = _make_scene_and_cutout()
    out = preserve_car.preserve_car(scene, cutout, car_box)
    assert out.size == scene.size
    assert out.mode == "RGB"


def test_preserve_car_keeps_car_region_dominant_at_default() -> None:
    """Default identity_strength=0.92 should keep the cutout dominant."""
    scene, cutout, car_box = _make_scene_and_cutout()
    out = preserve_car.preserve_car(scene, cutout, car_box)
    arr = np.asarray(out, dtype=np.int16)

    x, y, w, h = car_box
    inside = arr[y + 30 : y + h - 30, x + 30 : x + w - 30, :]
    # The cutout body is red. After re-paste with identity 0.92, the
    # interior must be much closer to red than to the grey scene.
    red_dominance = inside[..., 0].mean() - inside[..., 1].mean()
    assert red_dominance > 80, (
        f"car region not dominant; red_dominance={red_dominance:.1f}"
    )


def test_preserve_car_does_not_touch_outside_region() -> None:
    scene, cutout, car_box = _make_scene_and_cutout()
    out = preserve_car.preserve_car(scene, cutout, car_box)
    before = np.asarray(scene, dtype=np.int16)
    after = np.asarray(out, dtype=np.int16)

    x, y, w, h = car_box
    # Pixel rows entirely above the car should be untouched.
    assert np.array_equal(before[: y - 10, :, :], after[: y - 10, :, :])
    # Same for rows entirely below the car.
    assert np.array_equal(before[y + h + 10 :, :, :], after[y + h + 10 :, :, :])


def test_preserve_car_identity_full_strength_is_pure_paste() -> None:
    scene, cutout, car_box = _make_scene_and_cutout()
    out = preserve_car.preserve_car(
        scene, cutout, car_box, car_identity_strength=1.0, feather_px=0,
    )
    arr = np.asarray(out, dtype=np.int16)
    x, y, w, h = car_box
    # Centre of the cutout is exactly red (220,30,30) when feather=0.
    cx, cy = x + w // 2, y + h // 2
    assert tuple(arr[cy, cx, :3]) == (220, 30, 30)


def test_preserve_car_lower_identity_blends_more_scene() -> None:
    """At identity_strength=0.5 (the floor), some scene should be visible."""
    scene, cutout, car_box = _make_scene_and_cutout()
    high = preserve_car.preserve_car(
        scene, cutout, car_box, car_identity_strength=1.0, feather_px=0,
    )
    low = preserve_car.preserve_car(
        scene, cutout, car_box, car_identity_strength=0.5, feather_px=0,
    )
    high_arr = np.asarray(high, dtype=np.int16)
    low_arr = np.asarray(low, dtype=np.int16)

    x, y, w, h = car_box
    cx, cy = x + w // 2, y + h // 2
    # Lower identity should have more grey bleeding into the red.
    assert low_arr[cy, cx, 0] < high_arr[cy, cx, 0]  # less red
    assert low_arr[cy, cx, 1] > high_arr[cy, cx, 1]  # more green (toward grey)


def test_preserve_car_clamps_identity_lower_bound() -> None:
    """Even with identity=0.0 we never drop below the safety floor."""
    scene, cutout, car_box = _make_scene_and_cutout()
    out = preserve_car.preserve_car(
        scene, cutout, car_box, car_identity_strength=0.0, feather_px=0,
    )
    arr = np.asarray(out, dtype=np.int16)
    x, y, w, h = car_box
    cx, cy = x + w // 2, y + h // 2
    # At the clamped 0.5 floor, the centre should still be more red
    # than the surrounding grey (180,180,180).
    assert arr[cy, cx, 0] > arr[cy, cx, 1]
