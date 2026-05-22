"""Place a car cutout onto a background canvas.

Smart placement:
    - Detects the wheel contact y of the cutout (where the tyres meet the ground)
    - Reads the floor-line y of the background preset
    - Scales the car to a sensible canvas fraction
    - Anchors so the wheel contact line sits exactly on the floor line

Returns (canvas_rgba, car_box_x_y_w_h, contact_y_in_canvas).
"""
from __future__ import annotations

from PIL import Image

from .. import backgrounds
from .contact import find_wheel_contact_y


def compose(
    cutout: Image.Image,
    background_id: str,
    canvas_size: tuple[int, int] = (1920, 1280),
) -> tuple[Image.Image, tuple[int, int, int, int], int]:
    target_w, target_h = canvas_size
    preset = backgrounds.get_preset(background_id)
    bg = backgrounds.get_background(background_id, (target_w, target_h)).convert("RGBA")

    # Scale the car to fill ~78% canvas width or 70% height, whichever is tighter.
    car = cutout.copy()
    car_max_w = int(target_w * 0.78)
    car_max_h = int(target_h * 0.70)
    car.thumbnail((car_max_w, car_max_h), Image.LANCZOS)

    # Find where the wheels meet the ground in the *scaled* cutout
    contact_y_local = find_wheel_contact_y(car)
    floor_y_canvas = int(target_h * preset.floor_y_ratio)

    # x: center horizontally
    x = (target_w - car.width) // 2
    # y: align contact_y_local of the cutout with floor_y_canvas
    y = floor_y_canvas - contact_y_local

    # Clamp so the car never goes above the top edge or below the canvas
    y = max(min(y, target_h - 1), -car.height // 4)

    canvas = bg.copy()
    canvas.alpha_composite(car, (x, y))
    contact_y_canvas = y + contact_y_local
    return canvas, (x, y, car.width, car.height), contact_y_canvas
