"""Detect the wheel contact line of a car cutout.

The cutout's bbox bottom is rarely the actual ground-touch point — it's the
lowest pixel of the lower bumper, exhaust tip, or alpha noise. The wheels
contact the ground at a specific y where the silhouette starts widening
again as you scan upward.

Heuristic: scan upward from the bottom of the cutout. Track the alpha row
width at each y. The contact line is the highest y where the row width is
within ~95% of the maximum width. That y is the wheel touchdown line for
the car's actual rubber.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def find_wheel_contact_y(cutout: Image.Image, alpha_threshold: int = 32) -> int:
    """Return the y (in cutout coordinates) where the wheels touch the ground.

    The cutout is assumed to be trimmed to its bbox so y=0 is the top of the
    car and y=H-1 is the lowest pixel of any part of the car.
    """
    if cutout.mode != "RGBA":
        cutout = cutout.convert("RGBA")
    alpha = np.asarray(cutout.split()[-1], dtype=np.uint8)
    H, W = alpha.shape

    # Width of opaque pixels at each row
    row_widths = (alpha >= alpha_threshold).sum(axis=1)
    if not row_widths.any():
        return H - 1

    max_width = int(row_widths.max())
    if max_width == 0:
        return H - 1

    # Scan from the bottom upward. The contact y is the lowest row where
    # width is within 95% of max width — that's where the tyre meets the
    # ground. Below that point the silhouette tapers to the bumper / exhaust.
    threshold = int(max_width * 0.95)
    bottom_third_start = int(H * 0.55)

    contact_y = H - 1
    for y in range(H - 1, bottom_third_start - 1, -1):
        if row_widths[y] >= threshold:
            contact_y = y
            break

    return contact_y
