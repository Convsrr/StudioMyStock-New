from __future__ import annotations

import io

import pytest
from PIL import Image

from app.errors import ValidationError
from app.pipeline import decode


def test_decode_returns_rgb(car_jpeg_bytes: bytes) -> None:
    img = decode.decode(car_jpeg_bytes)
    assert img.mode == "RGB"
    assert max(img.size) <= 2048


def test_decode_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        decode.decode(b"")


def test_decode_rejects_garbage() -> None:
    with pytest.raises(ValidationError):
        decode.decode(b"not an image, definitely not")


def test_decode_applies_exif_orientation() -> None:
    # Build a portrait image with EXIF orientation 6 (rotated 90 CW)
    img = Image.new("RGB", (200, 400), "red")
    buf = io.BytesIO()
    exif = img.getexif()
    exif[0x0112] = 6  # orientation tag
    img.save(buf, format="JPEG", exif=exif.tobytes())

    decoded = decode.decode(buf.getvalue())
    # After EXIF transpose, dimensions should swap
    assert decoded.size == (400, 200)
