"""Decode and normalize input images.

- Validates magic bytes via python-magic
- Enforces size and pixel-count limits
- Applies EXIF orientation (otherwise sideways phone photos break the pipeline)
- Strips metadata
- Downscales to a working resolution so downstream stages stay fast
"""
from __future__ import annotations

import io

from PIL import Image, ImageOps

from ..errors import ValidationError
from ..settings import get_settings

# Pillow defaults to a low decompression bomb threshold; raise it explicitly so
# we can enforce our own limit instead of getting a Pillow warning.
Image.MAX_IMAGE_PIXELS = 1_000_000_000

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
MAX_INPUT_BYTES = 25 * 1024 * 1024  # 25 MB


def _detect_mime(data: bytes) -> str:
    try:
        import magic  # python-magic

        return magic.from_buffer(data, mime=True)
    except Exception:  # noqa: BLE001
        # Fall back to Pillow inspection if libmagic is missing in the runtime
        try:
            with Image.open(io.BytesIO(data)) as img:
                fmt = (img.format or "").lower()
            return {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(fmt, "")
        except Exception:  # noqa: BLE001
            return ""


def decode(input_bytes: bytes) -> Image.Image:
    """Validate and decode an input image, returning a clean RGB working image."""
    settings = get_settings()

    if len(input_bytes) > MAX_INPUT_BYTES:
        raise ValidationError(f"Input file too large (max {MAX_INPUT_BYTES // (1024 * 1024)} MB)")

    mime = _detect_mime(input_bytes)
    if mime and mime not in ALLOWED_MIME:
        raise ValidationError(f"Unsupported image type: {mime}")

    try:
        img = Image.open(io.BytesIO(input_bytes))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise ValidationError("Could not decode image") from exc

    pixels = img.width * img.height
    if pixels > settings.max_input_pixels:
        raise ValidationError(
            f"Image resolution too large ({pixels:,} pixels, max {settings.max_input_pixels:,})"
        )

    # Apply EXIF orientation, then strip metadata by re-saving through a new image
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    # Working downscale (preserve aspect)
    max_dim = settings.working_max_dim
    if max(img.width, img.height) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    return img
