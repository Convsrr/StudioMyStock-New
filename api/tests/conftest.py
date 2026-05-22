"""Test setup. Forces Replicate off so the pipeline runs deterministically."""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture(autouse=True, scope="session")
def _env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("storage")
    os.environ["USE_REPLICATE"] = "false"
    os.environ["REPLICATE_API_TOKEN"] = ""
    os.environ["STORAGE_BACKEND"] = "local"
    os.environ["STORAGE_DIR"] = str(tmp)
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp}/test.db"
    os.environ["API_KEYS"] = ""
    # Reset the cached settings/storage so the env above takes effect
    from app.settings import get_settings
    from app.storage import reset_storage_for_tests

    get_settings.cache_clear()
    reset_storage_for_tests()
    yield


@pytest.fixture
def car_jpeg_bytes() -> bytes:
    """Synthetic 'car' image: a colored rectangle on a textured background."""
    img = Image.new("RGB", (1024, 768), (200, 200, 210))
    # Draw a simple rectangle to stand in for the car body
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([(180, 320), (840, 600)], radius=40, fill=(50, 80, 200))
    draw.ellipse([(260, 540), (380, 660)], fill=(20, 20, 20))
    draw.ellipse([(640, 540), (760, 660)], fill=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()
