"""Real-world smoke test.

Downloads a stock car photo, runs the full pipeline (with the configured
segmentation provider, e.g. Picsart), and writes original + result to
``storage/smoke/`` so you can inspect them.

Run:
    .venv/bin/python -m scripts.smoke_test_real
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx

# Ensure the package is importable when run directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline import PipelineParams, run_pipeline  # noqa: E402
from app.settings import get_settings  # noqa: E402

# A representative car photo. Picsart's own docs use this image as their public sample.
# Use the thumbnail variant so it stays under our default 24MP input cap.
DEFAULT_IMAGE_URL = "https://cdn140.picsart.com/13902645939997000779.jpg?type=webp&to=min&r=640"


async def main(url: str = DEFAULT_IMAGE_URL) -> None:
    settings = get_settings()
    print(f"segmentation provider: {settings.active_segmentation_provider}")
    print(f"storage dir: {settings.storage_dir}")

    print(f"downloading: {url}")
    async with httpx.AsyncClient(timeout=60, headers={"User-Agent": "StudioMyStock-Smoke/0.1 (test)"}) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    input_bytes = resp.content
    print(f"input size: {len(input_bytes):,} bytes")

    out_dir = settings.storage_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "original.jpg").write_bytes(input_bytes)

    for bg in ("studio-gray", "studio-dark", "sunset-road"):
        print(f"\n--- background: {bg} ---")
        params = PipelineParams(background_id=bg, relight=False)
        result = await run_pipeline(input_bytes, params)
        path = out_dir / f"result_{bg}.jpg"
        path.write_bytes(result.image_bytes)
        print(f"wrote {path}  {result.width}x{result.height}")
        print(f"timings_ms: {result.timings_ms}")

    print(f"\nDone. Inspect {out_dir}/")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_URL))
