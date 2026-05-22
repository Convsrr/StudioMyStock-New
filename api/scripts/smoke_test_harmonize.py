"""End-to-end smoke test of the full pipeline including Qwen Image Edit harmonize.

Downloads a real car photo, runs it through the pipeline with harmonize=True
across all 5 studio backgrounds, and writes the results to ``storage/smoke/``.

Run:
    .venv/bin/python -m scripts.smoke_test_harmonize
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline import PipelineParams, run_pipeline  # noqa: E402
from app.settings import get_settings  # noqa: E402

# Picsart's own demo car photo, kept small so the input fits under our pixel cap.
DEFAULT_IMAGE_URL = "https://cdn140.picsart.com/13902645939997000779.jpg?type=webp&to=min&r=640"


async def main(url: str = DEFAULT_IMAGE_URL) -> None:
    settings = get_settings()
    print(f"replicate enabled: {settings.replicate_enabled}")
    print(f"segmentation provider: {settings.active_segmentation_provider}")

    async with httpx.AsyncClient(
        timeout=60, headers={"User-Agent": "StudioMyStock-Smoke/0.1"}
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    input_bytes = resp.content
    print(f"downloaded {len(input_bytes):,} bytes")

    out_dir = settings.storage_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "original.jpg").write_bytes(input_bytes)

    presets = ["studio-white", "studio-grey", "studio-charcoal", "studio-warm", "studio-blueprint"]
    for bg in presets:
        print(f"\n--- {bg} (harmonize=True) ---")
        params = PipelineParams(background_id=bg, harmonize=True)
        result = await run_pipeline(input_bytes, params)
        path = out_dir / f"harmonized_{bg}.jpg"
        path.write_bytes(result.image_bytes)
        print(f"wrote {path}  {result.width}x{result.height}")
        print(f"timings: {result.timings_ms}")

    print(f"\nDone. Inspect {out_dir}/")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_URL))
