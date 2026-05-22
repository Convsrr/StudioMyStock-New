"""Single-background smoke test. Cheaper way to verify Qwen integration.

Run:
    .venv/bin/python -m scripts.smoke_test_one [background_id]
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

DEFAULT_IMAGE_URL = "https://cdn140.picsart.com/13902645939997000779.jpg?type=webp&to=min&r=640"


async def main(bg: str = "studio-white") -> None:
    settings = get_settings()
    print(f"replicate enabled: {settings.replicate_enabled}")

    async with httpx.AsyncClient(timeout=60, headers={"User-Agent": "studio-smoke/0.1"}) as c:
        resp = await c.get(DEFAULT_IMAGE_URL)
        resp.raise_for_status()
    body = resp.content

    out_dir = settings.storage_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- {bg} (harmonize=True) ---")
    params = PipelineParams(background_id=bg, harmonize=True)
    result = await run_pipeline(body, params)
    path = out_dir / f"harmonized_{bg}.jpg"
    path.write_bytes(result.image_bytes)
    print(f"wrote {path}  {result.width}x{result.height}")
    print(f"timings: {result.timings_ms}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "studio-white"))
