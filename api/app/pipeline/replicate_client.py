"""Thin async wrapper around Replicate with retries, timeouts, and download."""
from __future__ import annotations

import asyncio
import io
from typing import Any

import httpx
from PIL import Image
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..errors import ExternalServiceError
from ..logging_setup import get_logger
from ..settings import get_settings

log = get_logger(__name__)


_REPLICATE_TIMEOUT = 300.0


async def run_model(model_ref: str, inputs: dict[str, Any]) -> str | list[str]:
    """Run a Replicate model and return the output URL(s).

    Retries on transient failures. Raises ExternalServiceError on terminal failure.
    """
    settings = get_settings()
    if not settings.replicate_enabled:
        raise ExternalServiceError("Replicate is not configured")

    import replicate

    client = replicate.Client(api_token=settings.replicate_api_token)

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
        reraise=True,
    ):
        with attempt:
            try:
                output = await asyncio.wait_for(
                    asyncio.to_thread(client.run, model_ref, input=inputs),
                    timeout=_REPLICATE_TIMEOUT,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("replicate.run.error", model=model_ref, error=str(exc))
                raise
    return output


async def download_image(url: str) -> Image.Image:
    """Download an image URL into a Pillow image (RGBA for PNG, RGB otherwise)."""
    async with httpx.AsyncClient(timeout=120) as client:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        ):
            with attempt:
                resp = await client.get(url)
                resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content))
    img.load()
    return img
