"""Picsart Creative API client.

Currently used for: background removal (removebg). The endpoint is synchronous
and returns a JSON payload with a URL pointing to the resulting cutout PNG.

Docs: https://docs.picsart.io/reference/image-remove-background
"""
from __future__ import annotations

import io

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


PICSART_REMOVEBG_URL = "https://api.picsart.io/tools/1.0/removebg"
_REQUEST_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


class PicsartError(ExternalServiceError):
    code = "picsart_error"


async def remove_background(source: Image.Image) -> Image.Image:
    """Send an image to Picsart removebg and return an RGBA cutout.

    The source is encoded to JPEG (lossless alpha doesn't exist in the input)
    and posted as multipart/form-data with output_type=cutout.
    """
    settings = get_settings()
    if not settings.picsart_api_key:
        raise PicsartError("PICSART_API_KEY is not configured")

    # Encode source to JPEG to keep upload size reasonable.
    buf = io.BytesIO()
    source.convert("RGB").save(buf, format="JPEG", quality=92)
    payload = buf.getvalue()

    headers = {
        "X-Picsart-API-Key": settings.picsart_api_key,
        "accept": "application/json",
    }
    data = {"output_type": "cutout", "format": "PNG"}
    files = {"image": ("source.jpg", payload, "image/jpeg")}

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        ):
            with attempt:
                resp = await client.post(
                    PICSART_REMOVEBG_URL, headers=headers, data=data, files=files
                )

        if resp.status_code >= 400:
            log.warning(
                "picsart.removebg.error",
                status=resp.status_code,
                body=resp.text[:300],
            )
            # 4xx errors aren't retried by tenacity (httpx doesn't raise) so map them here.
            raise PicsartError(f"Picsart removebg returned {resp.status_code}: {resp.text[:200]}")

        body = resp.json()
        if body.get("status") != "success" or "url" not in body.get("data", {}):
            raise PicsartError(f"Unexpected Picsart response: {body}")

        cdn_url = body["data"]["url"]

        # Download the resulting cutout PNG
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        ):
            with attempt:
                img_resp = await client.get(cdn_url)
                img_resp.raise_for_status()

    img = Image.open(io.BytesIO(img_resp.content))
    img.load()
    return img.convert("RGBA")
