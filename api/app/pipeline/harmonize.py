"""Harmonization stage powered by Qwen Image Edit 2511 on Replicate.

This is the stage that turns a "pasted" composite into a believable studio
photograph. We give the model:

  1. The rough composite (car cutout placed on the chosen studio cyc).
  2. A scene-specific prompt describing the target lighting and surfaces.
  3. Strict instructions to preserve the car identity (paint, badges, plate, wheels).

We send the input as a data URL to avoid uploading to a public URL first.
The model returns one or more URLs to the edited image.

Docs: https://replicate.com/qwen/qwen-image-edit-2511
"""
from __future__ import annotations

import asyncio
import base64
import io
from typing import Optional

import httpx
from PIL import Image
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .. import backgrounds
from ..errors import ExternalServiceError
from ..logging_setup import get_logger
from ..settings import get_settings

log = get_logger(__name__)


_MODEL = "qwen/qwen-image-edit-2511"
_REPLICATE_TIMEOUT = httpx.Timeout(180.0, connect=10.0)


_PRESERVE_INSTRUCTIONS = (
    "There is exactly one car in this image. There must be exactly one car in the output, "
    "in the same position and orientation as the input. Do not duplicate the car, do not add "
    "extra cars, do not extend the car body, do not generate any additional vehicles or "
    "reflections of the car. "
    "Focus your edits on: the studio walls, floor surface, ambient lighting on the car body, "
    "and the soft contact shadow directly underneath the car's wheels. "
    "Do not modify the car body shape, paint colour, badges, license plate digits, wheels, or trim."
)


def _aspect_ratio_for(size: tuple[int, int]) -> str:
    """Return Qwen's supported aspect_ratio. We use match_input_image so the
    output matches whatever canvas we sent (e.g. 3:2 isn't in the enum)."""
    return "match_input_image"


def _to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=94)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _build_prompt(background_id: str, extra: Optional[str] = None) -> str:
    preset = backgrounds.get_preset(background_id)
    parts = [
        f"This image shows a car in this scene: {preset.prompt}.",
        # Body-relight mode: car + shadow + scene are already composited correctly.
        # We want Qwen to harmonize the *lighting* on the car body only.
        "The car is correctly placed and the ground shadow is correctly drawn. "
        "Do NOT move the car. Do NOT change the car's position, scale, orientation, "
        "or proportions. Do NOT add any extra cars, people, or objects. "
        "Do NOT alter the floor, walls, ground shadow, or background composition. "
        "Only adjust the lighting and ambient reflections on the car body so it "
        "matches the studio lighting. Keep the car's paint colour, badges, license "
        "plate digits, wheels, and trim exactly as they appear in the input.",
        "Output: the same image with subtly improved lighting on the car only.",
    ]
    if extra:
        parts.append(extra)
    return " ".join(parts)


async def harmonize(
    composite: Image.Image,
    background_id: str,
    extra_prompt: Optional[str] = None,
) -> Image.Image:
    """Send a rough composite to Qwen Image Edit and return the harmonized result.

    Falls back to the unmodified composite on any failure so the pipeline never
    hard-fails because of a model hiccup.
    """
    settings = get_settings()
    if not settings.replicate_enabled:
        log.warning("harmonize.skipped.no_replicate")
        return composite

    prompt = _build_prompt(background_id, extra_prompt)
    aspect_ratio = _aspect_ratio_for(composite.size)
    image_data_url = _to_data_url(composite)

    log.info("harmonize.start", model=_MODEL, aspect_ratio=aspect_ratio, prompt_chars=len(prompt))

    try:
        result_url = await _run_replicate(
            settings.replicate_api_token,
            prompt=prompt,
            image_data_url=image_data_url,
            aspect_ratio=aspect_ratio,
        )
        result_img = await _download(result_url)
    except Exception as exc:  # noqa: BLE001
        log.warning("harmonize.failed", error=str(exc))
        return composite

    # The model returns a webp; convert to RGB and resize to match canvas.
    if result_img.size != composite.size:
        result_img = result_img.resize(composite.size, Image.LANCZOS)
    return result_img.convert("RGB")


async def _run_replicate(token: str, *, prompt: str, image_data_url: str, aspect_ratio: str) -> str:
    """Create a prediction, poll until it finishes, return the output URL."""
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
        "Prefer": "wait=60",  # max allowed; we poll for the rest
    }
    payload = {
        "input": {
            "prompt": prompt,
            "image": [image_data_url],
            "aspect_ratio": aspect_ratio,
            "output_format": "jpg",
            "output_quality": 95,
            "go_fast": False,  # Higher quality is worth the extra latency for cars
        }
    }

    async with httpx.AsyncClient(timeout=_REPLICATE_TIMEOUT) as client:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=8),
            retry=retry_if_exception_type((httpx.HTTPError, ExternalServiceError)),
            reraise=True,
        ):
            with attempt:
                resp = await client.post(
                    f"https://api.replicate.com/v1/models/{_MODEL}/predictions",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code == 422:
                    # Schema validation error - log the body so we can see what was wrong
                    log.error("harmonize.422", body=resp.text[:1000])
                    raise ExternalServiceError(f"replicate 422: {resp.text[:500]}")
                if resp.status_code >= 500:
                    raise ExternalServiceError(f"replicate {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()

        prediction = resp.json()
        status = prediction.get("status")

        # If the wait header didn't get us all the way to terminal, poll.
        poll_url = prediction.get("urls", {}).get("get")
        deadline_polls = 60  # ~120s at 2s each
        while status not in {"succeeded", "failed", "canceled"} and deadline_polls > 0:
            await asyncio.sleep(2.0)
            deadline_polls -= 1
            if not poll_url:
                break
            r = await client.get(poll_url, headers={"Authorization": f"Token {token}"})
            r.raise_for_status()
            prediction = r.json()
            status = prediction.get("status")

    if status != "succeeded":
        raise ExternalServiceError(f"prediction status={status}: {prediction.get('error')}")

    output = prediction.get("output")
    if isinstance(output, list) and output:
        return output[0]
    if isinstance(output, str):
        return output
    raise ExternalServiceError(f"unexpected output shape: {output!r}")


async def _download(url: str) -> Image.Image:
    async with httpx.AsyncClient(timeout=_REPLICATE_TIMEOUT) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content))
    img.load()
    return img
