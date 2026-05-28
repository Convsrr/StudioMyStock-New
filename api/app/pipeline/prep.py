"""Pre-segmentation cleanup of the input photo.

A phone snap on a forecourt usually has at least one of:

    - mild luminance noise from low light or high ISO
    - blown-out highlights on chrome / bumper / sky reflection
    - a strong tungsten or warm sunset cast on the car body

These all compound through the rest of the pipeline. Segmentation gets
worse near noisy / clipped pixels; preserve_car keeps 92% of the original
car's colour cast which then sticks out against a neutral studio.

This stage runs once, right after decode, and produces a cleaner working
image. It is intentionally gentle - we are not retouching, we are
removing the most obvious technical defects of phone photography so the
rest of the pipeline has a fighting chance.

Knobs are exposed via ``Settings`` so the whole stage can be disabled or
tuned per environment.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from ..logging_setup import get_logger

log = get_logger(__name__)


def prep(
    image: Image.Image,
    *,
    denoise: bool = True,
    highlight_recover: bool = True,
    white_balance: bool = True,
    wb_strength: float = 0.6,
) -> Image.Image:
    """Return a cleaned-up RGB working image.

    The ordering matters:

        1. denoise first, so noise doesn't fool downstream stats
        2. white balance next, so highlight recovery operates on neutral pixels
        3. highlight recovery last, so we don't pull noise back out of the
           clipped regions

    All three steps are tuned to be gentle. The goal is to make a phone
    photo *look like itself, on a calmer day*; not to retouch.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    out = image
    if denoise:
        out = _gentle_denoise(out)
    if white_balance:
        out = _gray_world_white_balance(out, strength=wb_strength)
    if highlight_recover:
        out = _recover_highlights(out)
    return out


# --- Denoise --------------------------------------------------------------


def _gentle_denoise(image: Image.Image) -> Image.Image:
    """Edge-preserving smoothing.

    Pillow's ``SMOOTH_MORE`` is too aggressive on textures; ``MedianFilter(3)``
    is great at salt-and-pepper noise but blurs fine paint flake. We use a
    very small Gaussian (~0.5 px) which removes obvious luminance grain
    without softening visible details.
    """
    return image.filter(ImageFilter.GaussianBlur(radius=0.5))


# --- Auto white balance ---------------------------------------------------


def _gray_world_white_balance(image: Image.Image, strength: float) -> Image.Image:
    """Gray-world white balance.

    Assumption: averaged across a busy scene, the world is grey. If the
    average colour skews warm or cool, the camera's white balance was off
    and we correct it.

    ``strength`` blends the corrected image back toward the original so
    we never go overboard. 0.0 = no change; 1.0 = full grey-world.

    We weight the average toward the centre and away from very dark or
    very bright pixels to avoid being pulled by big sky regions or asphalt.
    """
    arr = np.asarray(image, dtype=np.float32)
    H, W, _ = arr.shape

    # Mid-tone weight: triangle peaking at 128. Excludes near-black and
    # near-white pixels which carry no useful colour information.
    luma = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
    midtone = np.maximum(0.0, 1.0 - np.abs(luma - 128.0) / 128.0)

    # Centre weight: small bias toward the middle of the frame so big
    # background areas don't dominate.
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2.0, W / 2.0
    radial = np.sqrt(((xx - cx) / W) ** 2 + ((yy - cy) / H) ** 2)
    centre = np.clip(1.0 - 0.6 * radial, 0.4, 1.0)

    weight = midtone * centre
    total = weight.sum() + 1e-6
    if total < 1.0:
        return image

    means = (arr * weight[..., None]).sum(axis=(0, 1)) / total
    target = float(means.mean())
    if target < 1.0:
        return image

    # Per-channel gain to neutralise the cast.
    gains = target / np.maximum(means, 1.0)

    # Clamp gains so we never wildly tint an image that already had a
    # legitimate strong colour (a red car against a brick wall).
    gains = np.clip(gains, 0.85, 1.18)

    corrected = arr * gains[None, None, :]
    blended = arr * (1.0 - strength) + corrected * strength
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    log.info(
        "prep.wb",
        gains_r=round(float(gains[0]), 3),
        gains_g=round(float(gains[1]), 3),
        gains_b=round(float(gains[2]), 3),
        strength=strength,
    )
    return Image.fromarray(blended, mode="RGB")


# --- Highlight recovery ---------------------------------------------------


def _recover_highlights(image: Image.Image) -> Image.Image:
    """Pull back near-clipped highlights so chrome / bonnet reflections
    keep some structure.

    For pixels above ~245 in any channel, soft-roll them down toward 240
    using a smooth shoulder. The compression is per-pixel so untouched
    midtones stay exactly where they were.
    """
    arr = np.asarray(image, dtype=np.float32)
    knee = 235.0
    ceiling = 252.0
    # Above the knee we apply a smooth roll-off:
    # x' = knee + (ceiling - knee) * tanh((x - knee) / (ceiling - knee))
    over = arr > knee
    if not over.any():
        return image
    rolled = arr.copy()
    delta = (arr - knee) / (ceiling - knee)
    rolled[over] = knee + (ceiling - knee) * np.tanh(delta[over])
    rolled = np.clip(rolled, 0, 255).astype(np.uint8)
    return Image.fromarray(rolled, mode="RGB")
