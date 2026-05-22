"""Pipeline orchestrator: runs stages in order, records timings, returns the result."""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from .. import backgrounds
from ..errors import PipelineError, ValidationError
from ..logging_setup import get_logger
from ..settings import get_settings
from . import (
    color_match,
    compose,
    decode,
    harmonize as harmonize_stage,
    plate_blur,
    preserve_car as preserve_car_stage,
    refine_mask,
    relight as relight_stage,
    segment,
    shadow,
    upscale as upscale_stage,
    watermark as watermark_stage,
)

log = get_logger(__name__)


@dataclass
class PipelineParams:
    background_id: str
    harmonize: bool = True   # When true, run Qwen Image Edit harmonization (recommended)
    preserve_car: bool = True  # When true, paste original car pixels over harmonized output
    relight: bool = False    # Legacy IC-Light pass; ignored when harmonize=True
    plate_blur: bool = False
    upscale: bool = False
    watermark_bytes: Optional[bytes] = None
    extra_prompt: Optional[str] = None   # Optional user-supplied scene tweak


@dataclass
class PipelineResult:
    image_bytes: bytes
    content_type: str
    width: int
    height: int
    timings_ms: dict[str, int] = field(default_factory=dict)


def _encode(image: Image.Image, quality: int) -> tuple[bytes, str]:
    buf = io.BytesIO()
    image.convert("RGB").save(
        buf,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        subsampling="4:2:0",
    )
    return buf.getvalue(), "image/jpeg"


async def run_pipeline(input_bytes: bytes, params: PipelineParams) -> PipelineResult:
    settings = get_settings()
    timings: dict[str, int] = {}

    def _stage(name: str):
        return _StageTimer(name, timings)

    try:
        # 1. Decode + normalize
        with _stage("decode"):
            source = decode.decode(input_bytes)

        # 2. Segment
        with _stage("segment"):
            cutout = await segment.segment_car(source)

        # 3. Refine mask
        with _stage("refine_mask"):
            cutout = refine_mask.refine(cutout)

        # 4. Compose (smart placement: anchor wheels to floor line)
        with _stage("compose"):
            composite, car_box, contact_y = compose.compose(cutout, params.background_id)

        # 5. Synthesize a real perspective shadow from the actual silhouette
        with _stage("shadow"):
            preset = backgrounds.get_preset(params.background_id)
            composite = shadow.add_shadow(
                composite, cutout, car_box, contact_y,
                light_direction=preset.light_direction,
            )

        if params.harmonize:
            # We send Qwen the FULL composite (car + shadow + scene) and ask it
            # only to harmonize the lighting on the car body. We then re-paste
            # the original car so identity is guaranteed.
            #
            # Keep a copy of the deterministic pre-harmonize composite so we
            # can use it as the safe background base when preserve_car is on.
            # That guarantees any duplicate/ghost car Qwen may hallucinate
            # outside the original car region cannot survive into the output.
            pre_harmonize_rgb = composite.convert("RGB")
            with _stage("harmonize"):
                rgb = await harmonize_stage.harmonize(
                    pre_harmonize_rgb,
                    params.background_id,
                    extra_prompt=params.extra_prompt,
                )
            if params.preserve_car:
                with _stage("preserve_car"):
                    rgb = preserve_car_stage.preserve_car(
                        rgb,
                        cutout,
                        car_box,
                        safe_base=pre_harmonize_rgb,
                    )
            composite_rgb = rgb
        else:
            # Classical fallback path
            with _stage("color_match"):
                cutout_alpha = cutout.split()[-1]
                composite = color_match.color_match(composite, car_box, cutout_alpha)
            composite_rgb = composite.convert("RGB")
            if params.relight:
                with _stage("relight"):
                    composite_rgb = await relight_stage.relight(composite_rgb, params.background_id)

        # 8. Optional plate blur
        if params.plate_blur:
            with _stage("plate_blur"):
                composite_rgb = await plate_blur.blur_plates(composite_rgb)

        # 9. Optional upscale
        if params.upscale:
            with _stage("upscale"):
                composite_rgb = await upscale_stage.upscale(composite_rgb)

        # 10. Optional watermark
        if params.watermark_bytes:
            with _stage("watermark"):
                composite_rgb = watermark_stage.apply_watermark(composite_rgb, params.watermark_bytes)

        # 11. Encode
        with _stage("encode"):
            image_bytes, content_type = _encode(composite_rgb, settings.jpeg_quality)

        log.info("pipeline.done", timings_ms=timings)
        return PipelineResult(
            image_bytes=image_bytes,
            content_type=content_type,
            width=composite_rgb.width,
            height=composite_rgb.height,
            timings_ms=timings,
        )

    except (PipelineError, ValidationError):
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline.failed")
        raise PipelineError(str(exc)) from exc


class _StageTimer:
    def __init__(self, name: str, sink: dict[str, int]) -> None:
        self.name = name
        self.sink = sink
        self.t0 = 0.0

    def __enter__(self) -> "_StageTimer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.sink[self.name] = int((time.perf_counter() - self.t0) * 1000)
