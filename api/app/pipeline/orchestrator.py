"""Pipeline orchestrator: runs stages in order, records timings, returns the result.

Pipeline shape (AI path, ``harmonize=True``):

    decode -> prep -> [studio short-circuit] -> segment -> refine_mask
        -> compose -> shadow -> reflection
        -> harmonize -> quality_guard -> preserve_car
        -> [plate_blur] -> [upscale] -> [watermark] -> encode

If the input already looks like a studio shot, segment / compose / shadow
/ reflection are skipped and harmonize runs on the prep'd photo
directly.

Pipeline shape (classical path, ``harmonize=False``):

    decode -> prep -> [studio short-circuit] -> segment -> refine_mask
        -> compose -> shadow -> reflection -> color_match -> [relight]
        -> [plate_blur] -> [upscale] -> [watermark] -> encode

Design rules:
    1. The deterministic stages (compose + shadow + reflection) are the
       source of truth for car placement. The AI never controls where the
       car is, how big it is, or which direction it faces.
    2. The AI's only job is to lightly polish lighting / reflections.
    3. quality_guard compares the pre-AI composite to the AI output and
       falls back to the pre-AI composite if a duplicate vehicle is
       suspected.
    4. preserve_car re-pastes the original cutout pixels with a high
       identity weight so badges, plates and trim are dealer-accurate.
"""
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
    prep as prep_stage,
    preserve_car as preserve_car_stage,
    quality_guard,
    refine_mask,
    reflection as reflection_stage,
    relight as relight_stage,
    scene_detect,
    segment,
    shadow,
    upscale as upscale_stage,
    watermark as watermark_stage,
)

log = get_logger(__name__)


@dataclass
class PipelineParams:
    background_id: str
    harmonize: bool = True   # AI finishing pass on top of the deterministic composite
    preserve_car: bool = True  # Re-paste original cutout for dealer-accurate identity
    relight: bool = False    # Legacy IC-Light pass; only runs in the classical path
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

        # 2. Prep: gentle denoise, white balance, highlight recovery
        if settings.enable_prep:
            with _stage("prep"):
                source = prep_stage.prep(
                    source,
                    denoise=settings.prep_denoise,
                    highlight_recover=settings.prep_highlight_recover,
                    white_balance=settings.prep_white_balance,
                    wb_strength=settings.prep_wb_strength,
                )

        # 2a. "Already in a studio" short-circuit. If the input is already
        #     on a clean wall we skip segment/compose/shadow and let
        #     harmonize lightly polish the original photo. The car_box is
        #     unset on this path so plate_blur falls back to whole-image
        #     search.
        already_studio = False
        if settings.enable_studio_shortcircuit and params.harmonize:
            with _stage("scene_detect"):
                already_studio = scene_detect.looks_like_studio(source)

        if already_studio:
            log.info("pipeline.studio_shortcircuit")
            with _stage("harmonize"):
                composite_rgb = (
                    await harmonize_stage.harmonize(
                        source.convert("RGBA"),
                        params.background_id,
                        extra_prompt=params.extra_prompt,
                    )
                ).convert("RGB")
            car_box: Optional[tuple[int, int, int, int]] = None
        else:
            # 3. Segment the car out of its original background.
            with _stage("segment"):
                cutout = await segment.segment_car(source)

            # 4. Refine mask (despill, fill interior holes, feather edge).
            with _stage("refine_mask"):
                cutout = refine_mask.refine(cutout)

            # 5. Compose: place the car on the studio background. Pass the
            #    *original* photo's size so we honour input orientation
            #    (portrait vs landscape) instead of guessing from the
            #    cropped cutout.
            with _stage("compose"):
                composite, car_box, contact_y = compose.compose(
                    cutout,
                    params.background_id,
                    source_size=source.size,
                )

            # 6. Synthesize a perspective shadow from the actual silhouette.
            with _stage("shadow"):
                preset = backgrounds.get_preset(params.background_id)
                composite = shadow.add_shadow(
                    composite, cutout, car_box, contact_y,
                    light_direction=preset.light_direction,
                )

            # 7. Floor reflection (tinted to floor colour) + chassis AO.
            #    Both anchored at the contact line; both deterministic.
            if settings.enable_reflection:
                with _stage("reflection"):
                    composite = reflection_stage.add_reflection(
                        composite, cutout, car_box, contact_y,
                        opacity=settings.reflection_opacity,
                        blur_radius=settings.reflection_blur,
                        floor_tint_strength=settings.reflection_floor_tint,
                        add_chassis_ao=settings.enable_chassis_ao,
                        ao_strength=settings.chassis_ao_strength,
                    )

            # Snapshot the deterministic composite for quality_guard
            # comparison and as the fallback if the guard fires.
            deterministic_rgb = composite.convert("RGB")

            if params.harmonize:
                # 8. AI finishing pass. Studio polish only.
                with _stage("harmonize"):
                    ai_rgb = await harmonize_stage.harmonize(
                        composite,
                        params.background_id,
                        extra_prompt=params.extra_prompt,
                    )

                # 9. Duplicate-vehicle safety check.
                if settings.enable_duplicate_guard:
                    with _stage("quality_guard"):
                        flagged = quality_guard.detect_possible_duplicate_vehicle(
                            deterministic_rgb, ai_rgb, car_box,
                        )
                    if flagged:
                        log.warning(
                            "harmonize.fallback.duplicate_suspected",
                            car_box=car_box,
                        )
                        ai_rgb = deterministic_rgb

                # 10. preserve_car: re-paste the original cutout with high
                #     identity strength so badges/plates/trim are dealer-accurate.
                if params.preserve_car:
                    with _stage("preserve_car"):
                        composite_rgb = preserve_car_stage.preserve_car(
                            ai_rgb,
                            cutout,
                            car_box,
                            car_identity_strength=settings.car_identity_strength,
                            adaptive=settings.preserve_car_adaptive,
                        )
                else:
                    composite_rgb = ai_rgb
            else:
                # Classical path: deterministic only.
                with _stage("color_match"):
                    cutout_alpha = cutout.split()[-1]
                    composite = color_match.color_match(composite, car_box, cutout_alpha)
                composite_rgb = composite.convert("RGB")
                if params.relight:
                    with _stage("relight"):
                        composite_rgb = await relight_stage.relight(
                            composite_rgb, params.background_id,
                        )

        # 11. Optional plate blur. car_box is None when we short-circuited
        #     into harmonize; the heuristic detector falls back to whole-
        #     image search in that case.
        if params.plate_blur:
            with _stage("plate_blur"):
                composite_rgb = await plate_blur.blur_plates(
                    composite_rgb, car_box=car_box,
                )

        # 12. Optional upscale
        if params.upscale:
            with _stage("upscale"):
                composite_rgb = await upscale_stage.upscale(composite_rgb)

        # 13. Optional watermark
        if params.watermark_bytes:
            with _stage("watermark"):
                composite_rgb = watermark_stage.apply_watermark(
                    composite_rgb, params.watermark_bytes,
                )

        # 14. Encode
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
