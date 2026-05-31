"""FastAPI entrypoint."""
import hashlib
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import structlog
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import Response

from . import backgrounds
from .auth import require_api_key
from .db import init_db, session_dependency
from .errors import StudioError, ValidationError
from .logging_setup import configure_logging, get_logger
from .models import JobStatus
from .pipeline import PipelineParams, run_pipeline
from .queue import close_pool, enqueue_job
from .repositories import JobRepository
from .schemas import (
    BackgroundsResponse,
    HealthResponse,
    JobCreatedResponse,
    JobOut,
)
from .settings import get_settings
from .storage import get_storage

log = get_logger(__name__)

# Bump this whenever the rendering pipeline changes in a way that would make
# previously-cached outputs visually different. It is mixed into params_hash so
# old succeeded jobs are not reused after a meaningful pipeline change.
#
# History:
#   v1 - initial release
#   v2 - 2026-05-22: removed safe_base default in preserve_car so Qwen's
#        harmonized image is the canvas again (was producing flat,
#        pasted-looking results)
#   v3 - 2026-05-22: preserve_car uses ghost_guard halo to override Qwen
#        within a feathered ring around the car silhouette, killing
#        duplicate plates / ghost bumpers without flattening the scene
#   v4 - 2026-05-22: tighter Qwen prompt (no plates, no bumper extension,
#        nothing inside silhouette), refine_mask fills interior alpha
#        holes, ghost_guard halo extended downward to cover floor zone
#   v5 - 2026-05-22: scene-only mode - car is painted out of Qwen's input
#        so it builds an empty studio with no vehicle to duplicate. Real
#        car is pasted back via preserve_car. Eliminates the duplicate-car
#        failure mode at the source.
#   v6 - 2026-05-22: shadow is now a separate alpha mask applied
#        multiplicatively to Qwen's harmonized floor in preserve_car, so
#        the contact shadow lands on the real studio floor instead of
#        being erased by ghost_guard or pre-baked into the deterministic
#        background. Fixes the "car looks like a sticker" problem.
#   v7 - 2026-05-22: switch to Qwen Image Edit 2509 Plus + Fusion LoRA
#        (dx8152/Qwen-Image-Edit-2509-Fusion). Pass the original car photo
#        and the target studio background as two reference images to a
#        model trained specifically for product-into-background blending
#        with perspective and lighting correction. Drops the homemade
#        compose / scene_only / ghost_guard / shadow_mask infrastructure
#        on the AI path - the LoRA handles all of that natively.
#   v8 - 2026-05-22: deterministic-first architecture. compose + shadow +
#        new reflection stage place the car; Qwen 2511 is downgraded to a
#        finishing-only polish pass; quality_guard catches AI-introduced
#        duplicate cars and falls back to the deterministic composite;
#        preserve_car gets a configurable identity-strength blend.
#   v9 - 2026-05-28: image-processing overhaul.
#        - new prep stage (denoise + auto white balance + highlight
#          recovery) before segmentation
#        - new studio short-circuit that skips segment/compose for
#          inputs already on a clean wall
#        - compose picks landscape, square or portrait canvas from the
#          input's aspect ratio; backgrounds are centre-cropped to the
#          target aspect instead of stretched
#        - wheel-contact detector adds a two-blob path for 3/4 angles
#          and trucks
#        - reflection now tints toward the sampled floor colour and
#          adds a chassis ambient-occlusion blob under the silhouette
#        - preserve_car classifies chrome / glass / paint and lets more
#          of Qwen's lighting through on reflective surfaces while
#          keeping plates and badges at full identity
#        - plate_blur gains a local heuristic detector for when the
#          Replicate detector model is not configured
#   v10 - 2026-05-28: ghost-car / sticker fixes.
#        - refine_mask now floods from *every* transparent border pixel
#          before deciding what's an interior hole, so the gap between
#          the wheels stops getting filled when the silhouette touches
#          the bottom corners of its bbox
#        - contact detection runs on the pre-fill ``raw_alpha`` (passed
#          through from refine_mask) so the two-wheel detector can see
#          the gap even when interior fill closed it
#        - quality_guard gains a luma-blob signal for soft / blurry
#          duplicate cars and tightens the safe halo to 15%
#        - preserve_car chrome / glass identity floors raised from
#          0.55 / 0.65 to 0.78 / 0.75
#        - harmonize prompt repeats and strengthens the ghost-car
#          prohibition
#   v11 - 2026-05-28: preset asset / prompt alignment fixes.
#        - re-rendered studio-charcoal, studio-warm and studio-blueprint
#          assets so they actually match their style cues (charcoal:
#          horizon-band specular polished floor; warm: single warm
#          overhead glow + sandstone concrete with perspective; blueprint:
#          properly cool blue cyc + real tile floor with grout)
#        - per-preset prompts switched from "target scene description"
#          to short "style cues" so Qwen no longer feels obliged to
#          repaint walls and floor to satisfy a rich description
#        - harmonize prompt now pins existing wall colour, floor colour,
#          tile pattern, horizon line and light placement; explicit
#          prohibitions on adding tile grids / spotlights / windows /
#          architectural features
#        - quality_guard gets a reflection-band exemption: a horizontal
#          dark blob whose centroid sits in the deterministic floor
#          reflection band (~22% of car height immediately below the
#          car) is no longer treated as a duplicate-vehicle candidate
#   v12 - 2026-05-31: production-readiness fixes.
#        - watermark bytes are included in idempotency hashing
#        - provider degradation is exposed in stage_timings
#        - studio passthrough is opt-in so chosen backgrounds are honoured
#        - background cache keys include asset mtimes
#        - optional prompts are constrained before final safety rules
#        - production plate blur requires a configured detector
PIPELINE_VERSION = "12"

REQUEST_COUNT = Counter("studio_requests_total", "Total HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("studio_request_seconds", "Request latency", ["method", "path"])
JOBS_ENQUEUED = Counter("studio_jobs_enqueued_total", "Jobs enqueued")
JOBS_CACHE_HIT = Counter("studio_jobs_cache_hit_total", "Jobs served from idempotency cache")


async def enqueue_job_or_inline(job_id: str, background_tasks: BackgroundTasks) -> None:
    """Pick inline or queue based on settings.

    Falls back to inline processing if the queue is configured but unreachable
    (e.g. Redis isn't running locally). This keeps the dev experience smooth.
    """
    settings = get_settings()
    if settings.inline_processing:
        background_tasks.add_task(_run_job_inline, job_id)
        return
    try:
        await enqueue_job(job_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("queue.unavailable.fallback_inline", error=str(exc))
        background_tasks.add_task(_run_job_inline, job_id)


async def _run_job_inline(job_id: str) -> None:
    """Process a job in-process. Used when INLINE_PROCESSING=true (no worker)."""
    from .db import get_sessionmaker

    storage = get_storage()
    sm = get_sessionmaker()

    async with sm() as session:
        repo = JobRepository(session)
        job = await repo.get(job_id)
        if job.status in {JobStatus.succeeded, JobStatus.cancelled}:
            return
        await repo.mark_running(job_id)

    try:
        input_bytes = await storage.get(job.input_key)
        watermark_bytes = None
        if job.watermark_key:
            try:
                watermark_bytes = await storage.get(job.watermark_key)
            except Exception:  # noqa: BLE001
                watermark_bytes = None
        params = PipelineParams(
            background_id=job.background_id,
            harmonize=job.harmonize,
            preserve_car=job.preserve_car,
            relight=job.relight,
            plate_blur=job.plate_blur,
            upscale=job.upscale,
            watermark_bytes=watermark_bytes,
            extra_prompt=job.extra_prompt,
        )
        result = await run_pipeline(input_bytes, params)
        output_key = f"outputs/{job_id}.jpg"
        public_url = await storage.put(output_key, result.image_bytes, result.content_type)
    except StudioError as exc:
        async with sm() as session:
            await JobRepository(session).mark_failed(job_id, exc.code, exc.message)
        log.warning("inline.failed", job_id=job_id, code=exc.code)
        return
    except Exception as exc:  # noqa: BLE001
        async with sm() as session:
            await JobRepository(session).mark_failed(job_id, "internal_error", str(exc))
        log.exception("inline.crashed", job_id=job_id)
        return

    async with sm() as session:
        try:
            await JobRepository(session).mark_succeeded(job_id, output_key, public_url, result.timings_ms)
            log.info("inline.succeeded", job_id=job_id, url=public_url)
        except Exception as exc:  # noqa: BLE001
            log.exception("inline.mark_succeeded_failed", job_id=job_id, error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.app_env != "development")
    await init_db()
    log.info("app.startup", env=settings.app_env, storage=settings.storage_backend)
    try:
        yield
    finally:
        await close_pool()
        log.info("app.shutdown")


limiter = Limiter(key_func=get_remote_address, default_limits=[])


def _create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="StudioMyStock API", version="0.1.0", lifespan=lifespan)
    app.state.limiter = limiter

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            structlog.contextvars.clear_contextvars()
            raise
        latency = time.perf_counter() - t0
        response.headers["x-request-id"] = request_id
        try:
            REQUEST_COUNT.labels(request.method, request.url.path, str(response.status_code)).inc()
            REQUEST_LATENCY.labels(request.method, request.url.path).observe(latency)
        finally:
            structlog.contextvars.clear_contextvars()
        return response

    @app.exception_handler(StudioError)
    async def studio_error_handler(_: Request, exc: StudioError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(_: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"error": {"code": "rate_limited", "message": str(exc.detail)}},
        )

    if settings.storage_backend == "local":
        app.mount("/static", StaticFiles(directory=str(settings.storage_dir)), name="static")

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    settings = get_settings()
    rl = f"{settings.rate_limit_per_minute}/minute"

    @app.get("/api/health", response_model=HealthResponse)
    async def health():
        if settings.inline_processing:
            queue_ok = True
        else:
            queue_ok = False
            try:
                from .queue import get_pool

                pool = await get_pool()
                await pool.ping()
                queue_ok = True
            except Exception:  # noqa: BLE001
                queue_ok = False
        return HealthResponse(
            ok=True,
            replicate_configured=settings.replicate_enabled,
            storage_backend=settings.storage_backend,
            queue_connected=queue_ok,
        )

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/backgrounds", response_model=BackgroundsResponse)
    async def list_backgrounds():
        return BackgroundsResponse(backgrounds=backgrounds.list_presets())

    @app.post("/api/process", response_model=JobCreatedResponse)
    @limiter.limit(rl)
    async def process_image(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        background: str = Form(...),
        harmonize: bool = Form(True),
        preserve_car: bool = Form(True),
        relight: bool = Form(False),
        plate_blur: bool = Form(False),
        upscale: bool = Form(False),
        extra_prompt: Optional[str] = Form(None),
        watermark: Optional[UploadFile] = File(default=None),
        api_key_hash: Optional[str] = Depends(require_api_key),
        session: AsyncSession = Depends(session_dependency),
    ):
        if not backgrounds.is_valid(background):
            raise ValidationError(f"Unknown background: {background}")
        if file.content_type and file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValidationError("Only JPEG, PNG, or WebP images are accepted")

        body = await file.read()
        if not body:
            raise ValidationError("Empty file")

        if extra_prompt is not None:
            extra_prompt = " ".join(extra_prompt.split())[:500] or None

        watermark_bytes: Optional[bytes] = None
        watermark_content_type = ""
        watermark_hash = "none"
        if watermark is not None:
            watermark_bytes = await watermark.read()
            if watermark_bytes:
                watermark_content_type = watermark.content_type or "image/png"
                watermark_hash = hashlib.sha256(
                    watermark_content_type.encode() + b"\0" + watermark_bytes
                ).hexdigest()

        storage = get_storage()
        repo = JobRepository(session)

        # Idempotency: same input + same params returns the existing reusable job.
        # Reusable means pending, running, or succeeded. Failed/cancelled jobs do
        # not block retries.
        input_hash = hashlib.sha256(body).hexdigest()
        params_hash = hashlib.sha256(
            (
                f"v{PIPELINE_VERSION}|{background}|{harmonize}|{preserve_car}|"
                f"{relight}|{plate_blur}|{upscale}|{extra_prompt or ''}|"
                f"watermark:{watermark_hash}"
            ).encode()
        ).hexdigest()

        existing = await repo.find_existing_by_hash(input_hash, params_hash)
        if existing:
            JOBS_CACHE_HIT.inc()
            # cached=True for any reused job (so existing clients keep working).
            # reused=True specifically means an in-flight job (pending/running) was
            # reused rather than a succeeded result.
            is_succeeded = existing.status == JobStatus.succeeded
            return JobCreatedResponse(
                job_id=existing.id,
                status=existing.status,
                cached=True,
                reused=not is_succeeded,
            )

        job_id = uuid.uuid4().hex
        suffix = Path(file.filename or "").suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        input_key = f"uploads/{job_id}{suffix}"
        watermark_key: Optional[str] = None

        await storage.put(input_key, body, file.content_type or "image/jpeg")
        original_url = storage.public_url(input_key)

        if watermark_bytes:
            watermark_key = f"uploads/{job_id}_wm.png"
            await storage.put(watermark_key, watermark_bytes, watermark_content_type or "image/png")

        await repo.create(
            id=job_id,
            api_key_hash=api_key_hash,
            background_id=background,
            harmonize=harmonize,
            preserve_car=preserve_car,
            relight=relight,
            plate_blur=plate_blur,
            upscale=upscale,
            input_key=input_key,
            original_url=original_url,
            watermark_key=watermark_key,
            extra_prompt=extra_prompt,
            input_hash=input_hash,
            params_hash=params_hash,
        )

        await enqueue_job_or_inline(job_id, background_tasks)
        JOBS_ENQUEUED.inc()
        return JobCreatedResponse(job_id=job_id, status=JobStatus.pending, cached=False)

    @app.get("/api/jobs/{job_id}", response_model=JobOut)
    async def get_job(
        job_id: str,
        session: AsyncSession = Depends(session_dependency),
        _: Optional[str] = Depends(require_api_key),
    ):
        repo = JobRepository(session)
        job = await repo.get(job_id)
        return JobOut(
            id=job.id,
            status=job.status,
            background_id=job.background_id,
            harmonize=job.harmonize,
            preserve_car=job.preserve_car,
            relight=job.relight,
            plate_blur=job.plate_blur,
            upscale=job.upscale,
            original_url=job.original_url,
            result_url=job.result_url,
            error_code=job.error_code,
            error_message=job.error_message,
            duration_ms=job.duration_ms,
            stage_timings=job.stage_timings,
            created_at=job.created_at,
            updated_at=job.updated_at,
            finished_at=job.finished_at,
        )


app = _create_app()
