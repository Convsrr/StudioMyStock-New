# Architecture

This doc describes how StudioMyStock fits together end-to-end: request flow, pipeline stages, storage, queueing, and the design choices behind them.

## Components

```
┌─────────────────────┐        ┌─────────────────────────────┐
│  Next.js frontend   │  HTTP  │  FastAPI backend (api/app)  │
│  (web/)             │◀──────▶│  - Routes                   │
│  App Router, React  │        │  - Pipeline orchestrator    │
└─────────────────────┘        │  - Storage abstraction      │
                               │  - Job repository (SQLite)  │
                               └──┬─────────────┬────────────┘
                                  │             │
                         inline   │             │  enqueue (arq + Redis)
                                  ▼             ▼
                               ┌──────────────────────────┐
                               │  Pipeline stages         │
                               │  decode → segment → …    │
                               │  → harmonize (Replicate) │
                               │  → preserve_car → encode │
                               └──────────────────────────┘
                                          │
                              ┌───────────┴────────────┐
                              ▼                        ▼
                       Picsart / Replicate      Local FS or S3
                       (segmentation, harmonize, (uploads/outputs)
                        upscale)
```

## Request lifecycle

1. **Upload** (`POST /api/process`)
   - Validates content type, size, requested background
   - Hashes input bytes + params for idempotency
   - If a successful job with the same hashes exists, returns it (`cached: true`)
   - Otherwise stores the upload, creates a `Job` row in `pending`, and dispatches it
2. **Dispatch**
   - If `INLINE_PROCESSING=true` or Redis is unreachable: runs in a FastAPI `BackgroundTask` in the same process
   - Otherwise: enqueues an arq job; the worker (`run-worker.sh`) picks it up
3. **Process** (worker or inline)
   - Marks job `running`, runs the pipeline, captures per-stage timings
   - On success: writes output to storage, marks `succeeded`, populates `result_url`
   - On failure: marks `failed` with `error_code` + `error_message`
4. **Poll** (`GET /api/jobs/{id}`)
   - Frontend polls with exponential backoff (1s → 8s) until terminal status

## Pipeline orchestrator

The orchestrator (`api/app/pipeline/orchestrator.py`) runs the stages in order. Every stage is its own module so you can swap, A/B, or remove one without touching the rest.

| # | Stage | Module | What it does |
|---|-------|--------|--------------|
| 1 | decode | `decode.py` | Loads bytes, EXIF-rotates, caps to `WORKING_MAX_DIM`, returns RGB PIL image |
| 2 | segment | `segment.py` (+ `picsart_client.py`, `replicate_client.py`) | Returns RGBA cutout. Provider chosen by `SEGMENTATION_PROVIDER` (`auto` prefers Picsart) |
| 3 | refine_mask | `refine_mask.py` | Feathers and cleans the alpha edge so the cutout doesn't look stamped |
| 4 | compose | `compose.py` (+ `contact.py`) | Detects wheel contact y on the cutout, anchors it to the preset's `floor_y_ratio`, scales car to ~78% width / 70% height |
| 5 | shadow | `shadow.py` | Synthesizes a real perspective shadow from the silhouette using the preset's `light_direction` |
| 6 | harmonize | `harmonize.py` | Sends the composite + scene prompt to Qwen Image Edit 2511 on Replicate. Returns a relit version of the whole frame |
| 7 | preserve_car | `preserve_car.py` | Pastes the original car pixels back using the segmentation alpha, with a small luma shift toward the harmonized scene to avoid a "pasted" look |
| 8 | plate_blur *(opt)* | `plate_blur.py` | Detects + blurs visible plates |
| 9 | upscale *(opt)* | `upscale.py` | Real-ESRGAN to lift to `OUTPUT_MAX_DIM` |
| 10 | watermark *(opt)* | `watermark.py` | Overlays a user-supplied PNG |
| 11 | encode | (in `orchestrator.py`) | Progressive JPEG, 4:2:0 subsampling, configurable quality |

### Why harmonize → preserve_car instead of "just" relighting

Generative models produce beautifully lit scenes but subtly alter the car: plate digits drift, badges blur, wheel patterns shift. For automotive listings that's unusable. The two-pass design uses Qwen for what it's great at (lighting, ground reflections, environment) and the original pixels for what they're great at (identity).

If `harmonize=false`, the pipeline takes the classical fallback path: color-match the cutout to the background, optionally run IC-Light for relighting. Faster, cheaper, lower quality.

## Background catalog

`backgrounds.py` defines five presets. Each preset has:

- `prompt` — fed to Qwen as the target scene description
- `floor_y_ratio` — where the floor line sits, used for compositing
- `light_direction` — `top` / `top-left` / `top-right`, used for shadow synthesis
- A pre-rendered JPG asset in `api/app/assets/backgrounds/` and a sized version cached in `api/storage/cache/backgrounds/`

To add a preset: append to `PRESETS`, drop in an asset image, optionally regenerate cached sizes via `scripts/render_backgrounds.py`.

## Storage

The `Storage` abstraction (`api/app/storage.py`) has two backends:

- **LocalStorage** — files on disk under `STORAGE_DIR`, served by FastAPI's `StaticFiles` at `/static/<key>`
- **S3Storage** — any S3-compatible service (AWS S3, Cloudflare R2, MinIO). Public URLs are constructed from `S3_PUBLIC_BASE_URL`

Keys follow the pattern `uploads/<job_id>.<ext>` and `outputs/<job_id>.jpg`.

## Database

SQLite by default (`sqlite+aiosqlite`), Postgres-ready (`postgresql+asyncpg`). One table:

- `jobs` — job state, params, hashes for idempotency, stage timings (JSON), error details, timestamps

The schema is intentionally flat. No multi-tenant model yet — `api_key_hash` is recorded so you can attribute usage later.

## Queue

Optional. Uses [arq](https://arq-docs.helpmanual.io/) on top of Redis. The producer side (`api/app/queue.py`) is configured to fail fast (`conn_retries=1`, `conn_timeout=1`) so when Redis is down, `enqueue_job_or_inline` falls back to inline processing without a noticeable delay.

The worker is a separate process (`api/app/worker.py`, started by `run-worker.sh`). It calls the same `run_pipeline` function as the inline path.

## Auth, rate limiting, observability

- **Auth**: `X-API-Key` header, comma-separated whitelist in `API_KEYS`. SHA-256 of the key is stored on the job for traceability. Empty `API_KEYS` disables auth (dev only).
- **Rate limit**: SlowAPI, default 30/min per IP on `/api/process`. Configurable via `RATE_LIMIT_PER_MINUTE`.
- **Logs**: structlog with request-scoped context (`request_id`, `path`).
- **Metrics**: Prometheus at `/metrics` — request count + latency histogram, jobs enqueued, idempotency cache hits.
- **Health**: `/api/health` reports queue connectivity and Replicate config.

## Frontend (`web/`)

Single-page app, no global state library. The flow:

1. Fetch `/api/backgrounds` on mount, render the picker
2. User uploads → preview locally
3. Submit to `/api/process` → store the returned job
4. Poll `/api/jobs/{id}` with exponential backoff (1s → 8s capped) until terminal
5. Render `result_url` next to `original_url` for the before/after view

API helpers live in `web/lib/api.ts`. `web/app/page.tsx` is the main view.

## Failure handling

Each stage either succeeds or raises a typed error:

- `ValidationError` → 400
- `ExternalServiceError` → 502 with the upstream code
- `PipelineError` → 500
- `RateLimitExceeded` → 429

External calls (Replicate, Picsart) use `tenacity` with exponential backoff. Stuck jobs can be cleaned up with `api/scripts/fail_stuck_jobs.py`.

## Performance notes

- Working dimension is capped at `WORKING_MAX_DIM` (default 2048) to keep Qwen latency bounded
- `OUTPUT_MAX_DIM` controls the upscale ceiling (default 3840)
- Background images are pre-rendered and cached at the canvas size to avoid resize on every request
- Idempotency cache means a re-submission with the same image + params is a single DB read

## Adding a new pipeline stage

1. Create `api/app/pipeline/<stage>.py` exposing one async or sync function with `(image, ...) -> image`
2. Add it to the imports in `orchestrator.py`
3. Insert a `with _stage("<name>"):` block where it should run
4. Add a `pytest` test under `api/tests/`
5. If it has a config knob, add it to `Settings` in `settings.py` and document it in `.env.example`
