# StudioMyStock

Turn raw car photos into studio-quality images with swappable backgrounds. Built for dealerships and marketplaces that want catalog-grade product shots without renting a studio.

The pipeline cuts the car out of the source photo, drops it onto a chosen studio background, generates a perspective-correct shadow, then uses Qwen Image Edit to harmonize the lighting before pasting the original car pixels back on top so the make/model/plate/badges are pixel-identical.

## Repo layout

```
api/   FastAPI backend, async pipeline, job queue, storage
web/   Next.js 14 (App Router) frontend
```

## Studio backgrounds

Five presets ship with the app:

| ID | Name | Mood |
|----|------|------|
| `studio-white` | Studio White | Clean white cyc, light concrete floor |
| `studio-grey` | Studio Grey | Light cyc, mid-grey tile floor |
| `studio-charcoal` | Studio Charcoal | Moody dark cyc, polished black floor |
| `studio-warm` | Studio Warm | Warm cream cyc, sandstone floor |
| `studio-blueprint` | Studio Blueprint | Cool blue-tinted cyc, light tile floor |

Each preset carries a scene prompt and a floor-line ratio so the car gets anchored at the right horizon.

## Pipeline at a glance

1. **decode** — load and normalize the input image
2. **prep** — gentle denoise, auto white balance, highlight recovery
3. **scene_detect** — detect if the input is already a studio shot (opt-in short-circuit)
4. **segment** — background removal (Picsart by default, Replicate as fallback; production fails loudly if unconfigured)
5. **refine_mask** — feather and clean the alpha edge
6. **compose** — anchor the wheel contact line to the preset's floor line
7. **shadow** — synthesize a real perspective shadow from the silhouette
8. **reflection** — deterministic floor reflection + chassis ambient occlusion
9. **harmonize** — Qwen Image Edit 2511 (Replicate) relights the whole composite; fallback/degradation exposed in `stage_timings`
10. **quality_guard** — detect AI-introduced duplicate cars, fall back to deterministic composite
11. **preserve_car** — paste the original car pixels back over the harmonized scene
12. **plate_blur** *(optional)* — blur visible license plates (production requires a configured detector)
13. **upscale** *(optional)* — Real-ESRGAN
14. **watermark** *(optional)* — overlay a user-supplied logo
15. **encode** — JPEG output (configurable quality)

See [ARCHITECTURE.md](./ARCHITECTURE.md) for how the stages fit together and how the system handles jobs, storage, and failures. See [BUSINESS_DNA.md](./BUSINESS_DNA.md) for the product positioning and the principles every change should pass.

## Quick start

### Prereqs

- Python 3.11+
- Node 18+
- A Replicate API token (for harmonize / segmentation fallback / optional upscale)
- A Picsart API key (recommended for primary segmentation)
- Optional: Redis if you want a real worker queue. Without Redis, processing runs inline via FastAPI background tasks.

### 1. Backend

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in REPLICATE_API_TOKEN and PICSART_API_KEY in .env
./run.sh
```

API runs on http://localhost:8000.

If you want a worker queue, also start Redis (`brew services start redis`) and run:

```bash
./run-worker.sh
```

If Redis isn't reachable, the API silently falls back to inline processing — fine for dev, fine for single-host deployments.

### 2. Frontend

```bash
cd web
npm install
cp .env.local.example .env.local
npm run dev
```

App runs on http://localhost:3000.

## API

| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/api/process` | Multipart upload. Form fields: `file`, `background`, `harmonize`, `preserve_car`, `relight`, `plate_blur`, `upscale`, `extra_prompt`, optional `watermark` file. Returns `{ job_id, status, cached }`. |
| `GET`  | `/api/jobs/{id}` | Returns full job status, stage timings, `result_url`, `original_url`, error details. |
| `GET`  | `/api/backgrounds` | Lists available presets. |
| `GET`  | `/api/health` | Reports Replicate config, queue connectivity, storage backend. |
| `GET`  | `/metrics` | Prometheus exposition. |
| `GET`  | `/static/<path>` | Serves uploads and outputs when `STORAGE_BACKEND=local`. |

Auth: set `API_KEYS=key1,key2` in env to require an `X-API-Key` header. Empty value disables auth (dev only).

Idempotency: identical input bytes + identical params (including watermark bytes and content type) return the previously succeeded job instead of creating a new one.

## Configuration

The full set of env keys is in [`api/.env.example`](./api/.env.example). Highlights:

- `REPLICATE_API_TOKEN`, `PICSART_API_KEY` — provider credentials
- `SEGMENTATION_PROVIDER` — `auto` (default), `picsart`, `replicate`, or `none`. In production, `none` raises an error.
- `REPLICATE_PLATE_DETECTOR` — model ref for plate detection. Required in production if `plate_blur` is used.
- `INLINE_PROCESSING` — `true` to bypass Redis even when configured
- `STORAGE_BACKEND` — `local` or `s3` (S3-compatible: AWS, R2, MinIO). Local storage is blocked in production unless `ALLOW_PUBLIC_LOCAL_STORAGE=true`.
- `S3_PUBLIC_BASE_URL` — if empty, S3 URLs default to presigned URLs (TTL controlled by `S3_PRESIGNED_URL_TTL_SECONDS`, default 3600s)
- `ALLOW_STUDIO_SHORTCIRCUIT_PASSTHROUGH` — `false` by default; the studio short-circuit that skips segment/compose is opt-in so chosen backgrounds are always honoured
- `JOB_TIMEOUT_SECONDS` — worker job timeout (default 300s)
- `RATE_LIMIT_PER_MINUTE` — per-IP rate limit on `/api/process`
- `JPEG_QUALITY`, `OUTPUT_MAX_DIM` — output quality knobs

## Tests

```bash
cd api
source .venv/bin/activate
pytest -q
```

Tests use a temporary SQLite DB and stub external providers, so no API keys are required to run them. `PICSART_API_KEY` is explicitly cleared in the test conftest to prevent accidental network calls.

Test coverage includes regression tests for production-readiness behaviours: idempotency hashing, segmentation enforcement, plate blur enforcement, storage backend restrictions, harmonize fallback exposure, and worker timeout configuration.

## License

Proprietary.
