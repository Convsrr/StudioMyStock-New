# StudioMyStock

Turn raw car photos into studio-quality images with swappable backgrounds.

This is a prototype scaffold. It is intentionally minimal so you can iterate on the AI pipeline without fighting infrastructure.

## Architecture

```
web/   Next.js 14 (App Router) frontend
api/   FastAPI backend that orchestrates the AI pipeline via Replicate
```

The pipeline runs synchronously for now via FastAPI background tasks. Swap in Redis + a worker (RQ, Celery, or Arq) once you have real volume.

## AI pipeline (current prototype)

1. Background removal (Replicate: `851-labs/background-remover`)
2. Compositing onto a preset studio background (Pillow, server-side)
3. Optional relighting pass (Replicate: `zsxkib/ic-light`) so the car matches the new scene

Each stage is a function in `api/app/pipeline.py`. Add scratch removal, plate blur, upscaling, etc. as additional stages.

## Quick start

### 1. Backend

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your REPLICATE_API_TOKEN
./run.sh
```

API runs on http://localhost:8000.

### 2. Frontend

```bash
cd web
npm install
cp .env.local.example .env.local
npm run dev
```

App runs on http://localhost:3000.

## Endpoints

- `POST /api/process` — multipart upload (`file`, `background`, `relight`) returns `{ job_id }`
- `GET /api/jobs/{job_id}` — returns `{ status, result_url, original_url }`
- `GET /api/backgrounds` — list available preset backgrounds
- `GET /static/<path>` — serves uploaded and processed images

## What to build next

- Replace synchronous processing with a queue and worker pool
- Swap in a custom-trained relighting model (IC-Light is fine to start, not great for paint)
- Add bulk upload (the dealership use case)
- Add brand presets (saved background + watermark + crop config)
- Add S3/R2 storage instead of local disk
- Add auth (Clerk) and billing (Stripe) once the output quality is good enough to charge for
