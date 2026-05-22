"use client";

import { useEffect, useRef, useState } from "react";
import { fetchBackgrounds, fetchJob, submitJob, type Background, type Job } from "@/lib/api";

const POLL_MIN_MS = 1000;
const POLL_MAX_MS = 8000;

export default function Home() {
  const [backgrounds, setBackgrounds] = useState<Background[]>([]);
  const [selectedBg, setSelectedBg] = useState<string>("studio-white");
  const [harmonize, setHarmonize] = useState(true);
  const [preserveCar, setPreserveCar] = useState(true);
  const [relight, setRelight] = useState(false);
  const [plateBlur, setPlateBlur] = useState(false);
  const [upscale, setUpscale] = useState(false);
  const [extraPrompt, setExtraPrompt] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchBackgrounds()
      .then((bgs) => {
        setBackgrounds(bgs);
        if (bgs.length && !bgs.some((b) => b.id === selectedBg)) {
          setSelectedBg(bgs[0].id);
        }
      })
      .catch((e) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!job || job.status === "succeeded" || job.status === "failed" || job.status === "cancelled") {
      return;
    }
    let cancelled = false;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    let delay = POLL_MIN_MS;

    const tick = async () => {
      if (cancelled) return;
      try {
        const next = await fetchJob(job.id);
        if (cancelled) return;
        setJob(next);
        if (next.status === "succeeded" || next.status === "failed" || next.status === "cancelled") {
          return;
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
      delay = Math.min(delay * 2, POLL_MAX_MS);
      timeout = setTimeout(tick, delay);
    };

    timeout = setTimeout(tick, delay);
    return () => {
      cancelled = true;
      if (timeout) clearTimeout(timeout);
    };
  }, [job]);

  function handleFile(f: File | null) {
    setFile(f);
    setJob(null);
    setError(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(f ? URL.createObjectURL(f) : null);
  }

  async function handleSubmit() {
    if (!file) return;
    setSubmitting(true);
    setError(null);
    try {
      const { job_id } = await submitJob(file, selectedBg, {
        harmonize,
        preserveCar,
        relight,
        plateBlur,
        upscale,
        extraPrompt: extraPrompt.trim() || undefined,
      });
      const initial = await fetchJob(job_id);
      setJob(initial);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  const isProcessing = job && (job.status === "pending" || job.status === "running");
  const result = job?.status === "succeeded" ? job.result_url : null;

  return (
    <div className="container">
      <div className="header">
        <h1>StudioMyStock</h1>
        <span className="tag">Prototype</span>
      </div>

      <div className="row">
        <div className="panel">
          <h3 className="section-title">1. Upload a car photo</h3>
          <div
            className={`dropzone ${dragging ? "dragging" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const f = e.dataTransfer.files?.[0];
              if (f) handleFile(f);
            }}
          >
            <strong>{file ? file.name : "Drop an image here or click to browse"}</strong>
            <p>JPEG, PNG, or WebP up to ~25MB</p>
            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              style={{ display: "none" }}
              onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
            />
          </div>

          <h3 className="section-title" style={{ marginTop: 24 }}>
            2. Pick a background
          </h3>
          <div className="bg-grid">
            {backgrounds.map((bg) => (
              <div
                key={bg.id}
                className={`bg-card ${selectedBg === bg.id ? "selected" : ""}`}
                onClick={() => setSelectedBg(bg.id)}
              >
                <div className="name">{bg.name}</div>
                <div className="desc">{bg.description}</div>
              </div>
            ))}
          </div>

          <h3 className="section-title" style={{ marginTop: 24 }}>
            3. Options
          </h3>
          <label className="toggle">
            <input type="checkbox" checked={harmonize} onChange={(e) => setHarmonize(e.target.checked)} />
            AI harmonize (Qwen Image Edit, ~$0.03/image, 30-60s)
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={preserveCar}
              onChange={(e) => setPreserveCar(e.target.checked)}
              disabled={!harmonize}
            />
            Preserve original car (paste exact pixels back over harmonized scene)
          </label>
          <label className="toggle">
            <input type="checkbox" checked={relight} onChange={(e) => setRelight(e.target.checked)} disabled={harmonize} />
            Legacy relight (IC-Light, ignored when AI harmonize is on)
          </label>
          <label className="toggle">
            <input type="checkbox" checked={plateBlur} onChange={(e) => setPlateBlur(e.target.checked)} />
            Blur license plates
          </label>
          <label className="toggle">
            <input type="checkbox" checked={upscale} onChange={(e) => setUpscale(e.target.checked)} />
            Upscale to 4K
          </label>
          <div style={{ marginTop: 12 }}>
            <input
              type="text"
              value={extraPrompt}
              onChange={(e) => setExtraPrompt(e.target.value)}
              placeholder="Optional scene tweak (e.g. 'sunset light from the left')"
              style={{
                width: "100%",
                padding: "10px 12px",
                background: "var(--panel-2)",
                border: "1px solid var(--border)",
                color: "var(--text)",
                borderRadius: 8,
                fontSize: 14,
              }}
            />
          </div>

          <div style={{ marginTop: 20 }}>
            <button
              className="button"
              onClick={handleSubmit}
              disabled={!file || submitting || !!isProcessing}
            >
              {submitting || isProcessing ? <span className="spinner" /> : null}
              {isProcessing ? "Processing..." : "Generate"}
            </button>
          </div>

          {error && <div className="status error">{error}</div>}
          {job?.status === "failed" && (
            <div className="status error">
              {job.error_code ? `[${job.error_code}] ` : ""}
              {job.error_message ?? "Job failed"}
            </div>
          )}
          {isProcessing && (
            <div className="status info">
              {job?.status === "pending" ? "Queued..." : "Running pipeline..."}
            </div>
          )}
          {job?.status === "succeeded" && job.duration_ms != null && (
            <div className="status info">
              Done in {(job.duration_ms / 1000).toFixed(1)}s
              {job.stage_timings ? ` (${Object.keys(job.stage_timings).length} stages)` : ""}
            </div>
          )}
        </div>

        <div className="panel">
          <h3 className="section-title">Preview</h3>
          <div className="preview">
            <div>
              <div className="label">Original</div>
              <div className="frame">
                {previewUrl ? (
                  <img src={previewUrl} alt="Original" />
                ) : (
                  <span style={{ color: "#9aa0b4" }}>No image yet</span>
                )}
              </div>
            </div>
            <div>
              <div className="label">Result</div>
              <div className="frame">
                {result ? (
                  <img src={result} alt="Result" />
                ) : (
                  <span style={{ color: "#9aa0b4" }}>{isProcessing ? "Working..." : "—"}</span>
                )}
              </div>
            </div>
          </div>
          {result && (
            <div style={{ marginTop: 16 }}>
              <a className="button" href={result} download>
                Download
              </a>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
