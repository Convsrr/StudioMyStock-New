export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? "";

export type Background = {
  id: string;
  name: string;
  description: string;
};

export type JobStatus = "pending" | "running" | "succeeded" | "failed" | "cancelled";

export type Job = {
  id: string;
  status: JobStatus;
  background_id: string;
  harmonize: boolean;
  preserve_car: boolean;
  relight: boolean;
  plate_blur: boolean;
  upscale: boolean;
  original_url: string;
  result_url: string | null;
  error_code: string | null;
  error_message: string | null;
  duration_ms: number | null;
  stage_timings: Record<string, number> | null;
  created_at: string;
  updated_at: string;
  finished_at: string | null;
};

export type SubmitOptions = {
  harmonize?: boolean;
  preserveCar?: boolean;
  relight?: boolean;
  plateBlur?: boolean;
  upscale?: boolean;
  extraPrompt?: string;
  watermark?: File | null;
};

export type SubmitResponse = {
  job_id: string;
  status: JobStatus;
  cached: boolean;
};

function authHeaders(): HeadersInit {
  return API_KEY ? { "X-API-Key": API_KEY } : {};
}

async function parseError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (body?.error?.message) return body.error.message;
    return JSON.stringify(body);
  } catch {
    return res.statusText;
  }
}

export async function fetchBackgrounds(): Promise<Background[]> {
  const res = await fetch(`${API_BASE_URL}/api/backgrounds`, { cache: "no-store" });
  if (!res.ok) throw new Error(await parseError(res));
  const data = (await res.json()) as { backgrounds: Background[] };
  return data.backgrounds;
}

export async function submitJob(
  file: File,
  backgroundId: string,
  options: SubmitOptions = {}
): Promise<SubmitResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("background", backgroundId);
  form.append("harmonize", options.harmonize === false ? "false" : "true");
  form.append("preserve_car", options.preserveCar === false ? "false" : "true");
  form.append("relight", options.relight ? "true" : "false");
  form.append("plate_blur", options.plateBlur ? "true" : "false");
  form.append("upscale", options.upscale ? "true" : "false");
  if (options.extraPrompt) form.append("extra_prompt", options.extraPrompt);
  if (options.watermark) form.append("watermark", options.watermark);

  const res = await fetch(`${API_BASE_URL}/api/process`, {
    method: "POST",
    body: form,
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function fetchJob(jobId: string): Promise<Job> {
  const res = await fetch(`${API_BASE_URL}/api/jobs/${jobId}`, {
    cache: "no-store",
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}
