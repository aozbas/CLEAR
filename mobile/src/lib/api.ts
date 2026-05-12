import { supabase } from "./supabase";

const BASE = process.env.EXPO_PUBLIC_API_URL ?? "http://localhost:8000";

export type PredictionResponse = {
  label: string;
  confidence: number;
  image_url: string;
  signed_image_url?: string | null;
  scan_id: string;
};

export type Scan = {
  id: string;
  image_url: string;
  signed_image_url?: string | null;
  label: string;
  confidence: number;
  created_at: string;
};

async function authHeader(): Promise<Record<string, string>> {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function parseOrThrow<T>(r: Response): Promise<T> {
  if (!r.ok) {
    const body = await r.text().catch(() => "");
    if (r.status === 415) {
      throw new Error("Use a JPEG or PNG image.");
    }

    let detail = body;
    try {
      const parsed = JSON.parse(body) as { detail?: unknown };
      if (typeof parsed.detail === "string") detail = parsed.detail;
    } catch {
      // Keep the raw response body when the backend does not return JSON.
    }

    throw new Error(`HTTP ${r.status} ${r.statusText}${detail ? `: ${detail}` : ""}`);
  }
  return r.json();
}

export async function predict(imageUri: string): Promise<PredictionResponse> {
  const form = new FormData();
  // @ts-expect-error RN FormData file shape
  form.append("image", { uri: imageUri, name: "scan.jpg", type: "image/jpeg" });
  const r = await fetch(`${BASE}/predictions`, {
    method: "POST",
    body: form,
    headers: await authHeader(),
  });
  return parseOrThrow<PredictionResponse>(r);
}

export async function listScans(): Promise<{ scans: Scan[] }> {
  const r = await fetch(`${BASE}/scans`, { headers: await authHeader() });
  return parseOrThrow<{ scans: Scan[] }>(r);
}
