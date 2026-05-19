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

async function authHeader(forceRefresh = false): Promise<Record<string, string>> {
  const { data, error } = forceRefresh
    ? await supabase.auth.refreshSession()
    : await supabase.auth.getSession();
  if (error) {
    await supabase.auth.signOut({ scope: "local" });
    throw new Error("Session expired. Please sign in again.");
  }

  const token = data.session?.access_token;
  if (!token) {
    await supabase.auth.signOut({ scope: "local" });
    throw new Error("Please sign in to continue.");
  }

  return { Authorization: `Bearer ${token}` };
}

async function fetchWithAuth(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = { ...(init.headers as Record<string, string> | undefined), ...(await authHeader()) };
  const response = await fetch(`${BASE}${path}`, { ...init, headers });
  if (response.status !== 401) return response;

  const retryHeaders = {
    ...(init.headers as Record<string, string> | undefined),
    ...(await authHeader(true)),
  };
  return fetch(`${BASE}${path}`, { ...init, headers: retryHeaders });
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
  const r = await fetchWithAuth("/predictions", {
    method: "POST",
    body: form,
  });
  return parseOrThrow<PredictionResponse>(r);
}

export async function listScans(): Promise<{ scans: Scan[] }> {
  const r = await fetchWithAuth("/scans");
  return parseOrThrow<{ scans: Scan[] }>(r);
}
