import { hosted_database } from "./hosted_database";

const BASE = process.env.EXPO_PUBLIC_API_URL ?? "http://localhost:8000";

export type PredictionResponse = {
  label: string;
  confidence: number;
  image_url: string | null;
  signed_image_url?: string | null;
  scan_id: string | null;
  saved: boolean;
  should_retry: boolean;
  message?: string | null;
  model_version?: string | null;
};

export type Scan = {
  id: string;
  image_url: string | null;
  signed_image_url?: string | null;
  label: string;
  confidence: number;
  model_version?: string | null;
  created_at: string;
};

async function authHeader(forceRefresh = false): Promise<Record<string, string>> {
  const { data, error } = forceRefresh
    ? await hosted_database.auth.refreshSession()
    : await hosted_database.auth.getSession();
  if (error) {
    await hosted_database.auth.signOut({ scope: "local" });
    throw new Error("Session expired. Please sign in again.");
  }

  const token = data.session?.access_token;
  if (!token) {
    await hosted_database.auth.signOut({ scope: "local" });
    throw new Error("Please sign in to continue.");
  }

  return { Authorization: `Bearer ${token}` };
}

async function fetchWithAuth(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = {
    ...(init.headers as Record<string, string> | undefined),
    ...(await authHeader()),
  };
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
  const r = await fetchWithAuth("/predictions", {
    method: "POST",
    body: imageForm(imageUri),
  });
  return parseOrThrow<PredictionResponse>(r);
}

export async function predictDemo(imageUri: string): Promise<PredictionResponse> {
  const r = await fetch(`${BASE}/predictions/demo`, {
    method: "POST",
    body: imageForm(imageUri),
  });
  return parseOrThrow<PredictionResponse>(r);
}

function imageForm(imageUri: string): FormData {
  const form = new FormData();
  // @ts-expect-error RN FormData file shape
  form.append("image", { uri: imageUri, name: "scan.jpg", type: "image/jpeg" });
  return form;
}

export async function listScans(): Promise<{ scans: Scan[] }> {
  const r = await fetchWithAuth("/scans");
  return parseOrThrow<{ scans: Scan[] }>(r);
}
