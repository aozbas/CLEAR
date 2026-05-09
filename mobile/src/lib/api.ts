import { supabase } from "./supabase";

const BASE = process.env.EXPO_PUBLIC_API_URL ?? "http://localhost:8000";

async function authHeader(): Promise<Record<string, string>> {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function parseOrThrow(r: Response) {
  if (!r.ok) {
    const body = await r.text().catch(() => "");
    throw new Error(`HTTP ${r.status} ${r.statusText}${body ? `: ${body}` : ""}`);
  }
  return r.json();
}

export async function predict(imageUri: string) {
  const form = new FormData();
  // @ts-expect-error RN FormData file shape
  form.append("image", { uri: imageUri, name: "scan.jpg", type: "image/jpeg" });
  const r = await fetch(`${BASE}/predictions`, {
    method: "POST",
    body: form,
    headers: await authHeader(),
  });
  return parseOrThrow(r);
}

export async function listScans() {
  const r = await fetch(`${BASE}/scans`, { headers: await authHeader() });
  return parseOrThrow(r);
}
