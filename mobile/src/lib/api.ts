const BASE = process.env.EXPO_PUBLIC_API_URL ?? "http://localhost:8000";

export async function predict(imageUri: string) {
  const form = new FormData();
  // @ts-expect-error RN FormData file shape
  form.append("image", { uri: imageUri, name: "scan.jpg", type: "image/jpeg" });
  const r = await fetch(`${BASE}/predictions`, { method: "POST", body: form });
  return r.json();
}

export async function listScans() {
  const r = await fetch(`${BASE}/scans`);
  return r.json();
}
