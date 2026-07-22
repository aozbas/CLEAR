import * as FileSystem from "expo-file-system/legacy";

const BASE = (process.env.EXPO_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
const REQUEST_TIMEOUT_MS = 30_000;
export const MAX_IMAGE_BYTES = 8 * 1024 * 1024;

const SUPPORTED_IMAGE_TYPES = new Set(["image/jpeg", "image/png"]);

export type LocalImage = {
  uri: string;
  mimeType?: string | null;
  fileName?: string | null;
  fileSize?: number | null;
};

export type PredictionResponse = {
  result_type: "experimental_classification";
  outcome:
    | "classification_available"
    | "classifier_uncertain"
    | "poor_image_quality"
    | "unsupported_image";
  label: string | null;
  model_score: number | null;
  should_retry: boolean;
  message: string;
  model_version: string;
};

class SafeClientError extends Error {}

export async function predictDemo(
  image: LocalImage,
  externalSignal?: AbortSignal,
): Promise<PredictionResponse> {
  ensureSecureApiBase();
  const contentType = resolvedImageType(image);
  if (!SUPPORTED_IMAGE_TYPES.has(contentType)) {
    throw new SafeClientError(
      "Choose a JPEG or PNG image. HEIC and other formats are not supported.",
    );
  }
  await enforceImageSize(image);

  let timedOut = false;
  const task = FileSystem.createUploadTask(`${BASE}/predictions/demo`, image.uri, {
    httpMethod: "POST",
    uploadType: FileSystem.FileSystemUploadType.BINARY_CONTENT,
    headers: {
      Accept: "application/json",
      "Content-Type": contentType,
    },
  });
  const cancelTask = () => void task.cancelAsync().catch(() => undefined);
  externalSignal?.addEventListener("abort", cancelTask, { once: true });
  if (externalSignal?.aborted) cancelTask();

  const timeout = setTimeout(() => {
    timedOut = true;
    cancelTask();
  }, REQUEST_TIMEOUT_MS);
  try {
    const response = await task.uploadAsync();
    if (!response) throw cancellationError(externalSignal, timedOut);
    return parsePrediction(response.status, response.body);
  } catch (error) {
    if (error instanceof SafeClientError) throw error;
    if (externalSignal?.aborted) throw abortError();
    if (timedOut) {
      throw new SafeClientError("The request timed out. Check your connection and try again.");
    }
    throw new SafeClientError(
      "The network request could not be completed. Check your connection and try again.",
    );
  } finally {
    clearTimeout(timeout);
    externalSignal?.removeEventListener("abort", cancelTask);
  }
}

function ensureSecureApiBase(): void {
  if (!__DEV__ && !BASE.startsWith("https://")) {
    throw new SafeClientError("The demo backend is not securely configured.");
  }
}

export async function removeTemporaryPickerFile(uri: string | null): Promise<void> {
  const cacheDirectory = FileSystem.cacheDirectory;
  if (!uri || !cacheDirectory) return;

  const cacheRoot = cacheDirectory.endsWith("/") ? cacheDirectory : `${cacheDirectory}/`;
  if (!uri.startsWith(cacheRoot)) return;
  try {
    const relativePath = decodeURIComponent(uri.slice(cacheRoot.length));
    if (relativePath.split(/[\\/]/).includes("..")) return;
  } catch {
    return;
  }
  await FileSystem.deleteAsync(uri, { idempotent: true }).catch(() => undefined);
}

async function enforceImageSize(image: LocalImage): Promise<void> {
  if (typeof image.fileSize === "number") {
    if (image.fileSize > MAX_IMAGE_BYTES) {
      throw new SafeClientError("Choose an image smaller than 8 MB.");
    }
    return;
  }

  const info = await FileSystem.getInfoAsync(image.uri).catch(() => null);
  if (!info?.exists || info.isDirectory) {
    throw new SafeClientError("The selected image is no longer available. Choose it again.");
  }
  if (typeof info.size === "number" && info.size > MAX_IMAGE_BYTES) {
    throw new SafeClientError("Choose an image smaller than 8 MB.");
  }
}

function resolvedImageType(image: LocalImage): string {
  const declared = image.mimeType?.toLowerCase();
  if (declared === "image/jpg") return "image/jpeg";
  if (declared) return declared;

  const name = (image.fileName ?? image.uri).toLowerCase();
  if (name.endsWith(".jpg") || name.endsWith(".jpeg")) return "image/jpeg";
  if (name.endsWith(".png")) return "image/png";
  return "";
}

function parsePrediction(status: number, body: string): PredictionResponse {
  if (status < 200 || status >= 300) {
    throw new SafeClientError(errorMessageForStatus(status));
  }

  let value: unknown = null;
  try {
    value = JSON.parse(body);
  } catch {
    // The public error below intentionally omits the upstream response body.
  }
  if (!isPredictionResponse(value)) {
    throw new SafeClientError("The server returned an unexpected response. Try again later.");
  }
  return value;
}

function errorMessageForStatus(status: number): string {
  if (status === 400) return "The image upload was interrupted. Choose the image again.";
  if (status === 413) return "Choose an image smaller than 8 MB.";
  if (status === 415) return "Choose a complete JPEG or PNG image.";
  if (status === 429 || status === 503) {
    return "The experimental classifier is temporarily unavailable. Try again shortly.";
  }
  if (status === 504) return "The classification timed out. Try again.";
  return "The experimental classification could not be completed. Try again later.";
}

function cancellationError(signal: AbortSignal | undefined, timedOut: boolean): Error {
  if (signal?.aborted) return abortError();
  if (timedOut) {
    return new SafeClientError("The request timed out. Check your connection and try again.");
  }
  return new SafeClientError("The network request was interrupted. Try again.");
}

function abortError(): Error {
  const error = new Error("Request cancelled");
  error.name = "AbortError";
  return error;
}

function isPredictionResponse(value: unknown): value is PredictionResponse {
  if (!value || typeof value !== "object") return false;
  const result = value as Partial<PredictionResponse>;
  const validOutcome =
    result.outcome === "classification_available" ||
    result.outcome === "classifier_uncertain" ||
    result.outcome === "poor_image_quality" ||
    result.outcome === "unsupported_image";
  const validFields =
    result.outcome === "classification_available"
      ? typeof result.label === "string" &&
        typeof result.model_score === "number" &&
        result.should_retry === false
      : result.label === null && result.model_score === null && result.should_retry === true;
  return (
    result.result_type === "experimental_classification" &&
    validOutcome &&
    validFields &&
    typeof result.should_retry === "boolean" &&
    typeof result.message === "string" &&
    typeof result.model_version === "string"
  );
}
