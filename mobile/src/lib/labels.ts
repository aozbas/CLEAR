const LABELS: Record<string, string> = {
  melanoma: "Melanoma category",
  nevus: "Nevus category",
  basal_cell_carcinoma: "Basal cell carcinoma category",
  actinic_keratosis: "Actinic keratosis category",
  benign_keratosis: "Benign keratosis category",
  dermatofibroma: "Dermatofibroma category",
  vascular_lesion: "Vascular lesion category",
};

export function displayLabel(label: string | null | undefined): string {
  if (!label) return "No category displayed";
  return LABELS[label] ?? "Unknown model category";
}

export function isKnownLabel(label: string | null | undefined): boolean {
  return typeof label === "string" && label in LABELS;
}

export function formatModelScore(score: number | null | undefined): string {
  if (typeof score !== "number" || Number.isNaN(score)) return "--";
  return score.toFixed(2);
}
