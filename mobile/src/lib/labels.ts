export function displayLabel(label: string | null | undefined): string {
  if (label === "suspicious") return "Needs a closer look";
  if (label === "non_suspicious") return "Low concern";
  return "Result unavailable";
}

export function tagLabel(label: string | null | undefined): string {
  if (label === "suspicious") return "Closer look";
  if (label === "non_suspicious") return "Low concern";
  return "Unknown";
}

export function isCloserLook(label: string | null | undefined): boolean {
  return label === "suspicious";
}

export function formatConfidence(confidence: number | null | undefined): string {
  if (typeof confidence !== "number" || Number.isNaN(confidence)) return "--";
  return confidence.toFixed(2);
}
