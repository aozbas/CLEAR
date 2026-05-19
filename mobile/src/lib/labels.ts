type LabelInfo = {
  display: string;
  tag: string;
  closerLook: boolean;
};

const LABELS: Record<string, LabelInfo> = {
  suspicious: {
    display: "Needs a closer look",
    tag: "Closer look",
    closerLook: true,
  },
  non_suspicious: {
    display: "Low concern",
    tag: "Low concern",
    closerLook: false,
  },
  melanoma: {
    display: "Melanoma",
    tag: "Melanoma",
    closerLook: true,
  },
  nevus: {
    display: "Common mole",
    tag: "Nevus",
    closerLook: false,
  },
  basal_cell_carcinoma: {
    display: "Basal cell carcinoma",
    tag: "BCC",
    closerLook: true,
  },
  actinic_keratosis: {
    display: "Actinic keratosis",
    tag: "AK",
    closerLook: true,
  },
  benign_keratosis: {
    display: "Benign keratosis",
    tag: "BKL",
    closerLook: false,
  },
  dermatofibroma: {
    display: "Dermatofibroma",
    tag: "DF",
    closerLook: false,
  },
  vascular_lesion: {
    display: "Vascular lesion",
    tag: "Vascular",
    closerLook: false,
  },
};

export function displayLabel(label: string | null | undefined): string {
  if (!label) return "Result unavailable";
  return LABELS[label]?.display ?? "Result unavailable";
}

export function tagLabel(label: string | null | undefined): string {
  if (!label) return "Unknown";
  return LABELS[label]?.tag ?? "Unknown";
}

export function isCloserLook(label: string | null | undefined): boolean {
  if (!label) return false;
  return LABELS[label]?.closerLook ?? false;
}

export function formatConfidence(confidence: number | null | undefined): string {
  if (typeof confidence !== "number" || Number.isNaN(confidence)) return "--";
  return confidence.toFixed(2);
}
