import { Platform } from "react-native";

export const theme = {
  colors: {
    bg: "#F0EEE6",
    surface: "#FAF9F5",
    text: "#191919",
    muted: "#6B6B6B",
    line: "rgba(25,25,25,0.08)",
    lineStrong: "rgba(25,25,25,0.14)",
    accent: "#D97757",
    accentPressed: "#C56843",
    accentTint: "rgba(217,119,87,0.1)",
    calm: "#3D5A48",
    calmTint: "rgba(74,107,87,0.1)",
    error: "#A03E3E",
  },
  radii: { sm: 8, md: 12, lg: 18, sheet: 24, pill: 999 },
  spacing: { xs: 4, sm: 8, md: 16, lg: 24, xl: 32, xxl: 48, xxxl: 64 },
  // Per DESIGN.md the spec is Source Serif 4 / Geist via expo-font; we use the
  // platform fallbacks documented there until font loading is wired up.
  fonts: {
    sans: undefined as string | undefined,
    serif: Platform.select({ ios: "Georgia", android: "serif", default: "serif" }),
    mono: Platform.select({ ios: "Menlo", android: "monospace", default: "monospace" }),
  },
} as const;
