import React, { useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { theme } from "../theme";

type Props = {
  onContinue: () => Promise<void>;
};

export default function DisclaimerScreen({ onContinue }: Props) {
  const [busy, setBusy] = useState(false);

  async function accept() {
    setBusy(true);
    try {
      await onContinue();
    } finally {
      setBusy(false);
    }
  }

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.wordmark}>CLEAR</Text>
        <Text style={styles.title}>Not a medical device.</Text>
        <Text style={styles.body}>
          CLEAR is an experimental classification tool. It cannot diagnose disease and is not a
          substitute for a dermatologist.
        </Text>
        <Text style={styles.body}>
          Always talk to a medical professional for medical advice or concerns about a skin lesion.
        </Text>

        <Pressable
          style={({ pressed }) => [
            styles.primary,
            pressed && styles.primaryPressed,
            busy && styles.disabled,
          ]}
          onPress={accept}
          disabled={busy}
        >
          <Text style={styles.primaryLabel}>{busy ? "Saving..." : "I understand"}</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.bg,
    justifyContent: "center",
    paddingHorizontal: theme.spacing.lg,
  },
  content: { gap: theme.spacing.md },
  wordmark: {
    fontFamily: theme.fonts.serif,
    fontSize: 48,
    fontWeight: "500",
    color: theme.colors.text,
    textAlign: "center",
  },
  title: {
    fontFamily: theme.fonts.serif,
    fontSize: 30,
    fontWeight: "500",
    color: theme.colors.text,
    textAlign: "center",
    marginTop: theme.spacing.md,
  },
  body: {
    color: theme.colors.muted,
    fontSize: 15,
    lineHeight: 22,
    textAlign: "center",
  },
  primary: {
    backgroundColor: theme.colors.accent,
    borderRadius: theme.radii.pill,
    paddingVertical: 14,
    paddingHorizontal: theme.spacing.lg,
    alignItems: "center",
    marginTop: theme.spacing.lg,
    minHeight: 44,
  },
  primaryPressed: { backgroundColor: theme.colors.accentPressed },
  primaryLabel: { color: "#FFFFFF", fontSize: 15, fontWeight: "500" },
  disabled: { opacity: 0.6 },
});
