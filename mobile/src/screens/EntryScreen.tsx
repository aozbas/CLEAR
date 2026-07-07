import React from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { theme } from "../theme";

type Props = {
  onDemo: () => void;
  onSignIn: () => void;
};

export default function EntryScreen({ onDemo, onSignIn }: Props) {
  return (
    <View style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.wordmark}>CLEAR</Text>
        <Text style={styles.title}>Private demo mode</Text>
        <Text style={styles.body}>
          Try a one-time experimental classification without creating an account.
        </Text>
        <Text style={styles.body}>
          Demo mode does not save your photo, result, or scan history.
        </Text>

        <Pressable
          style={({ pressed }) => [styles.primary, pressed && styles.primaryPressed]}
          onPress={onDemo}
        >
          <Text style={styles.primaryLabel}>Try demo</Text>
        </Pressable>

        <Pressable style={styles.secondary} onPress={onSignIn}>
          <Text style={styles.secondaryLabel}>Sign in for private history</Text>
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
  secondary: { paddingVertical: 14, alignItems: "center", minHeight: 44 },
  secondaryLabel: { color: theme.colors.muted, fontSize: 15, fontWeight: "500" },
});
