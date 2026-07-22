import React from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { theme } from "../theme";

type Props = {
  onDemo: () => void;
};

export default function EntryScreen({ onDemo }: Props) {
  return (
    <View style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.wordmark}>CLEAR</Text>
        <Text style={styles.title}>Stateless experimental demo</Text>
        <Text style={styles.body}>
          Submit one JPEG or PNG and receive one experimental model category. No account is
          available or required.
        </Text>
        <Text style={styles.body}>
          CLEAR does not create a profile or retain the submitted image or result in application
          storage. Photos already in your library are never deleted.
        </Text>
        <Text style={styles.body}>
          CLEAR is not a medical device, cannot diagnose disease, and must not be used for medical
          decisions or reassurance.
        </Text>

        <Pressable
          style={({ pressed }) => [styles.primary, pressed && styles.primaryPressed]}
          onPress={onDemo}
        >
          <Text style={styles.primaryLabel}>Try experimental demo</Text>
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
});
