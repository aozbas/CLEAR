import * as ImagePicker from "expo-image-picker";
import React, { useState } from "react";
import {
  ActivityIndicator,
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { predict, PredictionResponse } from "../lib/api";
import { displayLabel, formatConfidence, isCloserLook } from "../lib/labels";
import { theme } from "../theme";

type Props = {
  onHistory: () => void;
};

export default function ScanScreen({ onHistory }: Props) {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submitImage(uri: string) {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const response = await predict(uri);
      setResult(response);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function choosePhoto() {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setError("Photo library permission is required to choose an image.");
      return;
    }

    const picked = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      quality: 0.9,
    });
    if (picked.canceled || !picked.assets[0]) return;

    const uri = picked.assets[0].uri;
    setImageUri(uri);
    await submitImage(uri);
  }

  async function takePhoto() {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      setError("Camera permission is required to take a photo.");
      return;
    }

    const picked = await ImagePicker.launchCameraAsync({
      mediaTypes: ["images"],
      quality: 0.9,
    });
    if (picked.canceled || !picked.assets[0]) return;

    const uri = picked.assets[0].uri;
    setImageUri(uri);
    await submitImage(uri);
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.topBar}>
        <Text style={styles.wordmark}>CLEAR</Text>
        <Pressable onPress={onHistory} style={styles.linkButton}>
          <Text style={styles.linkLabel}>History</Text>
        </Pressable>
      </View>

      <View style={styles.photoFrame}>
        {imageUri ? (
          <Image source={{ uri: imageUri }} style={styles.photo} />
        ) : (
          <View style={styles.photoPlaceholder}>
            <Text style={styles.placeholderText}>No photo selected</Text>
          </View>
        )}
        {busy ? (
          <View style={styles.loadingOverlay}>
            <ActivityIndicator color={theme.colors.accent} />
          </View>
        ) : null}
      </View>

      {result ? (
        <View style={styles.resultBlock}>
          <Text style={styles.resultLabel}>Result</Text>
          <Text style={styles.headline}>{displayLabel(result.label)}</Text>
          <View style={styles.confidenceRow}>
            <View
              style={[
                styles.confidenceDot,
                !isCloserLook(result.label) && styles.confidenceDotCalm,
              ]}
            />
            <Text style={styles.confidenceText}>
              Confidence {formatConfidence(result.confidence)}
            </Text>
          </View>
          <Text style={styles.savedText}>Saved to history</Text>
        </View>
      ) : null}

      {error ? <Text style={styles.error}>{error}</Text> : null}

      <View style={styles.actions}>
        <Pressable
          style={({ pressed }) => [
            styles.primary,
            pressed && styles.primaryPressed,
            busy && styles.disabled,
          ]}
          onPress={takePhoto}
          disabled={busy}
        >
          <Text style={styles.primaryLabel}>{busy ? "Working..." : "Take photo"}</Text>
        </Pressable>

        <Pressable style={styles.secondary} onPress={choosePhoto} disabled={busy}>
          <Text style={styles.secondaryLabel}>Choose from library</Text>
        </Pressable>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.bg },
  content: {
    paddingHorizontal: theme.spacing.lg,
    paddingTop: theme.spacing.lg,
    paddingBottom: 40,
    gap: theme.spacing.md,
  },
  topBar: {
    minHeight: 44,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  wordmark: {
    fontFamily: theme.fonts.serif,
    fontSize: 22,
    fontWeight: "500",
    color: theme.colors.text,
  },
  linkButton: {
    minHeight: 44,
    paddingHorizontal: theme.spacing.sm,
    alignItems: "center",
    justifyContent: "center",
  },
  linkLabel: { color: theme.colors.muted, fontSize: 15, fontWeight: "500" },
  photoFrame: {
    aspectRatio: 1,
    borderRadius: theme.radii.lg,
    overflow: "hidden",
    backgroundColor: theme.colors.surface,
    borderColor: theme.colors.line,
    borderWidth: StyleSheet.hairlineWidth,
  },
  photo: { width: "100%", height: "100%" },
  photoPlaceholder: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  placeholderText: { color: theme.colors.muted, fontSize: 14 },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(250,249,245,0.72)",
  },
  resultBlock: { gap: theme.spacing.sm },
  resultLabel: {
    fontFamily: theme.fonts.serif,
    fontStyle: "italic",
    color: theme.colors.muted,
    fontSize: 15,
  },
  headline: {
    color: theme.colors.text,
    fontSize: 28,
    fontWeight: "500",
  },
  confidenceRow: { flexDirection: "row", alignItems: "center", gap: theme.spacing.sm },
  confidenceDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: theme.colors.accent,
    shadowColor: theme.colors.accent,
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  confidenceDotCalm: {
    backgroundColor: theme.colors.calm,
    shadowColor: theme.colors.calm,
  },
  confidenceText: {
    color: theme.colors.muted,
    fontSize: 14,
    fontVariant: ["tabular-nums"],
  },
  savedText: { color: theme.colors.muted, fontSize: 14 },
  error: { color: theme.colors.error, fontSize: 14 },
  actions: { gap: theme.spacing.sm, marginTop: theme.spacing.sm },
  primary: {
    backgroundColor: theme.colors.accent,
    borderRadius: theme.radii.pill,
    paddingVertical: 14,
    paddingHorizontal: theme.spacing.lg,
    alignItems: "center",
    minHeight: 44,
  },
  primaryPressed: { backgroundColor: theme.colors.accentPressed },
  primaryLabel: { color: "#FFFFFF", fontSize: 15, fontWeight: "500" },
  secondary: { paddingVertical: 14, alignItems: "center", minHeight: 44 },
  secondaryLabel: { color: theme.colors.muted, fontSize: 15, fontWeight: "500" },
  disabled: { opacity: 0.6 },
});
