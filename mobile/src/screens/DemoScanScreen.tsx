import * as ImagePicker from "expo-image-picker";
import React, { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { predictDemo, PredictionResponse, removeTemporaryPickerFile } from "../lib/api";
import { displayLabel, formatModelScore, isKnownLabel } from "../lib/labels";
import { theme } from "../theme";

type Props = {
  onBack: () => void;
};

export default function DemoScanScreen({ onBack }: Props) {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestController = useRef<AbortController | null>(null);
  const temporaryImageUri = useRef<string | null>(null);

  useEffect(
    () => () => {
      requestController.current?.abort();
      void removeTemporaryPickerFile(temporaryImageUri.current);
    },
    [],
  );

  async function submitImage(asset: ImagePicker.ImagePickerAsset) {
    const controller = new AbortController();
    requestController.current?.abort();
    requestController.current = controller;
    temporaryImageUri.current = asset.uri;
    setImageUri(asset.uri);
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const response = await predictDemo(asset, controller.signal);
      if (controller.signal.aborted || requestController.current !== controller) return;
      if (response.outcome === "classification_available" && !isKnownLabel(response.label)) {
        setError("The server returned an unsupported model category. Try again later.");
        return;
      }
      setResult(response);
    } catch (caught) {
      if (!controller.signal.aborted) {
        setError(
          caught instanceof Error
            ? caught.message
            : "The experimental classification could not be completed.",
        );
      }
    } finally {
      await removeTemporaryPickerFile(asset.uri);
      if (requestController.current === controller) {
        requestController.current = null;
        temporaryImageUri.current = null;
        setImageUri(null);
        if (!controller.signal.aborted) setBusy(false);
      }
    }
  }

  async function choosePhoto() {
    try {
      const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!permission.granted) {
        setError("Photo library permission is required to choose an image.");
        return;
      }

      const picked = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ["images"],
        preferredAssetRepresentationMode:
          ImagePicker.UIImagePickerPreferredAssetRepresentationMode.Compatible,
        quality: 0.9,
      });
      if (!picked.canceled && picked.assets[0]) await submitImage(picked.assets[0]);
    } catch {
      setError("The photo library could not be opened. Try again.");
    }
  }

  async function takePhoto() {
    try {
      const permission = await ImagePicker.requestCameraPermissionsAsync();
      if (!permission.granted) {
        setError("Camera permission is required to take a photo.");
        return;
      }

      const picked = await ImagePicker.launchCameraAsync({
        mediaTypes: ["images"],
        quality: 0.9,
      });
      if (!picked.canceled && picked.assets[0]) await submitImage(picked.assets[0]);
    } catch {
      setError("The camera could not be opened. Try again.");
    }
  }

  function resetDemo() {
    requestController.current?.abort();
    requestController.current = null;
    const temporaryUri = temporaryImageUri.current;
    temporaryImageUri.current = null;
    void removeTemporaryPickerFile(temporaryUri);
    setBusy(false);
    setImageUri(null);
    setResult(null);
    setError(null);
  }

  function goBack() {
    resetDemo();
    onBack();
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.topBar}>
        <Text style={styles.wordmark}>CLEAR</Text>
        <Pressable onPress={goBack} style={styles.linkButton}>
          <Text style={styles.linkLabel}>Back</Text>
        </Pressable>
      </View>

      <View style={styles.notice}>
        <Text style={styles.noticeTitle}>One image, one transient result</Text>
        <Text style={styles.noticeText}>
          Image bytes are sent to the configured CLEAR backend for one experimental classification.
          CLEAR does not create an account or retain the image or result. App-owned picker cache is
          cleared after the request; photos in your library are never deleted. The output is not a
          diagnosis.
        </Text>
      </View>

      <View style={styles.photoFrame}>
        {imageUri ? (
          <Image source={{ uri: imageUri }} style={styles.photo} />
        ) : (
          <View style={styles.photoPlaceholder}>
            <Text style={styles.placeholderText}>
              {result ? "Temporary preview cleared" : "No photo selected"}
            </Text>
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
          <Text style={styles.resultLabel}>
            {result.outcome === "classification_available"
              ? "Experimental classification"
              : "No classification shown"}
          </Text>
          <Text style={styles.headline}>
            {result.outcome === "classification_available"
              ? displayLabel(result.label)
              : result.message}
          </Text>
          {result.outcome === "classification_available" && result.model_score !== null ? (
            <View style={styles.scoreRow}>
              <View style={styles.scoreDot} />
              <Text style={styles.scoreText}>
                Uncalibrated model score {formatModelScore(result.model_score)}
              </Text>
            </View>
          ) : null}
          {result.outcome === "classification_available" ? (
            <Text style={styles.resultNote}>{result.message}</Text>
          ) : null}

          <View style={styles.limitationCard}>
            <Text style={styles.limitationTitle}>Important limitation</Text>
            <Text style={styles.limitationText}>
              The current demo model was developed with PAD-UFES and HIBA clinical smartphone-image
              data. It failed the project&apos;s cross-source promotion gates and has no evidence of
              reliable behavior on patient- or consumer-taken photos. Its output may be wrong and
              must not be used for diagnosis, reassurance, or treatment decisions.
            </Text>
          </View>
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
          <Text style={styles.secondaryLabel}>Choose JPEG or PNG</Text>
        </Pressable>

        {imageUri || result || error ? (
          <Pressable style={styles.secondary} onPress={resetDemo}>
            <Text style={styles.secondaryLabel}>Clear on-screen result</Text>
          </Pressable>
        ) : null}
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
  notice: {
    backgroundColor: theme.colors.surface,
    borderColor: theme.colors.line,
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: theme.radii.md,
    padding: theme.spacing.md,
    gap: theme.spacing.xs,
  },
  noticeTitle: { color: theme.colors.text, fontSize: 15, fontWeight: "500" },
  noticeText: { color: theme.colors.muted, fontSize: 14, lineHeight: 20 },
  photoFrame: {
    aspectRatio: 1,
    borderRadius: theme.radii.lg,
    overflow: "hidden",
    backgroundColor: theme.colors.surface,
    borderColor: theme.colors.line,
    borderWidth: StyleSheet.hairlineWidth,
  },
  photo: { width: "100%", height: "100%" },
  photoPlaceholder: { flex: 1, alignItems: "center", justifyContent: "center" },
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
  headline: { color: theme.colors.text, fontSize: 28, fontWeight: "500" },
  scoreRow: { flexDirection: "row", alignItems: "center", gap: theme.spacing.sm },
  scoreDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: theme.colors.accent,
    shadowColor: theme.colors.accent,
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  scoreText: {
    color: theme.colors.muted,
    fontSize: 14,
    fontVariant: ["tabular-nums"],
  },
  resultNote: { color: theme.colors.muted, fontSize: 14 },
  limitationCard: {
    backgroundColor: theme.colors.surface,
    borderColor: theme.colors.line,
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: theme.radii.md,
    padding: theme.spacing.md,
    gap: theme.spacing.xs,
    marginTop: theme.spacing.sm,
  },
  limitationTitle: { color: theme.colors.text, fontSize: 14, fontWeight: "500" },
  limitationText: { color: theme.colors.muted, fontSize: 14, lineHeight: 20 },
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
