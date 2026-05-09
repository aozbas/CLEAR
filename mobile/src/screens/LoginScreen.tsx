import React, { useState } from "react";
import {
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { supabase } from "../lib/supabase";
import { theme } from "../theme";

export default function LoginScreen() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  async function signIn() {
    setBusy(true);
    setError(null);
    setInfo(null);
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) setError(error.message);
    setBusy(false);
  }

  async function signUp() {
    setBusy(true);
    setError(null);
    setInfo(null);
    const { data, error } = await supabase.auth.signUp({ email, password });
    if (error) setError(error.message);
    else if (!data.session) setInfo("Check your email to confirm your account, then sign in.");
    setBusy(false);
  }

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : undefined}
      style={styles.container}
    >
      <View style={styles.card}>
        <Text style={styles.wordmark}>CLEAR</Text>
        <Text style={styles.tagline}>skin lesion identification</Text>

        <TextInput
          style={styles.input}
          placeholder="Email"
          placeholderTextColor={theme.colors.muted}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="email-address"
          textContentType="emailAddress"
          value={email}
          onChangeText={setEmail}
          editable={!busy}
        />
        <TextInput
          style={styles.input}
          placeholder="Password"
          placeholderTextColor={theme.colors.muted}
          autoCapitalize="none"
          autoCorrect={false}
          secureTextEntry
          textContentType="password"
          value={password}
          onChangeText={setPassword}
          editable={!busy}
        />

        {error ? <Text style={styles.error}>{error}</Text> : null}
        {info ? <Text style={styles.info}>{info}</Text> : null}

        <Pressable
          style={({ pressed }) => [
            styles.primary,
            pressed && styles.primaryPressed,
            busy && styles.disabled,
          ]}
          onPress={signIn}
          disabled={busy}
        >
          <Text style={styles.primaryLabel}>{busy ? "…" : "Sign in"}</Text>
        </Pressable>

        <Pressable style={styles.secondary} onPress={signUp} disabled={busy}>
          <Text style={styles.secondaryLabel}>Create account</Text>
        </Pressable>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.bg,
    justifyContent: "center",
    paddingHorizontal: theme.spacing.lg,
  },
  card: { gap: theme.spacing.md },
  wordmark: {
    fontFamily: theme.fonts.serif,
    fontSize: 48,
    fontWeight: "500",
    color: theme.colors.text,
    textAlign: "center",
    letterSpacing: -1,
  },
  tagline: {
    fontFamily: theme.fonts.serif,
    fontStyle: "italic",
    fontSize: 14,
    color: theme.colors.muted,
    textAlign: "center",
    marginBottom: theme.spacing.lg,
  },
  input: {
    backgroundColor: theme.colors.surface,
    borderColor: theme.colors.line,
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: theme.radii.md,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: 14,
    fontSize: 15,
    color: theme.colors.text,
    minHeight: 44,
  },
  primary: {
    backgroundColor: theme.colors.accent,
    borderRadius: theme.radii.pill,
    paddingVertical: 14,
    paddingHorizontal: theme.spacing.lg,
    alignItems: "center",
    marginTop: theme.spacing.sm,
    minHeight: 44,
  },
  primaryPressed: { backgroundColor: theme.colors.accentPressed },
  primaryLabel: { color: "#FFFFFF", fontSize: 15, fontWeight: "500" },
  secondary: { paddingVertical: 14, alignItems: "center", minHeight: 44 },
  secondaryLabel: { color: theme.colors.muted, fontSize: 15, fontWeight: "500" },
  disabled: { opacity: 0.6 },
  error: { color: theme.colors.error, fontSize: 14, textAlign: "center" },
  info: { color: theme.colors.muted, fontSize: 14, textAlign: "center" },
});
