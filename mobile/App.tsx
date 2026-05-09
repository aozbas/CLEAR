import { Session } from "@supabase/supabase-js";
import React, { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { listScans } from "./src/lib/api";
import { supabase } from "./src/lib/supabase";
import LoginScreen from "./src/screens/LoginScreen";
import { theme } from "./src/theme";

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session);
      setLoading(false);
    });
    const { data: sub } = supabase.auth.onAuthStateChange((_event, s) => setSession(s));
    return () => sub.subscription.unsubscribe();
  }, []);

  if (loading) {
    return (
      <View style={[styles.container, styles.center]}>
        <StatusBar barStyle="dark-content" />
        <ActivityIndicator color={theme.colors.accent} />
      </View>
    );
  }

  return (
    <>
      <StatusBar barStyle="dark-content" />
      {session ? <Home email={session.user.email ?? "(no email)"} /> : <LoginScreen />}
    </>
  );
}

function Home({ email }: { email: string }) {
  const [result, setResult] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function ping() {
    setBusy(true);
    setResult(null);
    try {
      const r = await listScans();
      setResult(JSON.stringify(r, null, 2));
    } catch (e) {
      setResult(String(e));
    }
    setBusy(false);
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.homePadding}>
      <Text style={styles.title}>Signed in</Text>
      <Text style={styles.body}>{email}</Text>

      <Pressable
        style={({ pressed }) => [
          styles.primary,
          pressed && styles.primaryPressed,
          busy && styles.disabled,
        ]}
        onPress={ping}
        disabled={busy}
      >
        <Text style={styles.primaryLabel}>{busy ? "…" : "Test backend (GET /scans)"}</Text>
      </Pressable>

      {result ? <Text style={styles.code}>{result}</Text> : null}

      <Pressable style={styles.secondary} onPress={() => supabase.auth.signOut()}>
        <Text style={styles.secondaryLabel}>Sign out</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.bg },
  center: { alignItems: "center", justifyContent: "center" },
  homePadding: { padding: theme.spacing.lg, gap: theme.spacing.md, paddingTop: theme.spacing.xxl },
  title: {
    fontFamily: theme.fonts.serif,
    fontSize: 22,
    fontWeight: "500",
    color: theme.colors.text,
  },
  body: { fontSize: 15, color: theme.colors.muted },
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
  code: {
    backgroundColor: theme.colors.surface,
    borderColor: theme.colors.line,
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: theme.radii.md,
    padding: theme.spacing.md,
    fontFamily: theme.fonts.mono,
    fontSize: 12,
    color: theme.colors.text,
  },
});
