import { Session } from "@supabase/supabase-js";
import React, { useEffect, useState } from "react";
import { ActivityIndicator, StatusBar, StyleSheet, View } from "react-native";
import { supabase } from "./src/lib/supabase";
import LoginScreen from "./src/screens/LoginScreen";
import HistoryScreen from "./src/screens/HistoryScreen";
import ScanScreen from "./src/screens/ScanScreen";
import { theme } from "./src/theme";

type AppScreen = "scan" | "history";

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    supabase.auth.getSession().then(({ data }) => {
      if (!isMounted) return;
      setSession(data.session);
      setLoading(false);
    });
    const { data: sub } = supabase.auth.onAuthStateChange((_event, s) => {
      if (!isMounted) return;
      setSession(s);
    });
    return () => {
      isMounted = false;
      sub.subscription.unsubscribe();
    };
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
      {session ? (
        <Home email={session.user.email ?? "(no email)"} />
      ) : (
        <LoginScreen />
      )}
    </>
  );
}

function Home({ email }: { email: string }) {
  const [screen, setScreen] = useState<AppScreen>("scan");

  if (screen === "history") {
    return (
      <HistoryScreen
        email={email}
        onScan={() => setScreen("scan")}
        onSignOut={() => supabase.auth.signOut()}
      />
    );
  }

  return <ScanScreen onHistory={() => setScreen("history")} />;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.bg },
  center: { alignItems: "center", justifyContent: "center" },
});
