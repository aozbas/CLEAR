import AsyncStorage from "@react-native-async-storage/async-storage";
import { Session } from "@hosted_database/hosted_database-js";
import React, { useEffect, useState } from "react";
import { ActivityIndicator, StatusBar, StyleSheet, View } from "react-native";
import { hosted_database } from "./src/lib/hosted_database";
import DemoScanScreen from "./src/screens/DemoScanScreen";
import DisclaimerScreen from "./src/screens/DisclaimerScreen";
import EntryScreen from "./src/screens/EntryScreen";
import HistoryScreen from "./src/screens/HistoryScreen";
import LoginScreen from "./src/screens/LoginScreen";
import ScanScreen from "./src/screens/ScanScreen";
import { theme } from "./src/theme";

type AppScreen = "scan" | "history";
type PublicScreen = "entry" | "demo" | "login";
const DISCLAIMER_KEY = "clear.disclaimer.accepted";

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [disclaimerAccepted, setDisclaimerAccepted] = useState(false);
  const [loading, setLoading] = useState(true);
  const [publicScreen, setPublicScreen] = useState<PublicScreen>("entry");

  useEffect(() => {
    let isMounted = true;
    Promise.all([hosted_database.auth.getSession(), AsyncStorage.getItem(DISCLAIMER_KEY)])
      .then(([{ data }, accepted]) => {
        if (!isMounted) return;
        setSession(data.session);
        setDisclaimerAccepted(accepted === "true");
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    const { data: sub } = hosted_database.auth.onAuthStateChange((_event, s) => {
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

  if (!disclaimerAccepted) {
    return (
      <DisclaimerScreen
        onContinue={async () => {
          await AsyncStorage.setItem(DISCLAIMER_KEY, "true");
          setDisclaimerAccepted(true);
        }}
      />
    );
  }

  if (!session) {
    if (publicScreen === "demo") {
      return (
        <>
          <StatusBar barStyle="dark-content" />
          <DemoScanScreen onBack={() => setPublicScreen("entry")} />
        </>
      );
    }

    return (
      <>
        <StatusBar barStyle="dark-content" />
        {publicScreen === "login" ? (
          <LoginScreen />
        ) : (
          <EntryScreen
            onDemo={() => setPublicScreen("demo")}
            onSignIn={() => setPublicScreen("login")}
          />
        )}
      </>
    );
  }

  return (
    <>
      <StatusBar barStyle="dark-content" />
      <Home email={session.user.email ?? "(no email)"} />
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
        onSignOut={() => hosted_database.auth.signOut()}
      />
    );
  }

  return <ScanScreen onHistory={() => setScreen("history")} />;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.bg },
  center: { alignItems: "center", justifyContent: "center" },
});
