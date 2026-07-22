import AsyncStorage from "@react-native-async-storage/async-storage";
import React, { useEffect, useState } from "react";
import { ActivityIndicator, StatusBar, StyleSheet, View } from "react-native";
import DemoScanScreen from "./src/screens/DemoScanScreen";
import DisclaimerScreen from "./src/screens/DisclaimerScreen";
import EntryScreen from "./src/screens/EntryScreen";
import { theme } from "./src/theme";

type PublicScreen = "entry" | "demo";
const DISCLAIMER_KEY = "clear.disclaimer.accepted";

export default function App() {
  const [disclaimerAccepted, setDisclaimerAccepted] = useState(false);
  const [loading, setLoading] = useState(true);
  const [screen, setScreen] = useState<PublicScreen>("entry");

  useEffect(() => {
    let isMounted = true;
    AsyncStorage.getItem(DISCLAIMER_KEY)
      .then((accepted) => {
        if (isMounted) setDisclaimerAccepted(accepted === "true");
      })
      .catch(() => {
        // If local preferences are unavailable, show the disclaimer again.
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });
    return () => {
      isMounted = false;
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
          await AsyncStorage.setItem(DISCLAIMER_KEY, "true").catch(() => undefined);
          setDisclaimerAccepted(true);
        }}
      />
    );
  }

  return (
    <>
      <StatusBar barStyle="dark-content" />
      {screen === "demo" ? (
        <DemoScanScreen onBack={() => setScreen("entry")} />
      ) : (
        <EntryScreen onDemo={() => setScreen("demo")} />
      )}
    </>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.bg },
  center: { alignItems: "center", justifyContent: "center" },
});
