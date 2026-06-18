import AsyncStorage from "@react-native-async-storage/async-storage";
import { createClient, processLock } from "@hosted_database/hosted_database-js";
import { AppState, Platform } from "react-native";

const url = process.env.EXPO_PUBLIC_HOSTED_DATABASE_URL;
const key = process.env.EXPO_PUBLIC_HOSTED_DATABASE_PUBLISHABLE_KEY;

if (!url || !key) {
  throw new Error(
    "Missing Hosted database config. Create mobile/.env from mobile/.env.example and fill in EXPO_PUBLIC_HOSTED_DATABASE_URL and EXPO_PUBLIC_HOSTED_DATABASE_PUBLISHABLE_KEY.",
  );
}

export const hosted_database = createClient(url, key, {
  auth: {
    ...(Platform.OS !== "web" ? { storage: AsyncStorage, lock: processLock } : {}),
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: false,
  },
});

if (Platform.OS !== "web") {
  if (AppState.currentState === "active") {
    hosted_database.auth.startAutoRefresh();
  }

  AppState.addEventListener("change", (state) => {
    if (state === "active") {
      hosted_database.auth.startAutoRefresh();
    } else {
      hosted_database.auth.stopAutoRefresh();
    }
  });
}
