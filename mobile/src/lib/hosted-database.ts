import AsyncStorage from "@react-native-async-storage/async-storage";
import { createClient } from "@hosted_database/hosted_database-js";

const url = process.env.EXPO_PUBLIC_HOSTED_DATABASE_URL;
const key = process.env.EXPO_PUBLIC_HOSTED_DATABASE_ANON_KEY;

if (!url || !key) {
  throw new Error(
    "Missing Hosted database config. Create mobile/.env from mobile/.env.example and fill in EXPO_PUBLIC_HOSTED_DATABASE_URL and EXPO_PUBLIC_HOSTED_DATABASE_ANON_KEY."
  );
}

export const hosted_database = createClient(url, key, {
  auth: {
    storage: AsyncStorage,
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: false,
  },
});
