import { createClient } from "@hosted_database/hosted_database-js";

const url = process.env.EXPO_PUBLIC_HOSTED_DATABASE_URL ?? "";
const key = process.env.EXPO_PUBLIC_HOSTED_DATABASE_ANON_KEY ?? "";

export const hosted_database = createClient(url, key);
