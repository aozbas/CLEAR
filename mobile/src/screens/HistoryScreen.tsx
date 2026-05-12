import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Image,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { listScans, Scan } from "../lib/api";
import { formatConfidence, isCloserLook, tagLabel } from "../lib/labels";
import { theme } from "../theme";

type Props = {
  email: string;
  onScan: () => void;
  onSignOut: () => void;
};

type Filter = "all" | "closer" | "low";

export default function HistoryScreen({ email, onScan, onSignOut }: Props) {
  const [scans, setScans] = useState<Scan[]>([]);
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (mode: "initial" | "refresh" = "initial") => {
    if (mode === "initial") setLoading(true);
    if (mode === "refresh") setRefreshing(true);
    setError(null);
    try {
      const response = await listScans();
      setScans(response.scans);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const filteredScans = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return scans.filter((scan) => {
      if (filter === "closer" && !isCloserLook(scan.label)) return false;
      if (filter === "low" && isCloserLook(scan.label)) return false;
      if (!normalizedQuery) return true;
      return tagLabel(scan.label).toLowerCase().includes(normalizedQuery);
    });
  }, [filter, query, scans]);

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={() => load("refresh")} />
      }
    >
      <View style={styles.topBar}>
        <Text style={styles.wordmark}>CLEAR</Text>
        <Pressable onPress={onScan} style={styles.linkButton}>
          <Text style={styles.linkLabel}>+ Scan</Text>
        </Pressable>
      </View>

      <View style={styles.accountRow}>
        <Text style={styles.accountText}>{email}</Text>
        <Pressable onPress={onSignOut} style={styles.signOutButton}>
          <Text style={styles.signOutLabel}>Sign out</Text>
        </Pressable>
      </View>

      <Text style={styles.sectionCap}>History</Text>
      <TextInput
        style={styles.search}
        placeholder="Search scans..."
        placeholderTextColor={theme.colors.muted}
        value={query}
        onChangeText={setQuery}
        autoCapitalize="none"
        autoCorrect={false}
      />

      <View style={styles.filters}>
        <FilterChip label="All" active={filter === "all"} onPress={() => setFilter("all")} />
        <FilterChip
          label="Closer look"
          active={filter === "closer"}
          onPress={() => setFilter("closer")}
        />
        <FilterChip label="Low concern" active={filter === "low"} onPress={() => setFilter("low")} />
      </View>

      {loading ? (
        <View style={styles.centerBlock}>
          <ActivityIndicator color={theme.colors.accent} />
        </View>
      ) : null}

      {error ? <Text style={styles.error}>{error}</Text> : null}

      {!loading && filteredScans.length === 0 ? (
        <View style={styles.emptyBlock}>
          <Text style={styles.emptyTitle}>No scans yet</Text>
          <Text style={styles.emptyText}>Your saved scan results will appear here.</Text>
        </View>
      ) : null}

      <View style={styles.list}>
        {filteredScans.map((scan) => (
          <HistoryRow key={scan.id} scan={scan} />
        ))}
      </View>
    </ScrollView>
  );
}

function FilterChip({
  label,
  active,
  onPress,
}: {
  label: string;
  active: boolean;
  onPress: () => void;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={[styles.filterChip, active && styles.filterChipActive]}
    >
      <Text style={[styles.filterChipLabel, active && styles.filterChipLabelActive]}>
        {label}
      </Text>
    </Pressable>
  );
}

function HistoryRow({ scan }: { scan: Scan }) {
  const closer = isCloserLook(scan.label);

  return (
    <View style={styles.row}>
      {scan.signed_image_url ? (
        <Image source={{ uri: scan.signed_image_url }} style={styles.thumbnail} />
      ) : (
        <View style={styles.thumbnailPlaceholder} />
      )}

      <View style={styles.rowMain}>
        <View
          style={[
            styles.tag,
            closer ? styles.tagCloser : styles.tagLow,
          ]}
        >
          <Text style={[styles.tagText, closer ? styles.tagTextCloser : styles.tagTextLow]}>
            {tagLabel(scan.label)}
          </Text>
        </View>
        <Text style={styles.rowMeta}>{formatConfidence(scan.confidence)} confidence</Text>
      </View>

      <Text style={styles.date}>{formatDate(scan.created_at)}</Text>
    </View>
  );
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
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
  accountRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: theme.spacing.md,
  },
  accountText: { color: theme.colors.muted, fontSize: 14, flex: 1 },
  signOutButton: { minHeight: 44, justifyContent: "center" },
  signOutLabel: { color: theme.colors.muted, fontSize: 14, fontWeight: "500" },
  sectionCap: {
    color: theme.colors.muted,
    fontSize: 12,
    fontWeight: "500",
    letterSpacing: 0.8,
    textTransform: "uppercase",
  },
  search: {
    backgroundColor: theme.colors.surface,
    borderColor: theme.colors.line,
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: theme.radii.pill,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: 12,
    fontSize: 15,
    color: theme.colors.text,
    minHeight: 44,
  },
  filters: { flexDirection: "row", gap: theme.spacing.sm, flexWrap: "wrap" },
  filterChip: {
    borderRadius: theme.radii.pill,
    borderColor: theme.colors.line,
    borderWidth: StyleSheet.hairlineWidth,
    backgroundColor: theme.colors.surface,
    paddingHorizontal: theme.spacing.md,
    minHeight: 36,
    justifyContent: "center",
  },
  filterChipActive: {
    backgroundColor: theme.colors.accent,
    borderColor: theme.colors.accent,
  },
  filterChipLabel: { color: theme.colors.muted, fontSize: 12, fontWeight: "500" },
  filterChipLabelActive: { color: "#FFFFFF" },
  centerBlock: { paddingVertical: theme.spacing.xl, alignItems: "center" },
  error: { color: theme.colors.error, fontSize: 14 },
  emptyBlock: {
    backgroundColor: theme.colors.surface,
    borderColor: theme.colors.line,
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: theme.radii.md,
    padding: theme.spacing.md,
    gap: theme.spacing.xs,
  },
  emptyTitle: { color: theme.colors.text, fontSize: 15, fontWeight: "500" },
  emptyText: { color: theme.colors.muted, fontSize: 14 },
  list: {
    backgroundColor: theme.colors.surface,
    borderColor: theme.colors.line,
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: theme.radii.md,
    overflow: "hidden",
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 14,
    paddingHorizontal: theme.spacing.md,
    gap: theme.spacing.md,
    borderBottomColor: theme.colors.line,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  thumbnail: {
    width: 40,
    height: 40,
    borderRadius: theme.radii.sm,
    backgroundColor: theme.colors.bg,
  },
  thumbnailPlaceholder: {
    width: 40,
    height: 40,
    borderRadius: theme.radii.sm,
    backgroundColor: theme.colors.bg,
  },
  rowMain: { flex: 1, gap: 6 },
  tag: {
    alignSelf: "flex-start",
    borderRadius: theme.radii.pill,
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  tagCloser: { backgroundColor: theme.colors.accentTint },
  tagLow: { backgroundColor: theme.colors.calmTint },
  tagText: { fontSize: 12, fontWeight: "500" },
  tagTextCloser: { color: theme.colors.accentPressed },
  tagTextLow: { color: theme.colors.calm },
  rowMeta: {
    color: theme.colors.muted,
    fontSize: 14,
    fontVariant: ["tabular-nums"],
  },
  date: {
    color: theme.colors.muted,
    fontSize: 14,
    fontVariant: ["tabular-nums"],
  },
});
