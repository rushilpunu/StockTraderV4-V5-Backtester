import { StatusBar as ExpoStatusBar } from "expo-status-bar";
import { LinearGradient } from "expo-linear-gradient";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Modal,
  Platform,
  Pressable,
  RefreshControl,
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import {
  VictoryArea,
  VictoryAxis,
  VictoryBar,
  VictoryChart,
  VictoryGroup,
  VictoryLegend,
  VictoryLine,
} from "victory-native";
import { Calendar } from "react-native-calendars";

const API_BASE = process.env.EXPO_PUBLIC_API_BASE ?? "http://localhost:8001";

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as T;
}

const currency = (value?: number | null) =>
  value === undefined || value === null
    ? "—"
    : new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 0,
      }).format(value);

const normalizeDateString = (value: string) =>
  /Z|[+-]\d\d:?\d\d$/.test(value) ? value : `${value}Z`;

const formatDateTime = (value?: string | null, options?: Intl.DateTimeFormatOptions) => {
  if (!value) {
    return "—";
  }
  const date = new Date(normalizeDateString(value));
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString([], options ?? { hour: "2-digit", minute: "2-digit" });
};

const formatDate = (value?: string | null) => formatDateTime(value, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

const parseLocalDateTime = (value: string, fallback: Date = new Date()) => {
  if (!value) {
    return fallback;
  }
  const normalized = value.includes("T") ? value : `${value}T00:00`;
  const parsed = new Date(normalized);
  return Number.isNaN(parsed.getTime()) ? fallback : parsed;
};

const pad2 = (num: number) => String(num).padStart(2, "0");

const formatLocalInput = (date: Date) =>
  `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}T${pad2(date.getHours())}:${pad2(date.getMinutes())}`;

const formatReadableLocal = (date: Date) =>
  date.toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

const formatIsoDateOnly = (date: Date) =>
  `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}`;

const clampTimePart = (value: string, max: number, fallback: number) => {
  const normalized = value.replace(/[^0-9]/g, "");
  if (!normalized) {
    return fallback;
  }
  const parsed = Number.parseInt(normalized, 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(max, Math.max(0, parsed));
};

const combineDateAndTime = (isoDate: string, hour: number, minute: number) => {
  const base = new Date(`${isoDate}T00:00:00`);
  if (Number.isNaN(base.getTime())) {
    return new Date();
  }
  base.setHours(hour, minute, 0, 0);
  return base;
};

type MarketStatus = {
  isOpen: boolean | null;
  nextOpen: string | null;
  nextClose: string | null;
  timestamp: string | null;
};

type CycleDiagnostics = {
  startedAt: string;
  completedAt: string;
  tickersProcessed: number;
  alertsGenerated: number;
  decisionsMade: number;
  holds: number;
  entries: number;
  exits: number;
  errors: string[];
};

type DecisionRecord = {
  ticker: string;
  action: string;
  notional: number;
  confidence: number;
  reason: string;
  intent: string;
  createdAt: string;
};

type SnapshotResponse = {
  alerts: AlertItem[];
  decisions: DecisionRecord[];
  positions: PositionItem[];
  dayTrade: DayTradeSummary | null;
  lastUpdated: string | null;
  marketStatus: MarketStatus | null;
  diagnostics: CycleDiagnostics | null;
};

type SentimentResponse = {
  history: SentimentPoint[];
  lastUpdated: string | null;
};

type AccountResponse = {
  portfolioValue: number;
  buyingPower: number;
  cash: number;
  equity: number;
  status: string;
};

type AlertItem = {
  ticker: string;
  averageSentiment: number;
  sentimentDelta: number;
  reason: string;
  articleCount: number;
  seenAt: string;
};

type PositionItem = {
  ticker: string;
  side: string;
  quantity: number;
  marketValue: number;
  costBasis?: number | null;
  lastUpdated: string;
};

type DayTradeSummary = {
  used: number;
  limit: number;
  nextReset: string | null;
};

type SentimentPoint = {
  averageSentiment: number;
  sentimentDelta: number;
  articleCount: number;
  ticker: string;
  seenAt: string;
};

type Settings = {
  tickers: string[];
  gdeltMinutesBack: number;
  sentimentWindowMinutes: number;
  cyclePauseSeconds: number;
  thresholds: {
    minArticles: number;
    sentimentSpike: number;
    sentimentExtreme: number;
    averageSentimentFloor: number;
    singleArticleSpike: number;
  };
  risk: {
    maxCapitalFraction: number;
    stopLossPct: number;
    takeProfitPct: number;
    cooldownMinutes: number;
    entrySentimentThreshold: number;
    exitSentimentThreshold: number;
    maxPositionValue: number | null;
    allowShorting: boolean;
  };
};

const emptySettings: Settings = {
  tickers: [],
  gdeltMinutesBack: 180,
  sentimentWindowMinutes: 60,
  cyclePauseSeconds: 300,
  thresholds: {
    minArticles: 1,
    sentimentSpike: 0.2,
    sentimentExtreme: 0.45,
    averageSentimentFloor: 0.25,
    singleArticleSpike: 0.35,
  },
  risk: {
    maxCapitalFraction: 0.2,
    stopLossPct: 0.04,
    takeProfitPct: 0.08,
    cooldownMinutes: 30,
    entrySentimentThreshold: 0.1,
    exitSentimentThreshold: 0.05,
    maxPositionValue: null,
    allowShorting: false,
  },
};

type Screen = "dashboard" | "backtester";
type Engine = "legacy" | "v5";

type BacktestResult = {
  config: {
    tickers: string[];
    start: string;
    end: string;
    startingCash: number;
    timeframe: string;
    bots: string[];
  };
  results: Record<
    string,
    {
      totalReturn: number;
      alertPrecision: number;
      winRate: number;
      maxDrawdown: number;
      tradeCount: number;
      pnlCurve: number[];
      trades: {
        ticker: string;
        action: string;
        price: number;
        quantity: number;
        notional: number;
        timestamp: string;
        reason: string;
      }[];
      featureSummary: {
        averageScore: number;
        tradeCount: number;
        topKeywords: string[];
        snapshots: number;
      };
      debug: {
        steps: {
          timestamp: string;
          price: number;
          tone15: number;
          delta15: number;
          score?: string;
          action: string;
          notional: number;
          cash: number;
          equity: number;
        }[];
        sentimentSnapshots: number;
        pricePoints: number;
      };
    }
  >;
};

type V5BacktestResult = {
  summary: {
    start: string;
    end: string;
    initialEquity: number;
    finalEquity: number;
    totalReturnPct: number;
    maxDrawdownPct: number;
    sharpeRatio: number | null;
    totalTrades: number;
    winRate: number;
    avgWin: number;
    avgLoss: number;
    profitFactor: number | null;
    violations: number;
  };
  equityCurve: { date: string; equity: number }[];
  trades: {
    symbol: string;
    side: string;
    quantity: number;
    entryPrice: number;
    exitPrice: number | null;
    entryDate: string;
    exitDate: string | null;
    pnl: number;
    pnlPct: number;
    holdingDays: number;
    reason: string;
    bucket: string | null;
  }[];
  violations: { type: string; date: string; symbol: string; description: string; severity: string }[];
};

function Sidebar({ screen, setScreen }: { screen: Screen; setScreen: (s: Screen) => void }) {
  return (
    <View style={styles.sidebar}>
      <Text style={styles.sidebarTitle}>TraderV4</Text>
      <Pressable
        style={[styles.sidebarButton, screen === "dashboard" && styles.sidebarButtonActive]}
        onPress={() => setScreen("dashboard")}
      >
        <Text style={styles.sidebarButtonText}>Live Monitor</Text>
      </Pressable>
      <Pressable
        style={[styles.sidebarButton, screen === "backtester" && styles.sidebarButtonActive]}
        onPress={() => setScreen("backtester")}
      >
        <Text style={styles.sidebarButtonText}>Backtesting Lab</Text>
      </Pressable>
    </View>
  );
}

function SentimentChart({ history }: { history: SentimentPoint[] }) {
  if (!history.length) {
    return <Text style={styles.bodyText}>Waiting for sentiment data…</Text>;
  }
  const series = history.slice(-60);
  return (
    <VictoryChart width={360} height={240} padding={{ top: 40, bottom: 60, left: 60, right: 20 }}>
      <VictoryLegend
        x={80}
        y={10}
        orientation="horizontal"
        gutter={12}
        style={{ labels: { fill: "#cbd5f5", fontSize: 12 } }}
        data={[
          { name: "Avg Sentiment", symbol: { fill: "#38bdf8" } },
          { name: "Sentiment Δ", symbol: { fill: "#f472b6" } },
        ]}
      />
      <VictoryAxis
        tickFormat={(t) =>
          new Date(t).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })
        }
        style={{ tickLabels: { fill: "#94a3b8", fontSize: 11, angle: -35 } }}
      />
      <VictoryAxis
        dependentAxis
        tickFormat={(t) => t.toFixed(2)}
        style={{ tickLabels: { fill: "#94a3b8", fontSize: 11 } }}
      />
      <VictoryGroup>
        <VictoryArea
          data={series.map((point) => ({ x: point.seenAt, y: point.averageSentiment }))}
          style={{ data: { fill: "rgba(56,189,248,0.25)", stroke: "#38bdf8", strokeWidth: 2 } }}
        />
        <VictoryLine
          data={series.map((point) => ({ x: point.seenAt, y: point.sentimentDelta }))}
          style={{ data: { stroke: "#f472b6", strokeWidth: 2, strokeDasharray: "6,4" } }}
        />
      </VictoryGroup>
    </VictoryChart>
  );
}

function ExposureChart({ positions }: { positions: PositionItem[] }) {
  if (!positions.length) {
    return <Text style={styles.bodyText}>No open exposure.</Text>;
  }
  return (
    <VictoryChart width={360} height={220} padding={{ top: 40, bottom: 70, left: 70, right: 20 }}>
      <VictoryAxis
        tickFormat={(t) => t}
        style={{ tickLabels: { fill: "#94a3b8", fontSize: 11, angle: -35 } }}
      />
      <VictoryAxis
        dependentAxis
        tickFormat={(t) => `$${(t / 1000).toFixed(1)}k`}
        style={{ tickLabels: { fill: "#94a3b8", fontSize: 11 } }}
      />
      <VictoryBar
        data={positions.map((position) => ({ x: position.ticker, y: position.marketValue }))}
        style={{ data: { fill: "#4f46e5", width: 16 } }}
        cornerRadius={4}
      />
    </VictoryChart>
  );
}

function PDTGauge({ summary }: { summary: DayTradeSummary | null }) {
  if (!summary) {
    return <Text style={styles.bodyText}>Awaiting data…</Text>;
  }
  const usage = summary.limit ? summary.used / summary.limit : 0;
  return (
    <View style={styles.pdtContainer}>
      <LinearGradient colors={["#38bdf8", "#4f46e5"]} style={styles.pdtBarBackground}>
        <View style={[styles.pdtBarFill, { flex: usage }]} />
        <View style={{ flex: Math.max(0, 1 - usage) }} />
      </LinearGradient>
      <Text style={styles.pdtText}>
        {summary.used} of {summary.limit} intraday trades used
      </Text>
      <Text style={styles.metaText}>
        Next reset: {summary.nextReset ? new Date(summary.nextReset).toLocaleDateString() : "pending"}
      </Text>
    </View>
  );
}

function SettingsModal({
  visible,
  onClose,
  settings,
  onSubmit,
}: {
  visible: boolean;
  onClose: () => void;
  settings: Settings;
  onSubmit: (nextSettings: Settings) => Promise<void>;
}) {
  const [draft, setDraft] = useState<Settings>(settings);
  useEffect(() => setDraft(settings), [settings]);

  const update = (mutator: (settings: Settings) => void) => {
    setDraft((prev) => {
      const next: Settings = JSON.parse(JSON.stringify(prev));
      mutator(next);
      return next;
    });
  };

  const handleSubmit = async () => {
    try {
      await onSubmit(draft);
      onClose();
    } catch (err) {
      Alert.alert("Update failed", String(err));
    }
  };

  return (
    <Modal visible={visible} animationType="slide" presentationStyle="pageSheet">
      <SafeAreaView style={styles.modalContainer}>
        <Text style={styles.modalTitle}>Strategy Settings</Text>
        <ScrollView contentContainerStyle={styles.modalForm}>
          <Text style={styles.modalLabel}>Watchlist Tickers (comma separated)</Text>
          <TextInput
            style={styles.modalInput}
            value={draft.tickers.join(", ")}
            onChangeText={(value) => update((next) => (next.tickers = value.split(/\s*,\s*/).filter(Boolean)))}
          />
          <Text style={styles.modalLabel}>GDELT Minutes Back</Text>
          <TextInput
            style={styles.modalInput}
            keyboardType="numeric"
            value={String(draft.gdeltMinutesBack)}
            onChangeText={(value) =>
              update((next) => {
                const parsed = Number(value);
                if (!Number.isNaN(parsed) && parsed > 0) {
                  next.gdeltMinutesBack = parsed;
                }
              })
            }
          />
          <Text style={styles.modalLabel}>Sentiment Window (minutes)</Text>
          <TextInput
            style={styles.modalInput}
            keyboardType="numeric"
            value={String(draft.sentimentWindowMinutes)}
            onChangeText={(value) =>
              update((next) => {
                const parsed = Number(value);
                if (!Number.isNaN(parsed) && parsed > 0) {
                  next.sentimentWindowMinutes = parsed;
                }
              })
            }
          />
          <Text style={styles.modalLabel}>Cycle Pause (seconds)</Text>
          <TextInput
            style={styles.modalInput}
            keyboardType="numeric"
            value={String(draft.cyclePauseSeconds)}
            onChangeText={(value) =>
              update((next) => {
                const parsed = Number(value);
                if (!Number.isNaN(parsed) && parsed > 0) {
                  next.cyclePauseSeconds = parsed;
                }
              })
            }
          />
          <View style={styles.modalDivider} />
          <Text style={styles.modalLabel}>Threshold - Min Articles</Text>
          <TextInput
            style={styles.modalInput}
            keyboardType="numeric"
            value={String(draft.thresholds.minArticles)}
            onChangeText={(value) =>
              update((next) => {
                const parsed = Number(value);
                if (!Number.isNaN(parsed) && parsed >= 0) {
                  next.thresholds.minArticles = parsed;
                }
              })
            }
          />
          <Text style={styles.modalLabel}>Threshold - Sentiment Spike</Text>
          <TextInput
            style={styles.modalInput}
            keyboardType="decimal-pad"
            value={String(draft.thresholds.sentimentSpike)}
            onChangeText={(value) =>
              update((next) => {
                const parsed = Number(value);
                if (!Number.isNaN(parsed)) {
                  next.thresholds.sentimentSpike = parsed;
                }
              })
            }
          />
          <Text style={styles.modalLabel}>Threshold - Sentiment Extreme</Text>
          <TextInput
            style={styles.modalInput}
            keyboardType="decimal-pad"
            value={String(draft.thresholds.sentimentExtreme)}
            onChangeText={(value) =>
              update((next) => {
                const parsed = Number(value);
                if (!Number.isNaN(parsed)) {
                  next.thresholds.sentimentExtreme = parsed;
                }
              })
            }
          />
          <Text style={styles.modalLabel}>Threshold - Avg Sentiment Floor</Text>
          <TextInput
            style={styles.modalInput}
            keyboardType="decimal-pad"
            value={String(draft.thresholds.averageSentimentFloor)}
            onChangeText={(value) =>
              update((next) => {
                const parsed = Number(value);
                if (!Number.isNaN(parsed)) {
                  next.thresholds.averageSentimentFloor = parsed;
                }
              })
            }
          />
          <Text style={styles.modalLabel}>Threshold - Single Article Spike</Text>
          <TextInput
            style={styles.modalInput}
            keyboardType="decimal-pad"
            value={String(draft.thresholds.singleArticleSpike)}
            onChangeText={(value) =>
              update((next) => {
                const parsed = Number(value);
                if (!Number.isNaN(parsed)) {
                  next.thresholds.singleArticleSpike = parsed;
                }
              })
            }
          />
          <View style={styles.modalDivider} />
          <Text style={styles.modalLabel}>Risk - Max Capital Fraction</Text>
          <TextInput
            style={styles.modalInput}
            keyboardType="decimal-pad"
            value={String(draft.risk.maxCapitalFraction)}
            onChangeText={(value) =>
              update((next) => {
                const parsed = Number(value);
                if (!Number.isNaN(parsed) && parsed > 0) {
                  next.risk.maxCapitalFraction = parsed;
                }
              })
            }
          />
          <Text style={styles.modalLabel}>Risk - Stop Loss %</Text>
          <TextInput
            style={styles.modalInput}
            keyboardType="decimal-pad"
            value={String(draft.risk.stopLossPct)}
            onChangeText={(value) =>
              update((next) => {
                const parsed = Number(value);
                if (!Number.isNaN(parsed) && parsed >= 0) {
                  next.risk.stopLossPct = parsed;
                }
              })
            }
          />
          <Text style={styles.modalLabel}>Risk - Take Profit %</Text>
          <TextInput
            style={styles.modalInput}
            keyboardType="decimal-pad"
            value={String(draft.risk.takeProfitPct)}
            onChangeText={(value) =>
              update((next) => {
                const parsed = Number(value);
                if (!Number.isNaN(parsed) && parsed >= 0) {
                  next.risk.takeProfitPct = parsed;
                }
              })
            }
          />
          <Text style={styles.modalLabel}>Risk - Cooldown Minutes</Text>
          <TextInput
            style={styles.modalInput}
            keyboardType="numeric"
            value={String(draft.risk.cooldownMinutes)}
            onChangeText={(value) =>
              update((next) => {
                const parsed = Number(value);
                if (!Number.isNaN(parsed) && parsed >= 0) {
                  next.risk.cooldownMinutes = parsed;
                }
              })
            }
          />
          <Text style={styles.modalLabel}>Risk - Entry Sentiment Threshold</Text>
          <TextInput
            style={styles.modalInput}
            keyboardType="decimal-pad"
            value={String(draft.risk.entrySentimentThreshold)}
            onChangeText={(value) =>
              update((next) => {
                const parsed = Number(value);
                if (!Number.isNaN(parsed)) {
                  next.risk.entrySentimentThreshold = parsed;
                }
              })
            }
          />
          <Text style={styles.modalLabel}>Risk - Exit Sentiment Threshold</Text>
          <TextInput
            style={styles.modalInput}
            keyboardType="decimal-pad"
            value={String(draft.risk.exitSentimentThreshold)}
            onChangeText={(value) =>
              update((next) => {
                const parsed = Number(value);
                if (!Number.isNaN(parsed)) {
                  next.risk.exitSentimentThreshold = parsed;
                }
              })
            }
          />
          <Text style={styles.modalLabel}>Risk - Max Position Value (blank for unlimited)</Text>
          <TextInput
            style={styles.modalInput}
            keyboardType="decimal-pad"
            placeholder="$"
            value={draft.risk.maxPositionValue === null ? "" : String(draft.risk.maxPositionValue)}
            onChangeText={(value) =>
              update((next) => {
                const trimmed = value.trim();
                if (trimmed.length === 0) {
                  next.risk.maxPositionValue = null;
                  return;
                }
                const parsed = Number(trimmed);
                if (!Number.isNaN(parsed) && parsed >= 0) {
                  next.risk.maxPositionValue = parsed;
                }
              })
            }
          />
          <View style={styles.modalToggleRow}>
            <Text style={styles.modalLabel}>Risk - Allow Shorting</Text>
            <Pressable
              style={[styles.modalToggleButton, draft.risk.allowShorting && styles.modalToggleButtonActive]}
              onPress={() => update((next) => (next.risk.allowShorting = !next.risk.allowShorting))}
            >
              <Text style={styles.modalToggleText}>{draft.risk.allowShorting ? "ENABLED" : "DISABLED"}</Text>
            </Pressable>
          </View>
        </ScrollView>
        <View style={styles.modalButtons}>
          <Pressable style={[styles.button, styles.buttonSecondary]} onPress={onClose}>
            <Text style={styles.buttonText}>Cancel</Text>
          </Pressable>
          <Pressable style={[styles.button, styles.buttonPrimary]} onPress={handleSubmit}>
            <Text style={styles.buttonText}>Save</Text>
          </Pressable>
        </View>
      </SafeAreaView>
    </Modal>
  );
}

function DashboardScreen({
  snapshot,
  sentiment,
  account,
  refreshing,
  onRefresh,
  onTriggerCycle,
  onOpenSettings,
  settings,
}: {
  snapshot: SnapshotResponse | null;
  sentiment: SentimentPoint[];
  account: AccountResponse | null;
  refreshing: boolean;
  onRefresh: () => void;
  onTriggerCycle: () => void;
  onOpenSettings: () => void;
  settings: Settings;
}) {
  const positions = snapshot?.positions ?? [];
  const alerts = snapshot?.alerts ?? [];
  const decisions = snapshot?.decisions ?? [];
  const marketStatus = snapshot?.marketStatus ?? null;
  const diagnostics = snapshot?.diagnostics ?? null;
  const watchlist = useMemo(() => settings.tickers.join(" · "), [settings.tickers]);
  const marketStatusText = useMemo(() => {
    if (!marketStatus) {
      return "Market status unavailable";
    }
    if (marketStatus.isOpen) {
      return `Market open — closes ${formatDateTime(marketStatus.nextClose)}`;
    }
    return `Market closed — next open ${formatDateTime(marketStatus.nextOpen)}`;
  }, [marketStatus]);

  return (
    <ScrollView
      contentContainerStyle={styles.scrollArea}
      refreshControl={<RefreshControl refreshing={refreshing} tintColor="#38bdf8" onRefresh={onRefresh} />}
    >
      <LinearGradient colors={["#0f172a", "#1e293b"]} style={styles.hero}>
        <View style={styles.heroHeader}>
          <View>
            <Text style={styles.heroTitle}>TraderV4 Observatory</Text>
            <Text style={styles.heroSubtitle}>Real-time sentiment, risk, and execution</Text>
          </View>
          <Pressable style={styles.heroButton} onPress={onOpenSettings}>
            <Text style={styles.heroButtonText}>Settings</Text>
          </Pressable>
        </View>
        <View style={styles.heroStatsRow}>
          <View style={styles.heroStatCard}>
            <Text style={styles.heroStatLabel}>Portfolio Value</Text>
            <Text style={styles.heroStatValue}>{currency(account?.portfolioValue)}</Text>
          </View>
          <View style={styles.heroStatCard}>
            <Text style={styles.heroStatLabel}>Buying Power</Text>
            <Text style={styles.heroStatValue}>{currency(account?.buyingPower)}</Text>
          </View>
          <View style={styles.heroStatCard}>
            <Text style={styles.heroStatLabel}>Available Cash</Text>
            <Text style={styles.heroStatValue}>{currency(account?.cash)}</Text>
          </View>
        </View>
        <View style={styles.heroFooter}>
          <Text style={styles.metaText}>{marketStatusText}</Text>
          <Text style={styles.metaText}>Updated {formatDateTime(snapshot?.lastUpdated)}</Text>
          <Text style={styles.metaText}>Watchlist: {watchlist || "—"}</Text>
        </View>
      </LinearGradient>

      <View style={styles.sectionCard}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Automation Controls</Text>
          <Pressable style={styles.buttonPrimary} onPress={onTriggerCycle}>
            <Text style={styles.buttonText}>Run Cycle</Text>
          </Pressable>
        </View>
        <PDTGauge summary={snapshot?.dayTrade ?? null} />
      </View>

      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Cycle Diagnostics</Text>
        {diagnostics ? (
          <View style={styles.diagnosticsGrid}>
            <View style={styles.diagnosticItem}>
              <Text style={styles.diagnosticValue}>{diagnostics.tickersProcessed}</Text>
              <Text style={styles.diagnosticLabel}>Tickers</Text>
            </View>
            <View style={styles.diagnosticItem}>
              <Text style={styles.diagnosticValue}>{diagnostics.alertsGenerated}</Text>
              <Text style={styles.diagnosticLabel}>Alerts</Text>
            </View>
            <View style={styles.diagnosticItem}>
              <Text style={styles.diagnosticValue}>{diagnostics.decisionsMade}</Text>
              <Text style={styles.diagnosticLabel}>Decisions</Text>
            </View>
            <View style={styles.diagnosticItem}>
              <Text style={styles.diagnosticValue}>{diagnostics.entries}</Text>
              <Text style={styles.diagnosticLabel}>Entries</Text>
            </View>
            <View style={styles.diagnosticItem}>
              <Text style={styles.diagnosticValue}>{diagnostics.exits}</Text>
              <Text style={styles.diagnosticLabel}>Exits</Text>
            </View>
            <View style={styles.diagnosticItem}>
              <Text style={styles.diagnosticValue}>{diagnostics.holds}</Text>
              <Text style={styles.diagnosticLabel}>Holds</Text>
            </View>
          </View>
        ) : (
          <Text style={styles.bodyText}>Waiting for first cycle…</Text>
        )}
        <Text style={styles.metaText}>
          Cycle window: {formatDateTime(diagnostics?.startedAt)} → {formatDateTime(diagnostics?.completedAt)}
        </Text>
        {diagnostics?.errors?.length ? (
          <Text style={styles.diagnosticError}>Errors: {diagnostics.errors.join(" · ")}</Text>
        ) : null}
      </View>

      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Sentiment Momentum</Text>
        <SentimentChart history={sentiment} />
      </View>

      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Exposure Breakdown</Text>
        <ExposureChart positions={positions} />
      </View>

      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Open Positions ({positions.length})</Text>
        {positions.length === 0 ? (
          <Text style={styles.bodyText}>No open positions.</Text>
        ) : (
          positions.map((position) => (
            <LinearGradient
              key={position.ticker}
              colors={["rgba(79,70,229,0.15)", "rgba(56,189,248,0.12)"]}
              style={styles.positionCard}
            >
              <Text style={styles.positionTicker}>{position.ticker}</Text>
              <Text style={styles.positionDetail}>Side: {position.side}</Text>
              <Text style={styles.positionDetail}>Qty: {position.quantity.toFixed(3)}</Text>
              <Text style={styles.positionDetail}>Value: {currency(position.marketValue)}</Text>
              {typeof position.costBasis === "number" && (
                <Text style={styles.positionDetail}>Cost Basis: {currency(position.costBasis)}</Text>
              )}
              <Text style={styles.positionTimestamp}>Updated {position.lastUpdated}</Text>
            </LinearGradient>
          ))
        )}
      </View>

      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Latest Sentiment Alerts</Text>
        {alerts.length === 0 ? (
          <Text style={styles.bodyText}>No alerts fired in the last cycle.</Text>
        ) : (
          alerts.map((alert) => (
            <View key={`${alert.ticker}-${alert.seenAt}`} style={styles.alertCard}>
              <Text style={styles.alertTitle}>{alert.ticker}</Text>
              <Text style={styles.alertReason}>{alert.reason}</Text>
              <View style={styles.alertRow}>
                <Text style={styles.alertMetric}>Avg {alert.averageSentiment.toFixed(2)}</Text>
                <Text style={styles.alertMetric}>Δ {alert.sentimentDelta.toFixed(2)}</Text>
                <Text style={styles.alertMetric}>Articles {alert.articleCount}</Text>
              </View>
              <Text style={styles.alertTimestamp}>{new Date(alert.seenAt).toLocaleString()}</Text>
            </View>
          ))
        )}
      </View>

      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Recent Decisions</Text>
        {decisions.length === 0 ? (
          <Text style={styles.bodyText}>No decisions recorded yet.</Text>
        ) : (
          decisions.slice(0, 6).map((decision) => (
            <View key={`${decision.ticker}-${decision.createdAt}`} style={styles.decisionCard}>
              <View style={styles.decisionHeader}>
                <Text style={styles.decisionTicker}>{decision.ticker}</Text>
                <Text style={styles.decisionIntent}>{decision.intent.toUpperCase()}</Text>
              </View>
              <Text style={styles.decisionMeta}>
                {decision.action} · {currency(decision.notional)} · {(decision.confidence * 100).toFixed(0)}% confidence
              </Text>
              <Text style={styles.decisionReason}>{decision.reason}</Text>
              <Text style={styles.decisionTimestamp}>{formatDateTime(decision.createdAt)}</Text>
            </View>
          ))
        )}
      </View>
    </ScrollView>
  );
}

type BacktestRequest = {
  tickers: string[];
  start: string;
  end: string;
  startingCash: number;
  sentimentWindowMinutes: number;
  timeframe: string;
  bots: string[];
  useVader: boolean;
  useFinbert: boolean;
  useKeybert: boolean;
  includeTrace: boolean;
  maxWorkers: number | null;
};

function BacktestScreen({
  loading,
  result,
  onRun,
}: {
  loading: boolean;
  result: BacktestResult | null;
  onRun: (payload: BacktestRequest) => Promise<void>;
}) {
  const [tickers, setTickers] = useState("AAPL,MSFT");
  const [engine, setEngine] = useState<Engine>("legacy");
  const [startDate, setStartDate] = useState(() => parseLocalDateTime("2024-01-02T09:30"));
  const [endDate, setEndDate] = useState(() => parseLocalDateTime("2024-01-05T16:00"));
  const [cash, setCash] = useState("100000");
  const [window, setWindow] = useState("60");
  const [timeframe, setTimeframe] = useState("15Min");
  const [bots, setBots] = useState("traderv4");
  const [useVader, setUseVader] = useState(true);
  const [useFinbert, setUseFinbert] = useState(false);
  const [useKeybert, setUseKeybert] = useState(false);
  const [includeTrace, setIncludeTrace] = useState(false);
  const [maxWorkers, setMaxWorkers] = useState("0");
  const [pickerTarget, setPickerTarget] = useState<"start" | "end" | null>(null);
  const [tempDay, setTempDay] = useState<string | null>(null);
  const [tempHour, setTempHour] = useState<string>("00");
  const [tempMinute, setTempMinute] = useState<string>("00");
  const startLabel = useMemo(() => formatReadableLocal(startDate), [startDate]);
  const endLabel = useMemo(() => formatReadableLocal(endDate), [endDate]);

  const applyDateChange = useCallback(
    (target: "start" | "end", next: Date) => {
      const normalized = new Date(next.getTime());
      if (target === "start") {
        setStartDate(normalized);
      } else {
        setEndDate(normalized);
      }
    },
    []
  );

  const openPicker = useCallback(
    (target: "start" | "end") => {
      const current = target === "start" ? startDate : endDate;
      setTempDay(formatIsoDateOnly(current));
      setTempHour(pad2(current.getHours()));
      setTempMinute(pad2(current.getMinutes()));
      setPickerTarget(target);
    },
    [startDate, endDate]
  );

  const closePicker = useCallback(() => setPickerTarget(null), []);

  const pickerLabel = pickerTarget === "start" ? "Start" : "End";

  const markedDates = useMemo(() => {
    if (!tempDay) {
      return undefined;
    }
    return {
      [tempDay]: {
        selected: true,
        selectedColor: "#2563eb",
        selectedTextColor: "#f8fafc",
      },
    };
  }, [tempDay]);

  const handleHourChange = useCallback((value: string) => {
    setTempHour(value.replace(/[^0-9]/g, "").slice(0, 2));
  }, []);

  const handleMinuteChange = useCallback((value: string) => {
    setTempMinute(value.replace(/[^0-9]/g, "").slice(0, 2));
  }, []);

  const commitPicker = useCallback(() => {
    if (!pickerTarget || !tempDay) {
      closePicker();
      return;
    }
    const base = pickerTarget === "start" ? startDate : endDate;
    const hour = clampTimePart(tempHour, 23, base.getHours());
    const minute = clampTimePart(tempMinute, 59, base.getMinutes());
    const combined = combineDateAndTime(tempDay, hour, minute);
    applyDateChange(pickerTarget, combined);
    closePicker();
  }, [applyDateChange, closePicker, pickerTarget, tempDay, tempHour, tempMinute, startDate, endDate]);

  const TogglePill = ({ label, value, onToggle }: { label: string; value: boolean; onToggle: () => void }) => (
    <Pressable
      onPress={onToggle}
      style={[styles.togglePill, value ? styles.togglePillActive : styles.togglePillInactive]}
    >
      <Text style={styles.toggleLabel}>{label}</Text>
      <Text style={styles.toggleValue}>{value ? "ON" : "OFF"}</Text>
    </Pressable>
  );

  const handleRun = async () => {
    try {
      if (startDate >= endDate) {
        Alert.alert("Invalid range", "Start time must be earlier than the end time.");
        return;
      }
      const workersText = maxWorkers.trim();
      const parsedWorkers = workersText === "" ? null : Number(workersText);
      if (parsedWorkers !== null && (!Number.isFinite(parsedWorkers) || parsedWorkers < 0)) {
        Alert.alert("Invalid workers", "Max workers must be a non-negative number.");
        return;
      }
      const payload: BacktestRequest = {
        tickers: tickers.split(/\s*,\s*/).filter(Boolean).map((t) => t.toUpperCase()),
        start: formatLocalInput(startDate),
        end: formatLocalInput(endDate),
        startingCash: Number(cash) || 100000,
        sentimentWindowMinutes: Number(window) || 60,
        timeframe,
        bots: bots.split(/\s*,\s*/).filter(Boolean),
        useVader,
        useFinbert,
        useKeybert,
        includeTrace,
        maxWorkers: parsedWorkers,
      };
      if (!payload.tickers.length) {
        Alert.alert("Missing tickers", "Please provide at least one ticker symbol.");
        return;
      }
      await onRun(payload);
    } catch (err) {
      Alert.alert("Backtest error", String(err));
    }
  };

  const firstSeries = useMemo(() => {
    if (!result) return null;
    const [firstKey, firstMetrics] = Object.entries(result.results)[0] ?? [];
    if (!firstKey || !firstMetrics.pnlCurve.length) return null;
    return {
      label: firstKey,
      data: firstMetrics.pnlCurve.map((value, index) => ({ x: index, y: value })),
    };
  }, [result]);

  return (
    <>
      <ScrollView contentContainerStyle={styles.backtestScroll}>
      <Text style={styles.pageTitle}>Backtesting Lab</Text>
      <Text style={styles.metaText}>
        Replay historical price and GDELT sentiment to evaluate strategy performance.
      </Text>
      <View style={styles.backtestForm}>
        <View style={styles.formRow}>
          <View style={styles.formField}>
            <Text style={styles.modalLabel}>Engine</Text>
            <View style={styles.toggleRow}>
              <Pressable
                onPress={() => setEngine("legacy")}
                style={[styles.togglePill, engine === "legacy" ? styles.togglePillActive : styles.togglePillInactive]}
              >
                <Text style={styles.toggleLabel}>Legacy</Text>
                <Text style={styles.toggleValue}>{engine === "legacy" ? "ON" : "OFF"}</Text>
              </Pressable>
              <Pressable
                onPress={() => setEngine("v5")}
                style={[styles.togglePill, engine === "v5" ? styles.togglePillActive : styles.togglePillInactive]}
              >
                <Text style={styles.toggleLabel}>TraderV5</Text>
                <Text style={styles.toggleValue}>{engine === "v5" ? "ON" : "OFF"}</Text>
              </Pressable>
            </View>
          </View>
        </View>
        <View style={styles.formRow}>
          <View style={styles.formField}>
            <Text style={styles.modalLabel}>Tickers</Text>
            <TextInput
              style={styles.modalInput}
              value={tickers}
              onChangeText={setTickers}
              placeholder="AAPL,MSFT"
            />
          </View>
          <View style={styles.formField}>
            <Text style={styles.modalLabel}>Bots</Text>
            <TextInput
              style={styles.modalInput}
              value={bots}
              onChangeText={setBots}
              placeholder="traderv4"
            />
          </View>
        </View>
        <View style={styles.formRow}>
          <View style={styles.formField}>
            <Text style={styles.modalLabel}>Start</Text>
            <Pressable
              onPress={() => openPicker("start")}
              style={styles.datePickerButton}
            >
              <Text style={styles.datePickerValue}>{startLabel}</Text>
            </Pressable>
          </View>
          <View style={styles.formField}>
            <Text style={styles.modalLabel}>End</Text>
            <Pressable
              onPress={() => openPicker("end")}
              style={styles.datePickerButton}
            >
              <Text style={styles.datePickerValue}>{endLabel}</Text>
            </Pressable>
          </View>
        </View>
        <View style={styles.formRow}>
          <View style={styles.formField}>
            <Text style={styles.modalLabel}>Starting Cash</Text>
            <TextInput
              style={styles.modalInput}
              value={cash}
              onChangeText={setCash}
              keyboardType="numeric"
            />
          </View>
          <View style={styles.formField}>
            <Text style={styles.modalLabel}>Sentiment Window (min)</Text>
            <TextInput
              style={styles.modalInput}
              value={window}
              onChangeText={setWindow}
              keyboardType="numeric"
            />
          </View>
          <View style={styles.formField}>
            <Text style={styles.modalLabel}>Timeframe</Text>
            <TextInput
              style={styles.modalInput}
              value={timeframe}
              onChangeText={setTimeframe}
              placeholder="15Min"
            />
          </View>
          <View style={styles.formField}>
            <Text style={styles.modalLabel}>Max Workers (0 = auto)</Text>
            <TextInput
              style={styles.modalInput}
              value={maxWorkers}
              onChangeText={setMaxWorkers}
              keyboardType="numeric"
              placeholder="0"
            />
          </View>
        </View>
        <View style={styles.toggleRow}>
          <TogglePill label="VADER" value={useVader} onToggle={() => setUseVader((v) => !v)} />
          <TogglePill label="FinBERT" value={useFinbert} onToggle={() => setUseFinbert((v) => !v)} />
          <TogglePill label="KeyBERT" value={useKeybert} onToggle={() => setUseKeybert((v) => !v)} />
          <TogglePill label="Trace" value={includeTrace} onToggle={() => setIncludeTrace((v) => !v)} />
        </View>
        {useFinbert && (
          <Text style={styles.metaText}>
            FinBERT loads large models on first run; expect 30-60 seconds of setup.
          </Text>
        )}
        <Pressable style={[styles.button, styles.buttonPrimary, styles.runButton]} onPress={handleRun}>
          <Text style={styles.buttonText}>{loading ? "Running…" : "Run Backtest"}</Text>
        </Pressable>
      </View>

      {result ? (
        <View style={styles.sectionCard}>
          <Text style={styles.sectionTitle}>Results</Text>
          {firstSeries ? (
            <VictoryChart
              width={360}
              height={220}
              padding={{ top: 40, bottom: 60, left: 60, right: 20 }}
            >
              <VictoryLegend
                x={100}
                y={10}
                orientation="horizontal"
                gutter={12}
                style={{ labels: { fill: "#cbd5f5", fontSize: 12 } }}
                data={[{ name: firstSeries.label, symbol: { fill: "#38bdf8" } }]}
              />
              <VictoryAxis
                tickFormat={(t) => `${t}`}
                style={{ tickLabels: { fill: "#94a3b8", fontSize: 11 } }}
              />
              <VictoryAxis
                dependentAxis
                tickFormat={(t) => `$${(t / 1000).toFixed(1)}k`}
                style={{ tickLabels: { fill: "#94a3b8", fontSize: 11 } }}
              />
              <VictoryLine data={firstSeries.data} style={{ data: { stroke: "#38bdf8", strokeWidth: 2 } }} />
            </VictoryChart>
          ) : (
            <Text style={styles.bodyText}>No P&L curve returned.</Text>
          )}
          {Object.entries(result.results).map(([key, metrics]) => (
            <View key={key} style={styles.resultCard}>
              <Text style={styles.alertTitle}>{key}</Text>
              <View style={styles.resultRow}>
                <Text style={styles.resultMetric}>Return: {currency(metrics.totalReturn)}</Text>
                <Text style={styles.resultMetric}>
                  Alert precision: {(metrics.alertPrecision * 100).toFixed(1)}%
                </Text>
                <Text style={styles.resultMetric}>Win rate: {(metrics.winRate * 100).toFixed(1)}%</Text>
                <Text style={styles.resultMetric}>
                  Max drawdown: {(metrics.maxDrawdown * 100).toFixed(1)}%
                </Text>
                <Text style={styles.resultMetric}>Trades: {metrics.tradeCount}</Text>
              </View>
            <Text style={styles.metaText}>
              Avg score: {metrics.featureSummary.averageScore.toFixed(3)} · Snapshots used: {metrics.featureSummary.snapshots}
            </Text>
            {metrics.featureSummary.topKeywords.length > 0 && (
              <Text style={styles.metaText}>
                Top keywords: {metrics.featureSummary.topKeywords.join(", ")}
              </Text>
            )}
            {metrics.trades.slice(0, 6).map((trade) => (
              <Text key={`${trade.timestamp}-${trade.ticker}`} style={styles.tradeLine}>
                {trade.timestamp}: {trade.action} {trade.ticker} {trade.quantity.toFixed(2)} @ ${trade.price.toFixed(2)} — {trade.reason}
              </Text>
            ))}
            {metrics.trades.length > 6 && (
              <Text style={styles.metaText}>…plus {metrics.trades.length - 6} more trades</Text>
            )}
            {metrics.debug.steps.length > 0 && (
              <DebugObservations
                tickerKey={key}
                steps={metrics.debug.steps}
                snapshotCount={metrics.debug.sentimentSnapshots}
                priceCount={metrics.debug.pricePoints}
              />
            )}
            </View>
          ))}
        </View>
      ) : (
        <View style={styles.sectionCard}>
          <Text style={styles.bodyText}>Run a backtest to see results.</Text>
        </View>
      )}
      </ScrollView>
      {pickerTarget && (
        <Modal transparent animationType="fade" onRequestClose={closePicker}>
          <View style={styles.dateModalBackdrop}>
            <View style={styles.dateModalContainer}>
              <Text style={styles.modalTitle}>Select {pickerLabel} Time</Text>
              <Calendar
                current={tempDay ?? formatIsoDateOnly(pickerTarget === "start" ? startDate : endDate)}
                onDayPress={(day) => setTempDay(day.dateString)}
                markedDates={markedDates}
                theme={{
                  calendarBackground: "#0f172a",
                  dayTextColor: "#e2e8f0",
                  textDisabledColor: "#475569",
                  monthTextColor: "#f8fafc",
                  arrowColor: "#38bdf8",
                  selectedDayBackgroundColor: "#2563eb",
                  selectedDayTextColor: "#f8fafc",
                  todayTextColor: "#38bdf8",
                }}
                style={styles.calendar}
              />
              <View style={styles.timeRow}>
                <View style={styles.timeField}>
                  <Text style={styles.modalLabel}>Hour (0-23)</Text>
                  <TextInput
                    style={styles.timeInput}
                    keyboardType="numeric"
                    value={tempHour}
                    onChangeText={handleHourChange}
                    maxLength={2}
                  />
                </View>
                <View style={styles.timeField}>
                  <Text style={styles.modalLabel}>Minute (0-59)</Text>
                  <TextInput
                    style={styles.timeInput}
                    keyboardType="numeric"
                    value={tempMinute}
                    onChangeText={handleMinuteChange}
                    maxLength={2}
                  />
                </View>
              </View>
              <View style={styles.dateModalActions}>
                <Pressable
                  style={[styles.button, styles.buttonSecondary, styles.dateModalButton]}
                  onPress={closePicker}
                >
                  <Text style={styles.buttonText}>Cancel</Text>
                </Pressable>
                <Pressable
                  style={[styles.button, styles.buttonPrimary, styles.dateModalButton]}
                  onPress={commitPicker}
                >
                  <Text style={styles.buttonText}>Apply</Text>
                </Pressable>
              </View>
            </View>
          </View>
        </Modal>
      )}
    </>
  );
}

function DebugObservations({
  tickerKey,
  steps,
  snapshotCount,
  priceCount,
}: {
  tickerKey: string;
  steps: Array<any>;
  snapshotCount: number;
  priceCount: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => {
    if (!query.trim()) {
      return steps;
    }
    const needle = query.trim().toLowerCase();
    return steps.filter((step) =>
      Object.values(step).some((value) =>
        typeof value === "string" && value.toLowerCase().includes(needle)
      )
    );
  }, [steps, query]);

  return (
    <View style={styles.debugBlock}>
      <Pressable onPress={() => setExpanded((prev) => !prev)} style={styles.debugHeader}>
        <Text style={styles.metaText}>
          {expanded ? "Hide" : "Show"} {steps.length} observations (sentiment snapshots: {snapshotCount}, price bars: {priceCount})
        </Text>
        <Text style={styles.toggleValue}>{expanded ? "▲" : "▼"}</Text>
      </Pressable>
      {expanded && (
        <View style={styles.debugBody}>
          <TextInput
            style={styles.debugSearch}
            placeholder="Search observations"
            placeholderTextColor="#64748b"
            value={query}
            onChangeText={setQuery}
            autoCapitalize="none"
            autoCorrect={false}
          />
          <View style={styles.debugList}>
            {filtered.map((step, index) => (
              <Text key={`${tickerKey}-debug-${index}`} style={styles.debugLine}>
                {step.timestamp}: price ${Number(step.price).toFixed(2)} tone {Number(step.tone15).toFixed(2)} Δ {Number(step.delta15).toFixed(2)} → {step.action}
                {typeof step.score === "string" ? ` score ${step.score}` : ""}
              </Text>
            ))}
            {filtered.length === 0 && (
              <Text style={styles.metaText}>No observations match "{query}".</Text>
            )}
          </View>
        </View>
      )}
    </View>
  );
}

export default function App(): JSX.Element {
  const [screen, setScreen] = useState<Screen>("dashboard");
  const [snapshot, setSnapshot] = useState<SnapshotResponse | null>(null);
  const [sentiment, setSentiment] = useState<SentimentPoint[]>([]);
  const [account, setAccount] = useState<AccountResponse | null>(null);
  const [settings, setSettings] = useState<Settings>(emptySettings);
  const [refreshing, setRefreshing] = useState(false);
  const [settingsVisible, setSettingsVisible] = useState(false);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);

  const loadAll = useCallback(async () => {
    setRefreshing(true);
    try {
      const [nextSnapshot, nextSentiment, nextAccount, nextSettings] = await Promise.all([
        fetchJSON<SnapshotResponse>("/snapshot"),
        fetchJSON<SentimentResponse>("/sentiment"),
        fetchJSON<AccountResponse>("/account"),
        fetchJSON<Settings>("/settings"),
      ]);
      setSnapshot(nextSnapshot);
      setSentiment(nextSentiment.history || []);
      setAccount(nextAccount);
      setSettings(nextSettings);
    } catch (err) {
      console.warn("Load failure", err);
    } finally {
      setRefreshing(false);
    }
  }, []);

  const triggerCycle = useCallback(async () => {
    try {
      await fetchJSON("/run", { method: "POST", body: JSON.stringify({}) });
      await loadAll();
      Alert.alert("Cycle triggered", "Run cycle completed.");
    } catch (err) {
      Alert.alert("Unable to trigger cycle", String(err));
    }
  }, [loadAll]);

  const updateSettings = useCallback(
    async (nextSettings: Settings) => {
      await fetchJSON<Settings>("/settings", {
        method: "POST",
        body: JSON.stringify(nextSettings),
      });
      await loadAll();
    },
    [loadAll]
  );

  const runBacktest = useCallback(
    async (payload: BacktestRequest) => {
      setBacktestLoading(true);
      try {
        const result = await fetchJSON<BacktestResult>("/backtester/run", {
          method: "POST",
          body: JSON.stringify({
            tickers: payload.tickers,
            start: payload.start,
            end: payload.end,
            startingCash: payload.startingCash,
            sentimentWindowMinutes: payload.sentimentWindowMinutes,
            timeframe: payload.timeframe,
            bots: payload.bots,
            useVader: payload.useVader,
            useFinbert: payload.useFinbert,
            useKeybert: payload.useKeybert,
            includeTrace: payload.includeTrace,
            maxWorkers: payload.maxWorkers,
          }),
        });
        setBacktestResult(result);
      } catch (err) {
        Alert.alert("Backtest error", String(err));
      } finally {
        setBacktestLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    loadAll();
    const id = setInterval(loadAll, 20000);
    return () => clearInterval(id);
  }, [loadAll]);

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar barStyle="light-content" />
      <ExpoStatusBar style="light" />
      <View style={styles.appLayout}>
        <Sidebar screen={screen} setScreen={setScreen} />
        <View style={styles.content}>
          {screen === "dashboard" ? (
            <DashboardScreen
              snapshot={snapshot}
              sentiment={sentiment}
              account={account}
              refreshing={refreshing}
              onRefresh={loadAll}
              onTriggerCycle={triggerCycle}
              onOpenSettings={() => setSettingsVisible(true)}
              settings={settings}
            />
          ) : (
            <BacktestScreen loading={backtestLoading} result={backtestResult} onRun={runBacktest} />
          )}
        </View>
      </View>
      <SettingsModal
        visible={settingsVisible}
        onClose={() => setSettingsVisible(false)}
        settings={settings}
        onSubmit={updateSettings}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: "#0b1120",
  },
  appLayout: {
    flex: 1,
    flexDirection: "row",
  },
  sidebar: {
    width: 200,
    backgroundColor: "#0f172a",
    paddingVertical: 32,
    paddingHorizontal: 18,
    gap: 16,
    shadowColor: "#000",
    shadowOpacity: 0.3,
    shadowOffset: { width: 0, height: 6 },
    shadowRadius: 12,
    elevation: 6,
  },
  sidebarTitle: {
    fontSize: 20,
    fontWeight: "700",
    color: "#38bdf8",
    marginBottom: 24,
  },
  sidebarButton: {
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 16,
    backgroundColor: "transparent",
  },
  sidebarButtonActive: {
    backgroundColor: "rgba(56,189,248,0.15)",
  },
  sidebarButtonText: {
    color: "#e2e8f0",
    fontSize: 15,
    fontWeight: "600",
  },
  content: {
    flex: 1,
  },
  scrollArea: {
    padding: 16,
    paddingBottom: 48,
  },
  backtestScroll: {
    padding: 24,
    gap: 20,
  },
  pageTitle: {
    fontSize: 26,
    fontWeight: "700",
    color: "#f8fafc",
  },
  hero: {
    borderRadius: 24,
    padding: 20,
    marginBottom: 24,
    gap: 16,
  },
  heroHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  heroButton: {
    backgroundColor: "rgba(255,255,255,0.12)",
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 999,
  },
  heroButtonText: {
    color: "#f8fafc",
    fontSize: 14,
    fontWeight: "600",
  },
  heroTitle: {
    fontSize: 26,
    fontWeight: "700",
    color: "#f8fafc",
  },
  heroSubtitle: {
    fontSize: 15,
    color: "#cbd5f5",
    marginTop: 4,
  },
  heroStatsRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 12,
  },
  heroStatCard: {
    flex: 1,
    backgroundColor: "rgba(15,23,42,0.6)",
    borderRadius: 18,
    padding: 16,
  },
  heroStatLabel: {
    color: "#94a3b8",
    fontSize: 13,
  },
  heroStatValue: {
    color: "#f1f5f9",
    fontSize: 18,
    fontWeight: "600",
    marginTop: 6,
  },
  heroFooter: {
    gap: 4,
  },
  sectionCard: {
    backgroundColor: "#131c31",
    borderRadius: 20,
    padding: 18,
    marginBottom: 20,
    shadowColor: "#000",
    shadowOpacity: 0.3,
    shadowOffset: { width: 0, height: 6 },
    shadowRadius: 12,
    elevation: 6,
  },
  sectionHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: "600",
    color: "#e2e8f0",
  },
  bodyText: {
    fontSize: 15,
    color: "#cbd5f5",
  },
  metaText: {
    fontSize: 13,
    color: "#94a3b8",
  },
  pdtContainer: {
    gap: 10,
  },
  pdtBarBackground: {
    height: 14,
    borderRadius: 999,
    flexDirection: "row",
    overflow: "hidden",
  },
  pdtBarFill: {
    backgroundColor: "rgba(14,165,233,0.8)",
  },
  pdtText: {
    fontSize: 16,
    color: "#e2e8f0",
  },
  alertCard: {
    backgroundColor: "rgba(15,23,42,0.85)",
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
  },
  alertTitle: {
    fontSize: 18,
    fontWeight: "600",
    color: "#f8fafc",
  },
  alertReason: {
    fontSize: 14,
    color: "#cbd5f5",
    marginVertical: 6,
  },
  alertRow: {
    flexDirection: "row",
    gap: 12,
  },
  alertMetric: {
    fontSize: 13,
    color: "#818cf8",
  },
  alertTimestamp: {
    fontSize: 12,
    color: "#94a3b8",
    marginTop: 6,
  },
  diagnosticsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
    marginVertical: 12,
  },
  diagnosticItem: {
    flexGrow: 1,
    minWidth: 120,
    borderRadius: 12,
    padding: 12,
    backgroundColor: "rgba(148,163,184,0.15)",
  },
  diagnosticValue: {
    fontSize: 18,
    fontWeight: "600",
    color: "#f8fafc",
  },
  diagnosticLabel: {
    fontSize: 12,
    color: "#94a3b8",
    marginTop: 4,
    textTransform: "uppercase",
    letterSpacing: 0.6,
  },
  diagnosticError: {
    color: "#f87171",
    fontSize: 13,
    marginTop: 4,
  },
  positionCard: {
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
  },
  positionTicker: {
    fontSize: 18,
    fontWeight: "600",
    color: "#f8fafc",
  },
  positionDetail: {
    fontSize: 14,
    color: "#cbd5f5",
    marginTop: 2,
  },
  positionTimestamp: {
    fontSize: 12,
    color: "#94a3b8",
    marginTop: 8,
  },
  decisionCard: {
    backgroundColor: "rgba(15,23,42,0.85)",
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
  },
  decisionHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  decisionTicker: {
    fontSize: 18,
    fontWeight: "600",
    color: "#f8fafc",
  },
  decisionIntent: {
    fontSize: 13,
    fontWeight: "600",
    color: "#38bdf8",
  },
  decisionMeta: {
    fontSize: 13,
    color: "#cbd5f5",
    marginTop: 6,
  },
  decisionReason: {
    fontSize: 14,
    color: "#e2e8f0",
    marginTop: 6,
  },
  decisionTimestamp: {
    fontSize: 12,
    color: "#94a3b8",
    marginTop: 6,
  },
  button: {
    borderRadius: 999,
    paddingHorizontal: 16,
    paddingVertical: 10,
    minWidth: 110,
    alignItems: "center",
  },
  buttonPrimary: {
    backgroundColor: "#2563eb",
  },
  buttonSecondary: {
    backgroundColor: "rgba(148, 163, 184, 0.2)",
  },
  buttonText: {
    color: "#f8fafc",
    fontWeight: "600",
  },
  modalContainer: {
    flex: 1,
    backgroundColor: "#0b1120",
  },
  modalTitle: {
    fontSize: 24,
    fontWeight: "700",
    color: "#f8fafc",
    padding: 20,
    paddingBottom: 0,
  },
  modalForm: {
    paddingHorizontal: 20,
    paddingBottom: 40,
    gap: 14,
  },
  modalLabel: {
    color: "#94a3b8",
    fontSize: 14,
  },
  modalInput: {
    backgroundColor: "#131c31",
    padding: Platform.select({ ios: 14, default: 12 }),
    borderRadius: 12,
    color: "#f8fafc",
  },
  datePickerButton: {
    backgroundColor: "#131c31",
    paddingVertical: Platform.select({ ios: 14, default: 12 }),
    paddingHorizontal: 12,
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(148,163,184,0.35)",
  },
  datePickerValue: {
    color: "#f8fafc",
    fontSize: 15,
  },
  modalDivider: {
    borderBottomColor: "rgba(148, 163, 184, 0.2)",
    borderBottomWidth: StyleSheet.hairlineWidth,
    marginVertical: 12,
  },
  modalToggleRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  modalToggleButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 12,
    backgroundColor: "rgba(148,163,184,0.25)",
  },
  modalToggleButtonActive: {
    backgroundColor: "#2563eb",
  },
  modalToggleText: {
    color: "#f8fafc",
    fontWeight: "600",
  },
  modalButtons: {
    flexDirection: "row",
    justifyContent: "space-between",
    padding: 20,
    paddingTop: 0,
    gap: 12,
  },
  backtestForm: {
    backgroundColor: "#131c31",
    borderRadius: 16,
    padding: 18,
    gap: 16,
  },
  dateModalBackdrop: {
    flex: 1,
    backgroundColor: "rgba(8,11,22,0.7)",
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
  },
  dateModalContainer: {
    width: "100%",
    maxWidth: 360,
    backgroundColor: "#0f172a",
    borderRadius: 20,
    padding: 20,
    gap: 18,
  },
  dateModalButton: {
    flex: 1,
  },
  dateModalActions: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 12,
  },
  calendar: {
    alignSelf: "stretch",
    borderRadius: 12,
    overflow: "hidden",
  },
  timeRow: {
    flexDirection: "row",
    gap: 12,
  },
  timeField: {
    flex: 1,
    gap: 6,
  },
  timeInput: {
    backgroundColor: "#131c31",
    paddingVertical: Platform.select({ ios: 14, default: 12 }),
    paddingHorizontal: 12,
    borderRadius: 12,
    color: "#f8fafc",
    textAlign: "center",
    fontSize: 18,
  },
  formRow: {
    flexDirection: "row",
    gap: 12,
  },
  formField: {
    flex: 1,
    gap: 6,
  },
  toggleRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
    marginTop: 12,
  },
  togglePill: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 999,
    minWidth: 110,
  },
  togglePillActive: {
    backgroundColor: "rgba(56,189,248,0.25)",
  },
  togglePillInactive: {
    backgroundColor: "rgba(148,163,184,0.15)",
  },
  toggleLabel: {
    color: "#e2e8f0",
    fontSize: 13,
    fontWeight: "500",
  },
  toggleValue: {
    color: "#94a3b8",
    fontSize: 12,
    fontWeight: "600",
  },
  runButton: {
    alignSelf: "flex-start",
    paddingHorizontal: 24,
  },
  resultCard: {
    marginTop: 14,
    backgroundColor: "rgba(15,23,42,0.85)",
    borderRadius: 14,
    padding: 14,
    gap: 6,
  },
  resultRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
  },
  resultMetric: {
    color: "#f1f5f9",
    fontSize: 14,
  },
  tradeLine: {
    color: "#cbd5f5",
    fontSize: 12,
  },
  debugBlock: {
    backgroundColor: "rgba(148, 163, 184, 0.08)",
    borderRadius: 10,
    padding: 10,
    gap: 8,
  },
  debugLine: {
    color: "#94a3b8",
    fontSize: 11,
  },
  debugHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  debugBody: {
    gap: 8,
  },
  debugSearch: {
    backgroundColor: "#0f172a",
    borderRadius: 8,
    paddingVertical: Platform.select({ ios: 10, default: 8 }),
    paddingHorizontal: 12,
    color: "#f8fafc",
    fontSize: 13,
  },
  debugList: {
    gap: 4,
    maxHeight: 240,
  },
});
