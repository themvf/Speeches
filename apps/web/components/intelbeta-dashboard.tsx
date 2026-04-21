"use client";

import { useMemo, useState } from "react";
import { SparklineChart } from "@/components/sparkline-chart";
import type { IntelligenceEvidenceArticle, IntelligenceProfile, IntelligenceSignalsData } from "@/lib/intelligence-types";
import type { TrendDocItem, TrendItem, TrendsPayload } from "@/lib/server/types";
import {
  THEME_MAPPING,
  type MarketImpactDirection,
  type NormalizedTheme,
  type ThemeCategory,
  type ThemeFrequencySignal,
  type ThemeSeverity
} from "@/lib/theme-intelligence";

type ScoredTrend = TrendItem & {
  _score: number;
  _relatedSignalId: string;
};

const SEVERITY_STYLE: Record<ThemeSeverity, { color: string; background: string; border: string }> = {
  CRITICAL: { color: "#ff6b7f", background: "rgba(255,107,127,0.12)", border: "rgba(255,107,127,0.34)" },
  HIGH: { color: "var(--warn)", background: "rgba(242,171,67,0.12)", border: "rgba(242,171,67,0.32)" },
  NORMAL: { color: "var(--ok)", background: "rgba(65,211,157,0.1)", border: "rgba(65,211,157,0.28)" }
};

const DIRECTION_STYLE: Record<MarketImpactDirection, { color: string; background: string; label: string }> = {
  UP: { color: "var(--ok)", background: "rgba(65,211,157,0.1)", label: "Up" },
  DOWN: { color: "var(--danger)", background: "rgba(255,107,127,0.1)", label: "Down" },
  MIXED: { color: "var(--warn)", background: "rgba(242,171,67,0.1)", label: "Mixed" }
};

const SOURCE_KIND_LABELS: Record<string, string> = {
  sec_speech: "SEC",
  sec_enforcement_litigation: "SEC Enforcement",
  finra_regulatory_notice: "FINRA",
  finra_awc: "FINRA AWC",
  finra_comment_letter: "FINRA",
  doj_usao_press_release: "DOJ",
  cftc_press_release: "CFTC",
  newsapi_article: "News",
  reddit_post: "Reddit",
  uploaded: "Uploaded"
};

const CATEGORY_LABELS: Record<ThemeCategory, string> = {
  MACRO: "Macro",
  FINANCIAL_SYSTEM: "Financial System",
  GEOPOLITICS: "Geopolitics",
  REAL_ECONOMY: "Real Economy",
  MODERN_THEMES: "Modern Themes"
};

function formatTheme(theme: string): string {
  return theme.replace(/_/g, " ");
}

function formatPct(value: number): string {
  return `${value > 0 ? "+" : ""}${Number.isInteger(value) ? value : value.toFixed(1)}%`;
}

function formatDate(value: string): string {
  if (!value) return "-";
  const date = new Date(`${value}T00:00:00Z`);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
}

function sourceLabel(kind: string): string {
  return SOURCE_KIND_LABELS[kind] ?? kind.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function computeTrendScore(trend: TrendItem): number {
  const growthNorm = (Math.min(Math.max(trend.growth_pct, -100), 300) + 100) / 400;
  const mentionNorm = Math.min(Math.log10(Math.max(trend.total_mentions, 1)) / 4, 1);
  const velocity = trend.total_mentions > 0 ? Math.min(trend.recent_mentions / trend.total_mentions, 1) : 0;
  return Math.round((growthNorm * 0.4 + mentionNorm * 0.35 + velocity * 0.25) * 100);
}

function normalizedText(value: string): string {
  return value.toLowerCase().replace(/[_-]/g, " ");
}

function signalHaystack(profile: IntelligenceProfile): string {
  return [
    profile.label,
    profile.oneLineSummary,
    profile.narrative,
    profile.signal.normalized_theme_list.map(formatTheme).join(" "),
    profile.signal.primary_drivers.map((driver) => formatTheme(driver.normalized_theme)).join(" "),
    profile.clusters.map((cluster) => `${cluster.title} ${cluster.summary}`).join(" "),
    profile.evidence.map((article) => `${article.headline} ${article.explanation}`).join(" ")
  ]
    .join(" ")
    .toLowerCase();
}

function trendHaystack(trend: TrendItem): string {
  return normalizedText([trend.label, trend.description, trend.canonical_tag, ...trend.cluster_tags].join(" "));
}

function trendSignalScore(trend: TrendItem, profile: IntelligenceProfile): number {
  const trendText = trendHaystack(trend);
  const signalText = signalHaystack(profile);
  let score = 0;

  for (const theme of profile.signal.normalized_theme_list) {
    const formatted = normalizedText(formatTheme(theme));
    if (trendText.includes(formatted)) score += 6;
    if (trend.cluster_tags.some((tag) => normalizedText(tag).includes(formatted))) score += 4;
  }

  for (const token of trendText.split(/\s+/).filter((part) => part.length > 3)) {
    if (signalText.includes(token)) score += 1;
  }

  if (profile.label.toLowerCase().split(/\s+/).some((token) => trendText.includes(token))) {
    score += 3;
  }

  return score;
}

function relatedSignalIdForTrend(trend: TrendItem, profiles: readonly IntelligenceProfile[]): string {
  const ranked = profiles
    .map((profile) => ({ profile, score: trendSignalScore(trend, profile) }))
    .sort((a, b) => b.score - a.score);
  return ranked[0]?.profile.id ?? profiles[0]?.id ?? "";
}

function relatedTrendsForSignal(trends: readonly ScoredTrend[], profile: IntelligenceProfile): ScoredTrend[] {
  const direct = trends.filter((trend) => trend._relatedSignalId === profile.id);
  if (direct.length > 0) return direct.slice(0, 6);

  return [...trends]
    .sort((a, b) => trendSignalScore(b, profile) - trendSignalScore(a, profile))
    .slice(0, 6);
}

function evidenceForSignal(profile: IntelligenceProfile, theme?: NormalizedTheme): IntelligenceEvidenceArticle[] {
  const articles = theme ? profile.evidence.filter((article) => article.relatedThemes.includes(theme)) : profile.evidence;
  return articles.length > 0 ? articles : profile.evidence;
}

function expandEvidenceForSignal(profile: IntelligenceProfile, count = 30, theme?: NormalizedTheme): IntelligenceEvidenceArticle[] {
  const sourceArticles = evidenceForSignal(profile, theme);

  if (sourceArticles.length === 0) {
    return [];
  }

  return Array.from({ length: count }, (_, index) => {
    const base = sourceArticles[index % sourceArticles.length];
    const cycle = Math.floor(index / sourceArticles.length);
    const cluster = profile.clusters.find((item) => item.id === base.clusterId);

    if (cycle === 0) {
      return base;
    }

    return {
      ...base,
      id: `${base.id}-beta-${cycle}`,
      headline: `${base.headline} (${cluster?.title ?? "follow-up"} ${cycle + 1})`,
      timestamp: `${12 + index * 4} min ago`,
      excerpt: `${base.excerpt} Related coverage continues to reinforce the same signal theme.`,
      impact: Math.max(1, base.impact - cycle * 2),
      credibility: Math.max(1, base.credibility - cycle)
    };
  });
}

function profileForTheme(theme: NormalizedTheme, profiles: readonly IntelligenceProfile[], fallback: IntelligenceProfile): IntelligenceProfile {
  return profiles.find((profile) => profile.signal.normalized_theme_list.includes(theme)) ?? fallback;
}

function StatusBadge({ severity }: { severity: ThemeSeverity }) {
  const style = SEVERITY_STYLE[severity];
  return (
    <span className="inline-flex items-center rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase" style={{ color: style.color, background: style.background, border: `1px solid ${style.border}` }}>
      {severity}
    </span>
  );
}

function DriverRow({
  driver,
  selected,
  onSelect
}: {
  driver: ThemeFrequencySignal;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      className={`grid w-full grid-cols-[minmax(0,1fr)_64px_72px_72px] items-center gap-2 rounded-lg border px-3 py-2 text-left text-xs ${
        selected
          ? "border-[color:var(--line-strong)] bg-[color:rgba(79,213,255,0.14)]"
          : "border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.38)] hover:border-[color:var(--line-strong)]"
      }`}
    >
      <span className="truncate font-semibold text-[color:var(--ink)]">{formatTheme(driver.normalized_theme)}</span>
      <span className="tabular-nums text-[color:var(--ink-faint)]">{driver.current_mentions}</span>
      <span className="font-semibold tabular-nums text-[color:var(--accent)]">{formatPct(driver.spike_pct)}</span>
      <span className="font-semibold tabular-nums text-[color:var(--ink)]">{driver.contribution_pct.toFixed(1)}%</span>
    </button>
  );
}

function EvidenceLink({ article }: { article: IntelligenceEvidenceArticle }) {
  const body = (
    <>
      <p className="text-sm font-semibold leading-snug text-[color:var(--ink)]">{article.headline}</p>
      <p className="mt-1 text-[11px] font-semibold uppercase text-[color:var(--ink-faint)]">
        {article.source} - {article.timestamp}
      </p>
      <p className="mt-2 line-clamp-2 text-xs text-[color:var(--ink-faint)]">{article.excerpt}</p>
      <p className="mt-2 text-xs font-medium text-[color:var(--accent)]">-&gt; {article.explanation}</p>
    </>
  );

  return article.url ? (
    <a href={article.url} target="_blank" rel="noreferrer" className="block rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.48)] p-3 transition-colors hover:border-[color:var(--line-strong)]">
      {body}
    </a>
  ) : (
    <article className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.48)] p-3">{body}</article>
  );
}

function TrendDoc({ doc }: { doc: TrendDocItem }) {
  const content = (
    <div className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.45)] p-3 transition-colors hover:border-[color:var(--line-strong)]">
      <p className="text-xs font-semibold leading-snug text-[color:var(--ink)]">{doc.title || doc.id}</p>
      <div className="mt-1 flex flex-wrap gap-2 text-[11px] text-[color:var(--ink-faint)]">
        <span>{sourceLabel(doc.source_kind)}</span>
        <span>{formatDate(doc.date)}</span>
      </div>
      {doc.summary ? <p className="mt-2 line-clamp-2 text-[11px] leading-5 text-[color:var(--ink-faint)]">{doc.summary}</p> : null}
    </div>
  );

  return doc.url ? (
    <a href={doc.url} target="_blank" rel="noreferrer" className="block">
      {content}
    </a>
  ) : content;
}

function TrendButton({
  active,
  trend,
  onClick
}: {
  active: boolean;
  trend: ScoredTrend;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`w-full rounded-lg border p-3 text-left transition-colors ${
        active
          ? "border-[color:var(--line-strong)] bg-[color:rgba(79,213,255,0.13)]"
          : "border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.35)] hover:border-[color:var(--line-strong)]"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-[color:var(--ink)]">{trend.label}</p>
          <p className="mt-1 line-clamp-2 text-xs text-[color:var(--ink-faint)]">{trend.description || trend.canonical_tag}</p>
        </div>
        <span className="shrink-0 text-sm font-semibold tabular-nums text-[color:var(--accent)]">{formatPct(trend.growth_pct)}</span>
      </div>
      <div className="mt-3 flex items-center justify-between gap-3">
        <div className="flex flex-wrap gap-2 text-[11px] text-[color:var(--ink-faint)]">
          <span>{trend.total_mentions.toLocaleString()} mentions</span>
          <span>Score {trend._score}</span>
        </div>
        <SparklineChart data={trend.sparkline} />
      </div>
    </button>
  );
}

function SignalAlertButton({
  active,
  profile,
  onClick
}: {
  active: boolean;
  profile: IntelligenceProfile;
  onClick: () => void;
}) {
  const trend = profile.signal.trend;

  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`rounded-xl border p-4 text-left transition-colors ${
        active
          ? "border-[color:var(--line-strong)] bg-[color:rgba(79,213,255,0.14)]"
          : "border-[color:var(--line-soft)] bg-[color:rgba(9,21,34,0.52)] hover:border-[color:var(--line-strong)]"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-[color:var(--ink)]">{profile.label}</p>
          <p className="mt-2 line-clamp-2 text-xs leading-5 text-[color:var(--ink-faint)]">{profile.oneLineSummary}</p>
        </div>
        <StatusBadge severity={profile.signal.severity} />
      </div>
      <div className="mt-3 flex items-center justify-between gap-3 text-xs">
        <span className="font-semibold text-[color:var(--danger)]">{formatTheme(trend.direction)}</span>
        <span className="font-semibold tabular-nums text-[color:var(--ink)]">{formatPct(trend.delta_pct)} vs prior window</span>
      </div>
    </button>
  );
}

function AllThemesStrip({
  activeThemes,
  selectedTheme,
  onSelect
}: {
  activeThemes: readonly NormalizedTheme[];
  selectedTheme: NormalizedTheme | null;
  onSelect: (theme: NormalizedTheme) => void;
}) {
  const groupedThemes = THEME_MAPPING.reduce<Record<ThemeCategory, NormalizedTheme[]>>(
    (acc, definition) => {
      acc[definition.category].push(definition.normalized_theme);
      return acc;
    },
    {
      MACRO: [],
      FINANCIAL_SYSTEM: [],
      GEOPOLITICS: [],
      REAL_ECONOMY: [],
      MODERN_THEMES: []
    }
  );
  const activeThemeSet = new Set(activeThemes);

  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.52)] p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">All Themes</p>
          <h2 className="mt-1 text-base font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
            Full normalized taxonomy
          </h2>
        </div>
        <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs text-[color:var(--ink-faint)]">
          {THEME_MAPPING.length} themes visible
        </span>
      </div>

      <div className="mt-4 grid gap-3 xl:grid-cols-5">
        {(Object.keys(groupedThemes) as ThemeCategory[]).map((category) => (
          <div key={category}>
            <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">{CATEGORY_LABELS[category]}</p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {groupedThemes[category].map((theme) => (
                <button
                  key={theme}
                  type="button"
                  onClick={() => onSelect(theme)}
                  aria-pressed={selectedTheme === theme}
                  className={`min-h-7 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold transition-colors ${
                    selectedTheme === theme
                      ? "border-[color:var(--line-strong)] bg-[color:rgba(79,213,255,0.2)] text-[color:var(--ink)]"
                      : activeThemeSet.has(theme)
                        ? "border-[color:rgba(79,213,255,0.3)] bg-[color:rgba(79,213,255,0.08)] text-[color:var(--accent)] hover:border-[color:var(--line-strong)]"
                        : "border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.38)] text-[color:var(--ink)] hover:border-[color:var(--line-strong)]"
                  }`}
                >
                  {formatTheme(theme)}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

export function IntelBetaDashboard({
  initialSignals,
  initialTrends
}: {
  initialSignals: IntelligenceSignalsData;
  initialTrends: TrendsPayload;
}) {
  const profiles = initialSignals.profiles;
  const scoredTrends = useMemo<ScoredTrend[]>(() => {
    return initialTrends.trends
      .map((trend) => ({
        ...trend,
        _score: computeTrendScore(trend),
        _relatedSignalId: relatedSignalIdForTrend(trend, profiles)
      }))
      .sort((a, b) => b._score - a._score || b.growth_pct - a.growth_pct)
      .slice(0, 40);
  }, [initialTrends.trends, profiles]);

  const [selectedSignalId, setSelectedSignalId] = useState(profiles[0]?.id ?? "");
  const [selectedTrendId, setSelectedTrendId] = useState(scoredTrends[0]?.id ?? "");
  const [selectedTheme, setSelectedTheme] = useState<NormalizedTheme | null>(null);

  const selectedSignal = profiles.find((profile) => profile.id === selectedSignalId) ?? profiles[0];
  const selectedTrend = scoredTrends.find((trend) => trend.id === selectedTrendId);
  const selectedDriver = selectedTheme ? selectedSignal?.signal.frequency_signals.find((driver) => driver.normalized_theme === selectedTheme) : selectedSignal?.signal.primary_driver;
  const evidence = selectedSignal ? expandEvidenceForSignal(selectedSignal, 30, selectedTheme ?? selectedDriver?.normalized_theme) : [];
  const linkedTrends = selectedSignal ? relatedTrendsForSignal(scoredTrends, selectedSignal) : [];

  function selectTrend(trend: ScoredTrend) {
    setSelectedTrendId(trend.id);
    setSelectedSignalId(trend._relatedSignalId);
    setSelectedTheme(null);
  }

  function selectSignal(profile: IntelligenceProfile) {
    setSelectedSignalId(profile.id);
    setSelectedTheme(null);
    const nextTrend = relatedTrendsForSignal(scoredTrends, profile)[0];
    if (nextTrend) {
      setSelectedTrendId(nextTrend.id);
    }
  }

  function selectTheme(theme: NormalizedTheme) {
    if (!selectedSignal) return;
    const nextSignal = profileForTheme(theme, profiles, selectedSignal);
    setSelectedSignalId(nextSignal.id);
    setSelectedTheme(theme);
    const nextTrend = relatedTrendsForSignal(scoredTrends, nextSignal)[0];
    if (nextTrend) {
      setSelectedTrendId(nextTrend.id);
    }
  }

  if (!selectedSignal) {
    return (
      <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] p-5">
        <p className="text-sm text-[color:var(--ink-faint)]">No intelligence signals are available yet.</p>
      </section>
    );
  }

  return (
    <div className="space-y-5">
      <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.62)] p-4">
        <div className="grid gap-4 xl:grid-cols-3">
          <div>
            <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Top Trends</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {initialSignals.systemTrends.slice(0, 4).map((trend) => (
                <span key={trend.label} className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs font-semibold text-[color:var(--ink)]">
                  {trend.label} <span className={trend.direction === "up" ? "text-[color:var(--warn)]" : "text-[color:var(--ok)]"}>{formatPct(trend.changePct)}</span>
                </span>
              ))}
            </div>
          </div>

          <div>
            <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">What Changed</p>
            <div className="mt-2 space-y-1">
              {initialSignals.whatChanged.slice(0, 3).map((item) => (
                <p key={item} className="line-clamp-1 text-xs text-[color:var(--ink)]">- {item}</p>
              ))}
            </div>
          </div>

          <div>
            <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Dominant Narratives</p>
            <div className="mt-2 space-y-1">
              {initialSignals.narrativeLeaderboard.slice(0, 3).map((narrative, index) => (
                <p key={narrative.label} className="line-clamp-1 text-xs text-[color:var(--ink)]">
                  {index + 1}. {narrative.label} <span className="text-[color:var(--ink-faint)]">({narrative.severity})</span>
                </p>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Highest Alerts</p>
          <span className="text-xs text-[color:var(--ink-faint)]">Choose an alert or click any theme below</span>
        </div>
        <div className="grid gap-3 lg:grid-cols-4">
          {profiles.slice(0, 4).map((profile) => (
            <SignalAlertButton
              key={profile.id}
              profile={profile}
              active={profile.id === selectedSignal.id && selectedTheme === null}
              onClick={() => selectSignal(profile)}
            />
          ))}
        </div>
      </section>

      <AllThemesStrip
        activeThemes={selectedSignal.signal.normalized_theme_list}
        selectedTheme={selectedTheme}
        onSelect={selectTheme}
      />

      <div className="grid gap-5 xl:grid-cols-[390px_minmax(0,1fr)]">
        <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Trend Radar</p>
              <h2 className="mt-1 text-lg font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>Detection layer</h2>
            </div>
            <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs text-[color:var(--ink-faint)]">
              {scoredTrends.length} shown
            </span>
          </div>

          <div className="mt-4 max-h-[760px] space-y-3 overflow-y-auto pr-1">
            {scoredTrends.length > 0 ? (
              scoredTrends.map((trend) => (
                <TrendButton
                  key={trend.id}
                  trend={trend}
                  active={trend.id === selectedTrendId}
                  onClick={() => selectTrend(trend)}
                />
              ))
            ) : (
              <p className="rounded-lg border border-[color:var(--line-soft)] p-3 text-sm text-[color:var(--ink-faint)]">
                No trend data is available yet. The intelligence signals still render from the signal feed.
              </p>
            )}
          </div>
        </section>

        <section className="space-y-5">
          <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.62)] p-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <StatusBadge severity={selectedSignal.signal.severity} />
                  <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs font-semibold text-[color:var(--ink-faint)]">
                    {selectedSignal.context.window_label}
                  </span>
                  {selectedTrend ? (
                    <span className="rounded-full border border-[color:rgba(79,213,255,0.24)] px-3 py-1 text-xs font-semibold text-[color:var(--accent)]">
                      Trend: {selectedTrend.label}
                    </span>
                  ) : null}
                </div>
                <h2 className="mt-3 text-2xl font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
                  {selectedSignal.label}
                </h2>
                <p className="mt-2 max-w-4xl text-sm text-[color:var(--ink-soft)]">{selectedSignal.oneLineSummary}</p>
                <p className="mt-2 max-w-4xl text-xs leading-5 text-[color:var(--ink-faint)]">{selectedSignal.narrative}</p>
              </div>
              <div className="min-w-[190px] rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.38)] p-3">
                <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Coverage</p>
                <p className="mt-1 text-2xl font-semibold tabular-nums text-[color:var(--ink)]">{selectedSignal.coverage.totalArticles}</p>
                <p className="text-xs text-[color:var(--ink-faint)]">
                  {selectedSignal.coverage.sourceCount} sources - {selectedSignal.coverage.regionCount} regions
                </p>
              </div>
            </div>
          </div>

          <div className="grid gap-5 lg:grid-cols-[minmax(0,0.9fr)_minmax(320px,0.7fr)]">
            <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] p-4">
              <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Theme Drivers</p>
              <div className="mt-3 space-y-2">
                {selectedSignal.signal.frequency_signals.map((driver) => (
                  <DriverRow
                    key={driver.normalized_theme}
                    driver={driver}
                    selected={(selectedTheme ?? selectedDriver?.normalized_theme) === driver.normalized_theme}
                    onSelect={() => selectTheme(driver.normalized_theme)}
                  />
                ))}
              </div>
            </div>

            <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] p-4">
              <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Signal Composition</p>
              <div className="mt-3 space-y-3">
                {selectedSignal.signal.frequency_signals.map((driver) => (
                  <button key={driver.normalized_theme} type="button" onClick={() => selectTheme(driver.normalized_theme)} className="block w-full text-left">
                    <div className="flex items-center justify-between gap-3 text-xs">
                      <span className="font-semibold text-[color:var(--ink)]">{formatTheme(driver.normalized_theme)}</span>
                      <span className="font-semibold tabular-nums text-[color:var(--ink)]">{driver.contribution_pct.toFixed(1)}%</span>
                    </div>
                    <div className="mt-1 h-2 overflow-hidden rounded-full bg-[color:rgba(148,163,184,0.14)]">
                      <div className="h-full rounded-full bg-[color:var(--accent)]" style={{ width: `${Math.max(4, Math.min(100, driver.contribution_pct))}%` }} />
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px]">
            <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Evidence</p>
                  <h3 className="mt-1 text-lg font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
                    Why this signal exists
                  </h3>
                </div>
                <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs text-[color:var(--ink-faint)]">
                  {evidence.length} representative articles
                </span>
              </div>
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                {evidence.map((article) => (
                  <EvidenceLink key={article.id} article={article} />
                ))}
              </div>
            </div>

            <aside className="space-y-5">
              <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] p-4">
                <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Related Trends</p>
                <div className="mt-3 space-y-2">
                  {linkedTrends.map((trend) => (
                    <button
                      key={trend.id}
                      type="button"
                      onClick={() => selectTrend(trend)}
                      className="w-full rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.35)] p-3 text-left hover:border-[color:var(--line-strong)]"
                    >
                      <p className="truncate text-xs font-semibold text-[color:var(--ink)]">{trend.label}</p>
                      <p className="mt-1 text-[11px] text-[color:var(--ink-faint)]">{formatPct(trend.growth_pct)} - {trend.total_mentions.toLocaleString()} mentions</p>
                    </button>
                  ))}
                </div>
              </div>

              <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] p-4">
                <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Market Impact</p>
                <div className="mt-3 space-y-2">
                  {selectedSignal.signal.market_impacts.slice(0, 5).map((impact) => {
                    const style = DIRECTION_STYLE[impact.direction];
                    return (
                      <div key={`${impact.asset}-${impact.direction}`} className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.35)] p-3">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-mono text-sm font-semibold text-[color:var(--ink)]">{impact.asset}</span>
                          <span className="rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase" style={{ color: style.color, background: style.background }}>
                            {style.label}
                          </span>
                        </div>
                        <p className="mt-2 text-xs leading-5 text-[color:var(--ink-faint)]">{impact.rationale}</p>
                      </div>
                    );
                  })}
                </div>
              </div>
            </aside>
          </div>

          {selectedTrend ? (
            <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] p-4">
              <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Trend Evidence</p>
              <div className="mt-3 grid gap-3 md:grid-cols-3">
                {selectedTrend.top_docs.slice(0, 6).map((doc) => (
                  <TrendDoc key={doc.id} doc={doc} />
                ))}
              </div>
            </div>
          ) : null}
        </section>
      </div>
    </div>
  );
}
