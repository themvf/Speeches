"use client";

import { useEffect, useState } from "react";
import type {
  IntelligenceEvidenceArticle as EvidenceArticle,
  IntelligenceProfile,
  IntelligenceSignalsData,
  NarrativeRank,
  SystemTrend
} from "@/lib/intelligence-types";
import {
  THEME_MAPPING,
  type MarketImpactDirection,
  type NormalizedTheme,
  type SignalTrendDirection,
  type ThemeCategory,
  type ThemeFrequencySignal,
  type ThemeSeverity
} from "@/lib/theme-intelligence";

type ArticleSort = "recency" | "credibility" | "impact";
type EvidenceTab = "clusters" | "articles" | "impact";
type IntelligenceSignalsApiResponse =
  | { ok: true; data: IntelligenceSignalsData }
  | { ok: false; error: string; code: string };

const CATEGORY_LABELS: Record<ThemeCategory, string> = {
  MACRO: "Macro",
  FINANCIAL_SYSTEM: "Financial System",
  GEOPOLITICS: "Geopolitics",
  REAL_ECONOMY: "Real Economy",
  MODERN_THEMES: "Modern Themes"
};

const SEVERITY_STYLE: Record<ThemeSeverity, { color: string; background: string; border: string }> = {
  CRITICAL: { color: "#ff6b7f", background: "rgba(255,107,127,0.12)", border: "rgba(255,107,127,0.34)" },
  HIGH: { color: "var(--warn)", background: "rgba(242,171,67,0.12)", border: "rgba(242,171,67,0.32)" },
  NORMAL: { color: "var(--ok)", background: "rgba(65,211,157,0.1)", border: "rgba(65,211,157,0.28)" }
};

const TREND_STYLE: Record<SignalTrendDirection, { color: string; label: string }> = {
  ACCELERATING: { color: "var(--danger)", label: "Accelerating" },
  RISING: { color: "var(--warn)", label: "Rising" },
  STABLE: { color: "var(--accent)", label: "Stable" },
  FALLING: { color: "var(--ok)", label: "Cooling" }
};

const DIRECTION_STYLE: Record<MarketImpactDirection, { color: string; background: string; label: string }> = {
  UP: { color: "var(--ok)", background: "rgba(65,211,157,0.1)", label: "Up" },
  DOWN: { color: "var(--danger)", background: "rgba(255,107,127,0.1)", label: "Down" },
  MIXED: { color: "var(--warn)", background: "rgba(242,171,67,0.1)", label: "Mixed" }
};

function formatTheme(theme: string): string {
  return theme.replace(/_/g, " ");
}

function formatSigned(value: number): string {
  return value > 0 ? `+${value}` : String(value);
}

function formatPct(value: number): string {
  return `${formatSigned(value)}%`;
}

function formatContribution(value: number): string {
  return `${Number.isInteger(value) ? value : value.toFixed(1)}%`;
}

function sortArticles(articles: readonly EvidenceArticle[], sort: ArticleSort): EvidenceArticle[] {
  const minutes = (value: string) => Number.parseInt(value, 10) || 999;
  return [...articles].sort((a, b) => {
    if (sort === "credibility") return b.credibility - a.credibility || b.impact - a.impact;
    if (sort === "impact") return b.impact - a.impact || b.credibility - a.credibility;
    return minutes(a.timestamp) - minutes(b.timestamp);
  });
}

function expandEvidenceArticles(profile: IntelligenceProfile, count = 30, theme?: NormalizedTheme): EvidenceArticle[] {
  const themeArticles = theme ? profile.evidence.filter((article) => article.relatedThemes.includes(theme)) : profile.evidence;
  const sourceArticles = themeArticles.length > 0 ? themeArticles : profile.evidence;

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
      id: `${base.id}-expanded-${cycle}`,
      headline: `${base.headline} (${cluster?.title ?? "follow-up"} ${cycle + 1})`,
      timestamp: `${12 + index * 4} min ago`,
      excerpt: `${base.excerpt} Related coverage continues to reinforce the same theme cluster.`,
      impact: Math.max(1, base.impact - cycle * 2),
      credibility: Math.max(1, base.credibility - cycle)
    };
  });
}

function getThemeEvidence(theme: NormalizedTheme, profile: IntelligenceProfile) {
  const articles = profile.evidence.filter((article) => article.relatedThemes.includes(theme));
  const clusterIds = new Set(articles.map((article) => article.clusterId));
  const clusters = profile.clusters.filter((cluster) => clusterIds.has(cluster.id));

  return {
    articles,
    clusters
  };
}

function getProfileIdForTheme(theme: NormalizedTheme, profiles: readonly IntelligenceProfile[]): string {
  const directProfile = profiles.find((profile) => profile.signal.normalized_theme_list.includes(theme));
  if (directProfile) {
    return directProfile.id;
  }

  const definition = THEME_MAPPING.find((item) => item.normalized_theme === theme);
  if (!definition) {
    return profiles[0]?.id ?? "";
  }

  if (definition.category === "FINANCIAL_SYSTEM") {
    return "bank";
  }
  if (definition.category === "GEOPOLITICS") {
    return "geopolitical";
  }
  if (definition.category === "REAL_ECONOMY") {
    return theme === "SUPPLY_CHAIN" ? "geopolitical" : "macro";
  }
  if (definition.category === "MODERN_THEMES") {
    return "modern";
  }
  return "macro";
}

function getProfileForTheme(theme: NormalizedTheme, profiles: readonly IntelligenceProfile[]): IntelligenceProfile {
  const profileId = getProfileIdForTheme(theme, profiles);
  return profiles.find((profile) => profile.id === profileId) ?? profiles[0];
}

function getFirstClusterIdForTheme(theme: NormalizedTheme, profile: IntelligenceProfile): string {
  const evidence = getThemeEvidence(theme, profile);
  return evidence.articles[0]?.clusterId ?? evidence.clusters[0]?.id ?? "all";
}

function StatusBadge({ severity }: { severity: ThemeSeverity }) {
  const style = SEVERITY_STYLE[severity];
  return (
    <span className="inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold uppercase" style={{ color: style.color, background: style.background, border: `1px solid ${style.border}` }}>
      {severity}
    </span>
  );
}

const EVIDENCE_TABS: readonly { id: EvidenceTab; label: string }[] = [
  { id: "clusters", label: "Clusters" },
  { id: "articles", label: "Articles" },
  { id: "impact", label: "Impact" }
];

function SystemOverview({
  narrativeLeaderboard,
  systemTrends,
  whatChanged
}: {
  narrativeLeaderboard: readonly NarrativeRank[];
  systemTrends: readonly SystemTrend[];
  whatChanged: readonly string[];
}) {
  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.62)] p-4">
      <div className="grid gap-4 xl:grid-cols-3">
        <div>
          <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Top Trends</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {systemTrends.map((trend) => (
              <span
                key={trend.label}
                className="rounded-full border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.45)] px-3 py-1 text-xs font-semibold text-[color:var(--ink)]"
              >
                {trend.label}{" "}
                <span className="tabular-nums" style={{ color: trend.direction === "up" ? "var(--warn)" : "var(--ok)" }}>
                  {formatPct(trend.changePct)}
                </span>
              </span>
            ))}
          </div>
        </div>

        <div>
          <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">What Changed</p>
          <div className="mt-2 space-y-1">
            {whatChanged.slice(0, 3).map((item) => (
              <p key={item} className="line-clamp-1 text-xs text-[color:var(--ink)]">
                - {item}
              </p>
            ))}
          </div>
        </div>

        <div>
          <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Dominant Narratives</p>
          <div className="mt-2 space-y-1.5">
            {narrativeLeaderboard.slice(0, 3).map((item, index) => (
              <div key={item.label} className="flex items-center justify-between gap-3 text-xs">
                <span className="min-w-0 truncate font-semibold text-[color:var(--ink)]">
                  {index + 1}. {item.label}
                </span>
                <span className="shrink-0 rounded-full border border-[color:var(--line-soft)] px-2 py-0.5 text-[10px] font-semibold text-[color:var(--ink-faint)]">
                  {item.severity}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function AllThemesStrip({
  selectedTheme,
  activeThemes,
  onThemeSelect
}: {
  selectedTheme: NormalizedTheme | null;
  activeThemes: readonly NormalizedTheme[];
  onThemeSelect: (theme: NormalizedTheme) => void;
}) {
  const groupedThemes = THEME_MAPPING.reduce<Record<ThemeCategory, NormalizedTheme[]>>(
    (acc, item) => {
      acc[item.category].push(item.normalized_theme);
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
                  onClick={() => onThemeSelect(theme)}
                  aria-pressed={selectedTheme === theme}
                  className={`min-h-7 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold transition-colors ${
                    selectedTheme === theme
                      ? "border-[color:var(--line-strong)] bg-[color:rgba(79,213,255,0.18)] text-[color:var(--ink)]"
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

function SignalCard({
  profile,
  active,
  onClick
}: {
  profile: IntelligenceProfile;
  active: boolean;
  onClick: () => void;
}) {
  const signal = profile.signal;
  const trend = TREND_STYLE[signal.trend.direction];

  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`min-h-[112px] rounded-xl border p-4 text-left transition-colors ${
        active
          ? "border-[color:var(--line-strong)] bg-[color:rgba(79,213,255,0.13)]"
          : "border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] hover:border-[color:var(--line-strong)]"
      }`}
    >
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-semibold text-[color:var(--ink)]">{profile.label}</span>
        <StatusBadge severity={signal.severity} />
      </div>
      <p className="mt-2 line-clamp-2 text-xs text-[color:var(--ink-faint)]">{profile.oneLineSummary}</p>
      <div className="mt-2 flex items-center justify-between gap-3 text-xs">
        <span className="font-semibold" style={{ color: trend.color }}>
          {trend.label}
        </span>
        <span className="text-[color:var(--ink-faint)]">{formatSigned(signal.trend.delta)} vs prior window</span>
      </div>
    </button>
  );
}

function EvidenceItem({ article }: { article: EvidenceArticle }) {
  return (
    <article className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.48)] p-3">
      <h3 className="text-sm font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
        {article.url ? (
          <a href={article.url} target="_blank" rel="noreferrer" className="hover:text-[color:var(--accent)]">
            {article.headline}
          </a>
        ) : (
          article.headline
        )}
      </h3>
      <p className="mt-1 text-[11px] font-semibold uppercase text-[color:var(--ink-faint)]">
        {article.url ? (
          <a href={article.url} target="_blank" rel="noreferrer" className="hover:text-[color:var(--accent)]">
            {article.source}
          </a>
        ) : (
          article.source
        )}{" "}
        - {article.timestamp}
      </p>
      <p className="mt-2 text-xs text-[color:var(--ink-faint)]">{article.excerpt}</p>
      <p className="mt-2 text-xs font-medium text-[color:var(--accent)]">-&gt; {article.explanation}</p>
    </article>
  );
}

function ThemeDetailPanel({
  theme,
  profile,
  driver
}: {
  theme: NormalizedTheme;
  profile: IntelligenceProfile;
  driver: ThemeFrequencySignal | undefined;
}) {
  const definition = THEME_MAPPING.find((item) => item.normalized_theme === theme);
  const evidence = getThemeEvidence(theme, profile);
  const active = Boolean(driver);

  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] p-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Selected Theme</p>
          <h2 className="mt-1 text-xl font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
            {formatTheme(theme)}
          </h2>
          <p className="mt-1 text-xs text-[color:var(--ink-faint)]">
            {definition ? CATEGORY_LABELS[definition.category] : "Theme"} - Priority {definition?.weight ?? 0}/10
          </p>
        </div>
        <span
          className={`rounded-full border px-3 py-1 text-xs font-semibold ${
            active
              ? "border-[color:rgba(79,213,255,0.32)] bg-[color:rgba(79,213,255,0.1)] text-[color:var(--accent)]"
              : "border-[color:var(--line-soft)] text-[color:var(--ink-faint)]"
          }`}
        >
          {active ? `Active in ${profile.label}` : `Not active in ${profile.label}`}
        </span>
      </div>

      {driver ? (
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <div className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.38)] p-3">
            <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Mentions</p>
            <p className="mt-1 text-2xl font-semibold tabular-nums text-[color:var(--ink)]">{driver.current_mentions}</p>
          </div>
          <div className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.38)] p-3">
            <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Spike</p>
            <p className="mt-1 text-2xl font-semibold tabular-nums text-[color:var(--accent)]">{formatPct(driver.spike_pct)}</p>
          </div>
          <div className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.38)] p-3">
            <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Contribution</p>
            <p className="mt-1 text-2xl font-semibold tabular-nums text-[color:var(--ink)]">{formatContribution(driver.contribution_pct)}</p>
          </div>
        </div>
      ) : (
        <p className="mt-4 rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.35)] p-3 text-sm text-[color:var(--ink-faint)]">
          This theme is part of the full taxonomy, but it is not contributing to the selected signal. Choose a signal where it appears to inspect contribution and evidence.
        </p>
      )}

      {evidence.clusters.length > 0 || evidence.articles.length > 0 ? (
        <div className="mt-4 grid gap-4 lg:grid-cols-[0.85fr_1.15fr]">
          <div>
            <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Linked Clusters</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {evidence.clusters.map((cluster) => (
                <span key={cluster.id} className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs text-[color:var(--ink)]">
                  {cluster.title}
                </span>
              ))}
            </div>
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Supporting Evidence</p>
            <div className="mt-2 space-y-2">
              {evidence.articles.slice(0, 3).map((article) => (
                <div key={article.id} className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.35)] p-3">
                  <p className="text-sm font-semibold text-[color:var(--ink)]">
                    {article.url ? (
                      <a href={article.url} target="_blank" rel="noreferrer" className="hover:text-[color:var(--accent)]">
                        {article.headline}
                      </a>
                    ) : (
                      article.headline
                    )}
                  </p>
                  <p className="mt-1 text-[11px] uppercase text-[color:var(--ink-faint)]">
                    {article.url ? (
                      <a href={article.url} target="_blank" rel="noreferrer" className="hover:text-[color:var(--accent)]">
                        {article.source}
                      </a>
                    ) : (
                      article.source
                    )}{" "}
                    - {article.timestamp}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}

function CompactSignalPanel({
  profile,
  signal,
  trend,
  primaryDrivers,
  secondaryDrivers,
  selectedTheme,
  onThemeSelect,
  additionalDriversOpen,
  onToggleAdditionalDrivers
}: {
  profile: IntelligenceProfile;
  signal: IntelligenceProfile["signal"];
  trend: { color: string; label: string };
  primaryDrivers: readonly ThemeFrequencySignal[];
  secondaryDrivers: readonly ThemeFrequencySignal[];
  selectedTheme: NormalizedTheme | null;
  onThemeSelect: (theme: NormalizedTheme) => void;
  additionalDriversOpen: boolean;
  onToggleAdditionalDrivers: () => void;
}) {
  const visibleDrivers = additionalDriversOpen ? [...primaryDrivers, ...secondaryDrivers] : primaryDrivers;

  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.62)] p-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,0.72fr)_minmax(360px,0.28fr)]">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs font-semibold text-[color:var(--ink-faint)]">
              {profile.context.window_label}
            </span>
            <StatusBadge severity={signal.severity} />
            <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs font-semibold" style={{ color: trend.color }}>
              {trend.label}
            </span>
          </div>
          <h1 className="mt-3 text-2xl font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
            {profile.label}
          </h1>
          <p className="mt-2 max-w-4xl text-sm text-[color:var(--ink-soft)]">{profile.oneLineSummary}</p>
          <p className="mt-2 max-w-4xl text-xs leading-5 text-[color:var(--ink-faint)]">{profile.narrative}</p>
          <div className="mt-3 grid gap-2 md:grid-cols-3">
            {profile.whatChanged.slice(0, 3).map((item) => (
              <p key={item} className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.34)] p-2 text-xs leading-5 text-[color:var(--ink-faint)]">
                <span className="block text-[10px] font-semibold uppercase text-[color:var(--ink)]">What changed</span>
                {item}
              </p>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.42)] p-3">
          <div className="grid grid-cols-[minmax(0,1fr)_78px_70px_88px] gap-2 text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">
            <span>Drivers</span>
            <span>Mentions</span>
            <span>Spike</span>
            <span>Contribution</span>
          </div>
          <div className="mt-2 space-y-1">
            {visibleDrivers.map((driver) => (
              <button
                key={driver.normalized_theme}
                type="button"
                onClick={() => onThemeSelect(driver.normalized_theme)}
                aria-pressed={selectedTheme === driver.normalized_theme}
                className={`grid w-full grid-cols-[minmax(0,1fr)_78px_70px_88px] gap-2 rounded-md px-2 py-2 text-left text-xs transition-colors ${
                  selectedTheme === driver.normalized_theme
                    ? "border border-[color:var(--line-strong)] bg-[color:rgba(79,213,255,0.13)]"
                    : "border border-transparent hover:border-[color:var(--line-soft)] hover:bg-[color:rgba(79,213,255,0.06)]"
                }`}
              >
                <span className="truncate font-semibold text-[color:var(--ink)]">{formatTheme(driver.normalized_theme)}</span>
                <span className="tabular-nums text-[color:var(--ink-faint)]">{driver.current_mentions}</span>
                <span className="font-semibold tabular-nums text-[color:var(--accent)]">{formatPct(driver.spike_pct)}</span>
                <span className="font-semibold tabular-nums text-[color:var(--ink)]">{formatContribution(driver.contribution_pct)}</span>
              </button>
            ))}
          </div>
          {secondaryDrivers.length > 0 ? (
            <button
              type="button"
              onClick={onToggleAdditionalDrivers}
              aria-expanded={additionalDriversOpen}
              className="mt-2 min-h-8 rounded-lg border border-[color:var(--line-soft)] px-3 text-xs font-semibold text-[color:var(--accent)] hover:border-[color:var(--line-strong)]"
            >
              {additionalDriversOpen ? "Hide Additional Drivers" : `Show Additional Drivers (${secondaryDrivers.length})`}
            </button>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function EvidenceListSection({ articles, focusedTheme }: { articles: readonly EvidenceArticle[]; focusedTheme: NormalizedTheme | null }) {
  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Evidence</p>
          <h2 className="mt-1 text-xl font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
            {focusedTheme ? `Supporting articles for ${formatTheme(focusedTheme)}` : "Supporting articles"}
          </h2>
        </div>
        <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs text-[color:var(--ink-faint)]">{articles.length} articles shown</span>
      </div>
      <div className="mt-4 max-h-[560px] overflow-y-auto pr-1">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {articles.map((article) => (
            <EvidenceItem key={article.id} article={article} />
          ))}
        </div>
      </div>
    </section>
  );
}

function SignalCompositionPanel({
  drivers,
  profile,
  selectedTheme,
  onThemeSelect
}: {
  drivers: readonly ThemeFrequencySignal[];
  profile: IntelligenceProfile;
  selectedTheme: NormalizedTheme | null;
  onThemeSelect: (theme: NormalizedTheme) => void;
}) {
  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] p-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Signal Composition</p>
          <h2 className="mt-1 text-xl font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
            Ranked contribution by theme
          </h2>
        </div>
        <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs text-[color:var(--ink-faint)]">
          All {drivers.length} contributing themes shown
        </span>
      </div>

      <div className="mt-4 space-y-3">
        {drivers.map((driver) => {
          const evidence = getThemeEvidence(driver.normalized_theme, profile);
          const selected = selectedTheme === driver.normalized_theme;
          return (
            <button
              key={driver.normalized_theme}
              type="button"
              onClick={() => onThemeSelect(driver.normalized_theme)}
              aria-pressed={selected}
              className={`w-full rounded-lg border p-2 text-left transition-colors ${
                selected
                  ? "border-[color:var(--line-strong)] bg-[color:rgba(79,213,255,0.13)]"
                  : "border-transparent hover:border-[color:var(--line-soft)] hover:bg-[color:rgba(79,213,255,0.05)]"
              }`}
            >
              <div className="flex items-center justify-between gap-3 text-xs">
                <span className="font-semibold text-[color:var(--ink)]">{formatTheme(driver.normalized_theme)}</span>
                <span className="font-semibold tabular-nums text-[color:var(--ink)]">{formatContribution(driver.contribution_pct)}</span>
              </div>
              <div className="mt-1 h-2 overflow-hidden rounded-full bg-[color:rgba(148,163,184,0.14)]">
                <div className="h-full rounded-full bg-[color:var(--accent)]" style={{ width: `${Math.max(4, Math.min(100, driver.contribution_pct))}%` }} />
              </div>
              {evidence.articles[0] ? (
                <p className="mt-1 text-[11px] text-[color:var(--ink-faint)]">
                  Driven by {evidence.articles[0].headline}
                </p>
              ) : null}
            </button>
          );
        })}
      </div>

      <div className="mt-4 rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.32)] p-3">
        <p className="text-xs font-semibold text-[color:var(--ink)]">Showing top drivers based on:</p>
        <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-[color:var(--ink-faint)]">
          <span className="rounded-full border border-[color:var(--line-soft)] px-2 py-0.5">mention volume</span>
          <span className="rounded-full border border-[color:var(--line-soft)] px-2 py-0.5">spike vs baseline</span>
          <span className="rounded-full border border-[color:var(--line-soft)] px-2 py-0.5">contribution to signal</span>
        </div>
      </div>

      <div className="mt-4">
        <h3 className="text-sm font-semibold text-[color:var(--ink)]">Full Signal Composition</h3>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[560px] border-separate border-spacing-y-2 text-left text-xs">
            <thead className="text-[10px] uppercase text-[color:var(--ink-faint)]">
              <tr>
                <th className="px-3 font-semibold">Theme</th>
                <th className="px-3 font-semibold">Mentions</th>
                <th className="px-3 font-semibold">Spike</th>
                <th className="px-3 font-semibold">Contribution</th>
                <th className="px-3 font-semibold">Evidence</th>
              </tr>
            </thead>
            <tbody>
              {drivers.map((driver) => {
                const evidence = getThemeEvidence(driver.normalized_theme, profile);
                const selected = selectedTheme === driver.normalized_theme;
                return (
                  <tr
                    key={driver.normalized_theme}
                    className={`text-[color:var(--ink)] ${selected ? "bg-[color:rgba(79,213,255,0.13)]" : "bg-[color:rgba(6,15,24,0.38)]"}`}
                  >
                    <td className="rounded-l-lg border-y border-l border-[color:var(--line-soft)] px-3 py-2 font-semibold">
                      <button type="button" onClick={() => onThemeSelect(driver.normalized_theme)} className="text-left font-semibold text-[color:var(--ink)] hover:text-[color:var(--accent)]">
                        {formatTheme(driver.normalized_theme)}
                      </button>
                    </td>
                    <td className="border-y border-[color:var(--line-soft)] px-3 py-2 tabular-nums">{driver.current_mentions}</td>
                    <td className="border-y border-[color:var(--line-soft)] px-3 py-2 tabular-nums text-[color:var(--accent)]">{formatPct(driver.spike_pct)}</td>
                    <td className="border-y border-[color:var(--line-soft)] px-3 py-2 tabular-nums">{formatContribution(driver.contribution_pct)}</td>
                    <td className="rounded-r-lg border-y border-r border-[color:var(--line-soft)] px-3 py-2 text-[color:var(--ink-faint)]">
                      {evidence.clusters[0]?.title ?? evidence.articles[0]?.source ?? "Pending"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

export function ThemeIntelligenceLab({ initialData }: { initialData: IntelligenceSignalsData }) {
  const [data, setData] = useState<IntelligenceSignalsData>(initialData);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const profiles = data.profiles;
  const [profileId, setProfileId] = useState<string>(profiles[0]?.id ?? "");
  const [activeTab, setActiveTab] = useState<EvidenceTab>("clusters");
  const [selectedClusterId, setSelectedClusterId] = useState<string>("all");
  const [articleSort, setArticleSort] = useState<ArticleSort>("impact");
  const [additionalDriversOpen, setAdditionalDriversOpen] = useState(true);
  const [selectedTheme, setSelectedTheme] = useState<NormalizedTheme | null>(null);
  const profile = profiles.find((item) => item.id === profileId) ?? profiles[0];

  useEffect(() => {
    let cancelled = false;

    async function refreshSignals() {
      setIsRefreshing(true);
      try {
        const response = await fetch("/api/intelligence/signals", { cache: "no-store" });
        const payload = (await response.json()) as IntelligenceSignalsApiResponse;
        if (!response.ok || !payload.ok) {
          throw new Error(payload.ok ? `Request failed with ${response.status}` : payload.error);
        }
        if (!cancelled) {
          setData(payload.data);
          setLoadError(null);
        }
      } catch (error) {
        if (!cancelled) {
          setLoadError(error instanceof Error ? error.message : "Unable to refresh intelligence signals.");
        }
      } finally {
        if (!cancelled) {
          setIsRefreshing(false);
        }
      }
    }

    void refreshSignals();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (profiles.length > 0 && !profiles.some((item) => item.id === profileId)) {
      setProfileId(profiles[0].id);
      setSelectedTheme(null);
      setSelectedClusterId("all");
    }
  }, [profileId, profiles]);

  if (!profile) {
    return (
      <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] p-5">
        <p className="text-sm text-[color:var(--ink-faint)]">Loading intelligence signals...</p>
      </section>
    );
  }

  const signal = profile.signal;
  const trend = TREND_STYLE[signal.trend.direction];
  const selectedThemeForDisplay = selectedTheme ?? signal.primary_driver?.normalized_theme ?? null;
  const evidenceArticles = expandEvidenceArticles(profile, 30, selectedThemeForDisplay ?? undefined);
  const selectedArticles = selectedClusterId === "all" ? evidenceArticles : evidenceArticles.filter((article) => article.clusterId === selectedClusterId);
  const sortedArticles = sortArticles(selectedArticles, articleSort);
  const selectedCluster = profile.clusters.find((cluster) => cluster.id === selectedClusterId);
  const selectedVolume = selectedCluster?.volume ?? profile.coverage.totalArticles;
  const additionalSelectedArticles = Math.max(0, selectedVolume - selectedArticles.length);
  const primaryDrivers = signal.primary_drivers;
  const secondaryDrivers = signal.secondary_drivers;
  const selectedThemeDriver = selectedThemeForDisplay ? signal.frequency_signals.find((driver) => driver.normalized_theme === selectedThemeForDisplay) : undefined;
  const handleThemeSelect = (theme: NormalizedTheme) => {
    const nextProfile = getProfileForTheme(theme, profiles);
    setSelectedTheme(theme);
    setProfileId(nextProfile.id);
    setSelectedClusterId(getFirstClusterIdForTheme(theme, nextProfile));
    setActiveTab("clusters");
    setAdditionalDriversOpen(true);
  };

  return (
    <div className="space-y-6">
      <SystemOverview narrativeLeaderboard={data.narrativeLeaderboard} systemTrends={data.systemTrends} whatChanged={data.whatChanged} />
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-[color:var(--ink-faint)]">
        <span>Signals generated {new Date(data.generatedAt).toLocaleString()}</span>
        <span>{isRefreshing ? "Refreshing signal feed..." : loadError ? `Refresh issue: ${loadError}` : "API-backed signal feed"}</span>
      </div>

      <section>
        <div className="mb-3 flex items-center justify-between gap-3">
          <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Highest Alerts</p>
          <span className="text-xs text-[color:var(--ink-faint)]">Click an alert or choose a theme below</span>
        </div>
        <div className="grid gap-3 lg:grid-cols-4">
          {profiles.map((item) => (
            <SignalCard
              key={item.id}
              profile={item}
              active={item.id === profileId}
              onClick={() => {
                setProfileId(item.id);
                setSelectedTheme(null);
                setSelectedClusterId("all");
                setActiveTab("clusters");
                setAdditionalDriversOpen(true);
              }}
            />
          ))}
        </div>
      </section>

      <AllThemesStrip
        selectedTheme={selectedThemeForDisplay}
        activeThemes={signal.normalized_theme_list}
        onThemeSelect={handleThemeSelect}
      />

      {selectedThemeForDisplay ? <ThemeDetailPanel theme={selectedThemeForDisplay} profile={profile} driver={selectedThemeDriver} /> : null}

      <CompactSignalPanel
        profile={profile}
        signal={signal}
        trend={trend}
        primaryDrivers={primaryDrivers}
        secondaryDrivers={secondaryDrivers}
        selectedTheme={selectedThemeForDisplay}
        onThemeSelect={handleThemeSelect}
        additionalDriversOpen={additionalDriversOpen}
        onToggleAdditionalDrivers={() => setAdditionalDriversOpen((open) => !open)}
      />

      <EvidenceListSection articles={evidenceArticles} focusedTheme={selectedThemeForDisplay} />

      <SignalCompositionPanel
        drivers={signal.frequency_signals}
        profile={profile}
        selectedTheme={selectedThemeForDisplay}
        onThemeSelect={handleThemeSelect}
      />

      <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] p-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Signal Detail</p>
            <h2 className="mt-1 text-xl font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
              Clusters, articles, and impact
            </h2>
          </div>
          <div className="flex flex-wrap gap-2 text-xs text-[color:var(--ink-faint)]">
            <span
              title={`Coverage diversity: ${profile.coverage.diversity}. Dimensions: ${profile.coverage.dimensions.join(", ")}.`}
              className="rounded-full border border-[color:var(--line-soft)] px-3 py-1"
            >
              {profile.coverage.totalArticles} articles
            </span>
            <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1">{profile.coverage.sourceCount} sources</span>
            <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1">{profile.coverage.regionCount} regions</span>
            <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-[color:var(--accent)]">{formatPct(profile.coverage.changePct)}</span>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap gap-2 border-b border-[color:var(--line-soft)] pb-3" role="tablist" aria-label="Evidence detail views">
          {EVIDENCE_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`min-h-9 rounded-lg border px-3 text-sm font-semibold transition-colors ${
                activeTab === tab.id
                  ? "border-[color:var(--line-strong)] bg-[color:rgba(79,213,255,0.13)] text-[color:var(--ink)]"
                  : "border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.35)] text-[color:var(--ink-faint)] hover:border-[color:var(--line-strong)]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {activeTab === "clusters" ? (
          <div className="mt-4 grid gap-3 lg:grid-cols-2">
            {profile.clusters.map((cluster, index) => (
              <article key={cluster.id} className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.45)] p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">{cluster.importance}</p>
                    <h3 className="mt-1 text-sm font-semibold text-[color:var(--ink)]">
                      {index + 1}. {cluster.title}
                    </h3>
                  </div>
                  <span className="rounded-full border border-[color:rgba(242,171,67,0.24)] bg-[color:rgba(242,171,67,0.09)] px-2 py-0.5 text-xs font-semibold text-[color:var(--warn)]">
                    {formatPct(cluster.changePct)}
                  </span>
                </div>
                <p className="mt-2 text-xs leading-5 text-[color:var(--ink-soft)]">{cluster.summary}</p>
                <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-[color:var(--ink-faint)]">
                  <span>{cluster.volume} articles</span>
                  <span>First seen {cluster.firstSeen}</span>
                  <span>Peak {cluster.peakAcceleration}</span>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setSelectedClusterId(cluster.id);
                    setActiveTab("articles");
                  }}
                  className="mt-3 min-h-9 rounded-lg border border-[color:var(--line-soft)] px-3 text-xs font-semibold text-[color:var(--accent)] hover:border-[color:var(--line-strong)]"
                >
                  View Articles
                </button>
              </article>
            ))}
          </div>
        ) : null}

        {activeTab === "articles" ? (
          <div className="mt-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => setSelectedClusterId("all")}
                  aria-pressed={selectedClusterId === "all"}
                  className={`min-h-9 rounded-lg border px-3 text-xs font-semibold ${
                    selectedClusterId === "all"
                      ? "border-[color:var(--line-strong)] bg-[color:rgba(79,213,255,0.13)] text-[color:var(--ink)]"
                      : "border-[color:var(--line-soft)] text-[color:var(--ink-faint)]"
                  }`}
                >
                  All Coverage
                </button>
                {profile.clusters.map((cluster) => (
                  <button
                    key={cluster.id}
                    type="button"
                    onClick={() => setSelectedClusterId(cluster.id)}
                    aria-pressed={selectedClusterId === cluster.id}
                    className={`min-h-9 rounded-lg border px-3 text-xs font-semibold ${
                      selectedClusterId === cluster.id
                        ? "border-[color:var(--line-strong)] bg-[color:rgba(79,213,255,0.13)] text-[color:var(--ink)]"
                        : "border-[color:var(--line-soft)] text-[color:var(--ink-faint)]"
                    }`}
                  >
                    {cluster.title}
                  </button>
                ))}
              </div>
              <select aria-label="Sort articles" className="form-control min-h-10 px-3 text-sm" value={articleSort} onChange={(event) => setArticleSort(event.target.value as ArticleSort)}>
                <option value="impact">Sort by impact</option>
                <option value="credibility">Sort by source credibility</option>
                <option value="recency">Sort by recency</option>
              </select>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {profile.sourceDistribution.map((item) => (
                <span key={item.source} className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs text-[color:var(--ink-faint)]">
                  {item.source} ({item.count})
                </span>
              ))}
            </div>
            <p className="mt-3 text-xs text-[color:var(--ink-faint)]">
              Representative sample shown. {additionalSelectedArticles} more articles reinforce this {selectedCluster ? "cluster" : "signal"}.
            </p>
            <div className="mt-3 max-h-[520px] space-y-3 overflow-y-auto pr-1">
              {sortedArticles.map((article) => (
                <EvidenceItem key={article.id} article={article} />
              ))}
            </div>
          </div>
        ) : null}

        {activeTab === "impact" ? (
          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {signal.market_impacts.slice(0, 6).map((impact) => {
              const style = DIRECTION_STYLE[impact.direction];
              const linkedEvidence = profile.evidence.find((article) => article.relatedThemes.some((theme) => impact.themes.includes(theme)));
              return (
                <div key={`${impact.asset}-${impact.direction}`} className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.45)] p-3">
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-mono text-base font-semibold text-[color:var(--ink)]">{impact.asset}</span>
                    <span className="rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase" style={{ color: style.color, background: style.background }}>
                      {style.label}
                    </span>
                  </div>
                  <p className="mt-2 text-xs text-[color:var(--ink-faint)]">{impact.rationale}</p>
                  {linkedEvidence ? (
                    <p className="mt-3 border-t border-[color:var(--line-soft)] pt-2 text-xs text-[color:var(--accent)]">
                      Evidence: {linkedEvidence.source} - {linkedEvidence.explanation.toLowerCase()}
                    </p>
                  ) : null}
                </div>
              );
            })}
          </div>
        ) : null}
      </section>
    </div>
  );
}
