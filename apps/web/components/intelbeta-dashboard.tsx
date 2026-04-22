"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { SparklineChart } from "@/components/sparkline-chart";
import type {
  IntelligenceEvidenceArticle,
  IntelligenceEvidenceSource,
  IntelligenceProfile,
  IntelligenceSignalsData
} from "@/lib/intelligence-types";
import type { TrendItem, TrendsPayload } from "@/lib/server/types";
import {
  PRODUCT_CATEGORY_LABELS,
  PRODUCT_CATEGORY_ORDER,
  THEME_MAPPING,
  focusAreasForProductCategory,
  themesForProductCategory,
  type MarketImpactDirection,
  type NormalizedTheme,
  type ProductCategory,
  type ProductFocusArea,
  type ThemeFrequencySignal,
  type ThemeSeverity
} from "@/lib/theme-intelligence";

type ScoredTrend = TrendItem & {
  _score: number;
  _relatedSignalId: string;
};

const AML_EVIDENCE_KEY = "category:AML";
type CommandCategory = ProductCategory | "ALL";
type AmlFocusId = string;
type SeverityFilter = ThemeSeverity | "ALL";
type LiveEvidenceMeta = {
  source: IntelligenceEvidenceSource;
  coverage: {
    totalArticles: number;
    sourceCount: number;
  };
  sourceDistribution: { source: string; count: number }[];
};
type GdeltEvidenceApiResponse =
  | {
      ok: true;
      data: {
        profileId?: string;
        category?: string;
        label?: string;
        focusId?: string | null;
        source: IntelligenceEvidenceSource;
        evidence: IntelligenceEvidenceArticle[];
        coverage: {
          totalArticles: number;
          sourceCount: number;
        };
        sourceDistribution: { source: string; count: number }[];
      };
    }
  | { ok: false; error: string; code: string };

const SEVERITY_STYLE: Record<ThemeSeverity, { color: string; background: string; border: string }> = {
  CRITICAL: { color: "#ff595e", background: "rgba(255,89,94,0.12)", border: "rgba(255,89,94,0.52)" },
  HIGH: { color: "#ffbe3b", background: "rgba(255,190,59,0.11)", border: "rgba(255,190,59,0.42)" },
  NORMAL: { color: "#41d39d", background: "rgba(65,211,157,0.1)", border: "rgba(65,211,157,0.34)" }
};

const DIRECTION_STYLE: Record<MarketImpactDirection, { color: string; label: string }> = {
  UP: { color: "#41d39d", label: "UP" },
  DOWN: { color: "#ff595e", label: "DOWN" },
  MIXED: { color: "#ffbe3b", label: "MIXED" }
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

function formatTheme(theme: string): string {
  return theme.replace(/_/g, " ");
}

function formatPct(value: number): string {
  return `${value > 0 ? "+" : ""}${Number.isInteger(value) ? value : value.toFixed(1)}%`;
}

function formatCompactDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value || "Live";
  }
  return date.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short"
  });
}

function sourceLabel(kind: string): string {
  return SOURCE_KIND_LABELS[kind] ?? kind.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function articleLinkLabel(article: IntelligenceEvidenceArticle): string {
  return article.url ? "Open Source" : "Source URL Unavailable";
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
    profile.signal.product_categories.map((category) => `${category.label} ${category.subcategories.map((subcategory) => subcategory.label).join(" ")}`).join(" "),
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
  if (direct.length > 0) return direct.slice(0, 5);

  return [...trends]
    .sort((a, b) => trendSignalScore(b, profile) - trendSignalScore(a, profile))
    .slice(0, 5);
}

function evidenceForSignal(profile: IntelligenceProfile, theme?: NormalizedTheme): IntelligenceEvidenceArticle[] {
  const articles = theme ? profile.evidence.filter((article) => article.relatedThemes.includes(theme)) : profile.evidence;
  return articles.length > 0 ? articles : profile.evidence;
}

function evidenceForArticles(
  articles: readonly IntelligenceEvidenceArticle[],
  fallbackArticles: readonly IntelligenceEvidenceArticle[],
  theme?: NormalizedTheme
): IntelligenceEvidenceArticle[] {
  const themedArticles = theme ? articles.filter((article) => article.relatedThemes.includes(theme)) : articles;
  if (themedArticles.length > 0) {
    return [...themedArticles];
  }
  if (articles.length > 0) {
    return [...articles];
  }
  return [...fallbackArticles];
}

function sourceDistributionFromEvidence(articles: readonly IntelligenceEvidenceArticle[]) {
  const counts = new Map<string, number>();

  for (const article of articles) {
    counts.set(article.source, (counts.get(article.source) ?? 0) + 1);
  }

  return Array.from(counts, ([source, count]) => ({ source, count }))
    .sort((a, b) => b.count - a.count || a.source.localeCompare(b.source));
}

function profileForTheme(theme: NormalizedTheme, profiles: readonly IntelligenceProfile[], fallback: IntelligenceProfile): IntelligenceProfile {
  return profiles.find((profile) => profile.signal.normalized_theme_list.includes(theme)) ?? fallback;
}

function categoryForProfile(profile: IntelligenceProfile): ProductCategory {
  return profile.signal.primary_product_category?.category ?? "ECONOMIC_GROWTH";
}

function profileContainsCategory(profile: IntelligenceProfile, category: CommandCategory): boolean {
  return category === "ALL" || profile.signal.product_categories.some((item) => item.category === category);
}

function themeMatchesSearch(theme: NormalizedTheme, query: string): boolean {
  if (!query) return true;
  const definition = THEME_MAPPING.find((item) => item.normalized_theme === theme);
  const haystack = `${theme} ${definition?.raw_patterns.join(" ") ?? ""}`.toLowerCase();
  return haystack.includes(query.toLowerCase());
}

function buildCoverageBars(profile: IntelligenceProfile, trend?: ScoredTrend) {
  if (trend?.sparkline?.length) {
    return trend.sparkline.slice(-24).map((point) => ({ label: point.date.slice(5), count: point.count }));
  }

  return profile.signal.frequency_signals.slice(0, 18).map((driver) => ({
    label: formatTheme(driver.normalized_theme),
    count: driver.current_mentions
  }));
}

function CommandBadge({ severity }: { severity: ThemeSeverity }) {
  const style = SEVERITY_STYLE[severity];
  return (
    <span
      className="inline-flex items-center rounded-sm px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.18em]"
      style={{ color: style.color, background: style.background, border: `1px solid ${style.border}` }}
    >
      {severity}
    </span>
  );
}

function FilterButton({
  active,
  children,
  onClick
}: {
  active: boolean;
  children: ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`min-h-8 rounded-sm border px-3 text-[10px] font-bold uppercase tracking-[0.16em] transition-colors ${
        active
          ? "border-[#f2f2f2] bg-[#f2f2f2] text-[#0c0d12]"
          : "border-[#272b36] bg-[#11141c] text-[#a2a6b3] hover:border-[#515766] hover:text-[#f2f2f2]"
      }`}
    >
      {children}
    </button>
  );
}

function CoverageChart({ bars }: { bars: { label: string; count: number }[] }) {
  const maxCount = Math.max(...bars.map((bar) => bar.count), 1);

  return (
    <div className="rounded-sm border border-[#242832] bg-[#10131b] p-4">
      <div className="mb-5 flex items-center justify-between">
        <p className="text-[10px] font-bold uppercase tracking-[0.28em] text-[#777d8e]">Coverage Over Last 24h</p>
        <p className="text-[10px] font-semibold text-[#777d8e]">Peak {maxCount.toLocaleString()}/h</p>
      </div>
      <div className="flex h-24 items-end gap-1">
        {bars.map((bar, index) => (
          <div key={`${bar.label}-${index}`} className="flex min-w-0 flex-1 flex-col items-center justify-end gap-1">
            <div
              className="w-full rounded-t-[2px] bg-[#ff595e]"
              style={{ height: `${Math.max(8, Math.round((bar.count / maxCount) * 100))}%`, opacity: 0.35 + (bar.count / maxCount) * 0.65 }}
              title={`${bar.label}: ${bar.count} mentions`}
            />
            {index % 5 === 0 ? <span className="text-[9px] text-[#4f5564]">{bar.label}</span> : <span className="h-[11px]" />}
          </div>
        ))}
      </div>
    </div>
  );
}

function ThemeCommandStrip({
  category,
  activeThemes,
  focusAreas,
  selectedFocusId,
  selectedTheme,
  query,
  onSelectFocus,
  onSelect
}: {
  category: CommandCategory;
  activeThemes: readonly NormalizedTheme[];
  focusAreas: readonly ProductFocusArea[];
  selectedFocusId: AmlFocusId | null;
  selectedTheme: NormalizedTheme | null;
  query: string;
  onSelectFocus: (focusId: AmlFocusId) => void;
  onSelect: (theme: NormalizedTheme) => void;
}) {
  const activeSet = new Set(activeThemes);
  const visibleFocusAreas = category === "AML"
    ? focusAreas.filter((focusArea) => {
        if (!query) return true;
        const haystack = `${focusArea.label} ${focusArea.raw_patterns.join(" ")}`.toLowerCase();
        return haystack.includes(query.toLowerCase());
      })
    : [];
  const themes = (category === "ALL"
    ? THEME_MAPPING.map((item) => item.normalized_theme)
    : themesForProductCategory(category))
    .filter((theme) => themeMatchesSearch(theme, query));

  return (
    <div className="flex gap-2 overflow-x-auto border-t border-[#1d2029] px-5 py-3">
      {visibleFocusAreas.length > 0 ? visibleFocusAreas.map((focusArea) => (
        <button
          key={focusArea.id}
          type="button"
          onClick={() => onSelectFocus(focusArea.id)}
          aria-pressed={selectedFocusId === focusArea.id}
          className={`min-h-7 shrink-0 rounded-sm border px-2.5 text-[10px] font-bold uppercase tracking-[0.12em] ${
            selectedFocusId === focusArea.id
              ? "border-[#f2f2f2] bg-[#f2f2f2] text-[#0c0d12]"
              : "border-[#3a5066] bg-[#121926] text-[#b8daf1] hover:border-[#4fd5ff] hover:text-[#f2f2f2]"
          }`}
        >
          {focusArea.label}
        </button>
      )) : themes.map((theme) => (
        <button
          key={theme}
          type="button"
          onClick={() => onSelect(theme)}
          aria-pressed={selectedTheme === theme}
          className={`min-h-7 shrink-0 rounded-sm border px-2.5 text-[10px] font-bold uppercase tracking-[0.12em] ${
            selectedTheme === theme
              ? "border-[#f2f2f2] bg-[#f2f2f2] text-[#0c0d12]"
              : activeSet.has(theme)
                ? "border-[#3a5066] bg-[#121926] text-[#b8daf1]"
                : "border-[#272b36] bg-[#0d1017] text-[#7f8594] hover:border-[#4d5566] hover:text-[#f2f2f2]"
          }`}
        >
          {formatTheme(theme)}
        </button>
      ))}
    </div>
  );
}

function SignalRailItem({
  profile,
  trend,
  active,
  onClick
}: {
  profile: IntelligenceProfile;
  trend?: ScoredTrend;
  active: boolean;
  onClick: () => void;
}) {
  const severity = SEVERITY_STYLE[profile.signal.severity];
  const category = PRODUCT_CATEGORY_LABELS[categoryForProfile(profile)];

  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`group w-full border-l-2 px-4 py-3 text-left transition-colors ${
        active
          ? "border-l-[#ff595e] bg-[#12151e]"
          : "border-l-transparent bg-transparent hover:border-l-[#515766] hover:bg-[#10131a]"
      }`}
    >
      <div className="flex items-start gap-3">
        <span className="mt-1 h-2 w-2 shrink-0 rounded-full" style={{ background: severity.color }} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-2">
            <p className="line-clamp-1 text-xs font-bold text-[#f0f1f5]">{profile.label}</p>
            <span className="text-[10px] font-bold text-[#ffbe3b]">{formatPct(profile.signal.trend.delta_pct)}</span>
          </div>
          <p className="mt-1 text-[10px] uppercase tracking-[0.16em] text-[#666d7d]">
            {category} - {profile.coverage.totalArticles} articles
          </p>
          <div className="mt-2 flex items-center justify-between gap-2">
            <span className="text-[10px] text-[#747b8c]">{profile.context.window_label}</span>
            {trend ? <SparklineChart data={trend.sparkline} color={severity.color} /> : null}
          </div>
        </div>
      </div>
    </button>
  );
}

function DriverCard({
  driver,
  profile,
  evidenceArticles,
  index,
  onSelect
}: {
  driver: ThemeFrequencySignal;
  profile: IntelligenceProfile;
  evidenceArticles: readonly IntelligenceEvidenceArticle[];
  index: number;
  onSelect: () => void;
}) {
  const supporting = evidenceForArticles(evidenceArticles, profile.evidence, driver.normalized_theme)[0];

  return (
    <button
      type="button"
      onClick={onSelect}
      className="min-h-[74px] rounded-sm border border-[#272b36] bg-[#11141c] p-3 text-left transition-colors hover:border-[#ff595e]"
    >
      <p className="text-[9px] font-bold uppercase tracking-[0.18em] text-[#ff595e]">Driver {String(index + 1).padStart(2, "0")}</p>
      <p className="mt-2 line-clamp-2 text-xs font-semibold leading-5 text-[#f0f1f5]">{supporting?.explanation ?? `${formatTheme(driver.normalized_theme)} is contributing to the signal.`}</p>
      <div className="mt-2 flex items-center justify-between text-[10px] text-[#7b8190]">
        <span>{formatTheme(driver.normalized_theme)}</span>
        <span className="font-bold text-[#f0f1f5]">{driver.contribution_pct.toFixed(1)}%</span>
      </div>
    </button>
  );
}

function MarketImpactCard({ impact }: { impact: { asset: string; direction: MarketImpactDirection; rationale: string } }) {
  const style = DIRECTION_STYLE[impact.direction];

  return (
    <div className="rounded-sm border border-[#272b36] bg-[#11141c] p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-sm font-bold text-[#f0f1f5]">{impact.asset}</span>
        <span className="rounded-full border border-current px-2 py-0.5 text-[9px] font-bold" style={{ color: style.color }}>
          {style.label}
        </span>
      </div>
      <p className="mt-2 line-clamp-3 text-[11px] leading-4 text-[#9399a8]">{impact.rationale}</p>
    </div>
  );
}

function EvidenceStreamItem({
  article,
  index,
  active,
  onSelect
}: {
  article: IntelligenceEvidenceArticle;
  index: number;
  active: boolean;
  onSelect: () => void;
}) {
  const className = `block w-full border-l-2 px-4 py-3 text-left transition-colors ${
    active
      ? "border-l-[#4fd5ff] bg-[#121722]"
      : "border-l-transparent bg-transparent hover:border-l-[#515766] hover:bg-[#10131a]"
  }`;
  const content = (
    <>
      <div className="flex items-center justify-between gap-3 text-[10px] font-bold uppercase tracking-[0.12em] text-[#7a8190]">
        <span>{String(index + 1).padStart(2, "0")}</span>
        <span>{article.source}</span>
        <span className="ml-auto normal-case tracking-normal">{article.timestamp}</span>
      </div>
      <p className="mt-2 line-clamp-2 text-xs font-bold leading-5 text-[#f0f1f5] underline-offset-2 hover:underline">{article.headline}</p>
      <div className="mt-1 flex items-center justify-between gap-3">
        <p className="line-clamp-1 text-[11px] text-[#4fd5ff]">-&gt; {article.explanation}</p>
        <span className="shrink-0 text-[9px] font-bold uppercase tracking-[0.12em] text-[#7a8190]">{articleLinkLabel(article)}</span>
      </div>
    </>
  );

  if (article.url) {
    return (
      <a href={article.url} target="_blank" rel="noreferrer" onClick={onSelect} className={className}>
        {content}
      </a>
    );
  }

  return (
    <button type="button" onClick={onSelect} aria-pressed={active} className={className}>
      {content}
    </button>
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
  const [selectedAmlFocusId, setSelectedAmlFocusId] = useState<AmlFocusId | null>(null);
  const [selectedArticleId, setSelectedArticleId] = useState<string>("");
  const [categoryFilter, setCategoryFilter] = useState<CommandCategory>("ALL");
  const [severityFilter, setSeverityFilter] = useState<SeverityFilter>("ALL");
  const [searchQuery, setSearchQuery] = useState("");
  const [evidenceByProfile, setEvidenceByProfile] = useState<Record<string, IntelligenceEvidenceArticle[]>>({});
  const [evidenceMetaByProfile, setEvidenceMetaByProfile] = useState<Record<string, LiveEvidenceMeta>>({});
  const [loadingEvidenceProfileId, setLoadingEvidenceProfileId] = useState<string | null>(null);
  const [evidenceLoadError, setEvidenceLoadError] = useState<string | null>(null);
  const requestedEvidenceProfilesRef = useRef<Record<string, true>>({});

  const selectedSignal = profiles.find((profile) => profile.id === selectedSignalId) ?? profiles[0];
  const selectedTrend = scoredTrends.find((trend) => trend.id === selectedTrendId);
  const selectedDriver = selectedTheme
    ? selectedSignal?.signal.frequency_signals.find((driver) => driver.normalized_theme === selectedTheme)
    : selectedSignal?.signal.primary_driver;
  const isAmlFocusView = categoryFilter === "AML";
  const amlFocusAreas = useMemo(() => focusAreasForProductCategory("AML"), []);
  const selectedAmlFocus = selectedAmlFocusId ? amlFocusAreas.find((focusArea) => focusArea.id === selectedAmlFocusId) ?? null : null;
  const selectedProfileEvidence = selectedSignal ? evidenceByProfile[selectedSignal.id] ?? selectedSignal.evidence : [];
  const selectedEvidenceMeta = selectedSignal ? evidenceMetaByProfile[selectedSignal.id] : undefined;
  const allAmlEvidence = evidenceByProfile[AML_EVIDENCE_KEY] ?? [];
  const amlEvidenceMeta = evidenceMetaByProfile[AML_EVIDENCE_KEY];
  const amlMatchedEvidence = selectedAmlFocus
    ? allAmlEvidence.filter((article) => article.focusAreaId === selectedAmlFocus.id)
    : allAmlEvidence;
  const hasLiveEvidence = isAmlFocusView
    ? amlEvidenceMeta?.source === "gdelt-doc" || amlEvidenceMeta?.source === "gdelt-gkg" || amlEvidenceMeta?.source === "stored-news"
    : selectedEvidenceMeta?.source === "gdelt-doc" || selectedEvidenceMeta?.source === "gdelt-gkg" || selectedEvidenceMeta?.source === "stored-news";
  const evidence = isAmlFocusView
    ? amlMatchedEvidence.slice(0, 30)
    : selectedSignal
    ? evidenceForArticles(selectedProfileEvidence, selectedSignal.evidence, selectedTheme ?? selectedDriver?.normalized_theme).slice(0, 30)
    : [];
  const query = searchQuery.trim().toLowerCase();
  const streamEvidence = query
    ? evidence.filter((article) => `${article.headline} ${article.source} ${article.explanation} ${article.excerpt}`.toLowerCase().includes(query))
    : evidence;
  const selectedArticle = streamEvidence.find((article) => article.id === selectedArticleId) ?? streamEvidence[0] ?? evidence[0];
  const linkedTrends = selectedSignal ? relatedTrendsForSignal(scoredTrends, selectedSignal) : [];
  const relatedProfiles = profiles.filter((profile) => profile.id !== selectedSignal?.id).slice(0, 3);
  const amlFocusCards = amlFocusAreas.map((focusArea) => ({
    focusArea,
    count: allAmlEvidence.filter((article) => article.focusAreaId === focusArea.id).length
  }));
  const coverageBars = isAmlFocusView
    ? amlFocusCards.map((item) => ({ label: item.focusArea.label, count: item.count }))
    : selectedSignal ? buildCoverageBars(selectedSignal, selectedTrend) : [];
  const sourceDistribution = isAmlFocusView
    ? sourceDistributionFromEvidence(evidence)
    : selectedSignal ? (hasLiveEvidence && selectedEvidenceMeta ? selectedEvidenceMeta.sourceDistribution : selectedSignal.sourceDistribution) : [];
  const coverageArticleCount = isAmlFocusView
    ? evidence.length
    : selectedSignal ? (hasLiveEvidence && selectedEvidenceMeta ? selectedEvidenceMeta.coverage.totalArticles : selectedSignal.coverage.totalArticles) : 0;
  const coverageSourceCount = isAmlFocusView
    ? new Set(evidence.map((article) => article.source)).size
    : selectedSignal ? (hasLiveEvidence && selectedEvidenceMeta ? selectedEvidenceMeta.coverage.sourceCount : selectedSignal.coverage.sourceCount) : 0;
  const isLoadingEvidence = isAmlFocusView ? loadingEvidenceProfileId === AML_EVIDENCE_KEY : selectedSignal ? loadingEvidenceProfileId === selectedSignal.id : false;

  useEffect(() => {
    if (!selectedSignalId || requestedEvidenceProfilesRef.current[selectedSignalId]) {
      return;
    }

    let cancelled = false;
    requestedEvidenceProfilesRef.current[selectedSignalId] = true;
    setLoadingEvidenceProfileId(selectedSignalId);
    setEvidenceLoadError(null);

    async function loadLiveEvidence() {
      try {
        const response = await fetch(`/api/intelligence/gdelt-evidence?profileId=${encodeURIComponent(selectedSignalId)}`, {
          cache: "no-store"
        });
        const payload = (await response.json()) as GdeltEvidenceApiResponse;
        if (!response.ok || !payload.ok) {
          throw new Error(payload.ok ? `Request failed with ${response.status}` : payload.error);
        }

        if (!cancelled) {
          const evidenceKey = payload.data.profileId ?? selectedSignalId;
          setEvidenceByProfile((current) => ({ ...current, [evidenceKey]: payload.data.evidence }));
          setEvidenceMetaByProfile((current) => ({
            ...current,
            [evidenceKey]: {
              source: payload.data.source,
              coverage: payload.data.coverage,
              sourceDistribution: payload.data.sourceDistribution
            }
          }));
        }
      } catch (error) {
        if (!cancelled) {
          setEvidenceLoadError(error instanceof Error ? error.message : "Unable to load live GDELT evidence.");
        }
      } finally {
        if (!cancelled) {
          setLoadingEvidenceProfileId(null);
        }
      }
    }

    void loadLiveEvidence();

    return () => {
      cancelled = true;
    };
  }, [selectedSignalId]);

  useEffect(() => {
    if (categoryFilter !== "AML") {
      return;
    }

    if (requestedEvidenceProfilesRef.current[AML_EVIDENCE_KEY]) {
      return;
    }

    let cancelled = false;
    requestedEvidenceProfilesRef.current[AML_EVIDENCE_KEY] = true;
    setLoadingEvidenceProfileId(AML_EVIDENCE_KEY);
    setEvidenceLoadError(null);

    async function loadAmlEvidence() {
      try {
        const response = await fetch("/api/intelligence/gdelt-evidence?category=AML", {
          cache: "no-store"
        });
        const payload = (await response.json()) as GdeltEvidenceApiResponse;
        if (!response.ok || !payload.ok) {
          throw new Error(payload.ok ? `Request failed with ${response.status}` : payload.error);
        }

        if (!cancelled) {
          setEvidenceByProfile((current) => ({ ...current, [AML_EVIDENCE_KEY]: payload.data.evidence }));
          setEvidenceMetaByProfile((current) => ({
            ...current,
            [AML_EVIDENCE_KEY]: {
              source: payload.data.source,
              coverage: payload.data.coverage,
              sourceDistribution: payload.data.sourceDistribution
            }
          }));
        }
      } catch (error) {
        if (!cancelled) {
          setEvidenceLoadError(error instanceof Error ? error.message : "Unable to load AML evidence.");
        }
      } finally {
        if (!cancelled) {
          setLoadingEvidenceProfileId(null);
        }
      }
    }

    void loadAmlEvidence();

    return () => {
      cancelled = true;
    };
  }, [categoryFilter]);

  const categoryCounts = useMemo(() => {
    return PRODUCT_CATEGORY_ORDER.map((category) => ({
      category,
      count: profiles.filter((profile) => profileContainsCategory(profile, category)).length
    }));
  }, [profiles]);

  const visibleProfiles = profiles.filter((profile) => {
    const severityMatch = severityFilter === "ALL" || profile.signal.severity === severityFilter;
    const searchMatch = !query || signalHaystack(profile).includes(query);
    return severityMatch && searchMatch;
  });

  const keyDrivers = useMemo(() => {
    if (!selectedSignal) return [];
    const drivers = selectedSignal.signal.primary_drivers;
    if (!selectedDriver) return drivers.slice(0, 3);
    return [selectedDriver, ...drivers.filter((driver) => driver.normalized_theme !== selectedDriver.normalized_theme)].slice(0, 3);
  }, [selectedDriver, selectedSignal]);

  function selectCategory(category: CommandCategory) {
    setCategoryFilter(category);
    setSelectedTheme(null);
    setSelectedAmlFocusId(null);
    setSelectedArticleId("");
  }

  function selectTrend(trend: ScoredTrend) {
    setSelectedTrendId(trend.id);
    setSelectedSignalId(trend._relatedSignalId);
    setSelectedTheme(null);
    setSelectedAmlFocusId(null);
    setSelectedArticleId("");
  }

  function selectSignal(profile: IntelligenceProfile) {
    setSelectedSignalId(profile.id);
    setSelectedTheme(null);
    setSelectedAmlFocusId(null);
    setSelectedArticleId("");
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
    setSelectedAmlFocusId(null);
    setSelectedArticleId("");
    const nextTrend = relatedTrendsForSignal(scoredTrends, nextSignal)[0];
    if (nextTrend) {
      setSelectedTrendId(nextTrend.id);
    }
  }

  function selectAmlFocus(focusId: AmlFocusId) {
    setSelectedTheme(null);
    setSelectedAmlFocusId((current) => current === focusId ? null : focusId);
    setSelectedArticleId("");
  }

  if (!selectedSignal) {
    return (
      <section className="rounded-md border border-[#272b36] bg-[#090b10] p-5">
        <p className="text-sm text-[#9aa1af]">No intelligence signals are available yet.</p>
      </section>
    );
  }

  const criticalCount = profiles.filter((profile) => profile.signal.severity === "CRITICAL").length;
  const highCount = profiles.filter((profile) => profile.signal.severity === "HIGH").length;
  const sourceCount = profiles.reduce((sum, profile) => sum + profile.coverage.sourceCount, 0);
  const category = PRODUCT_CATEGORY_LABELS[categoryForProfile(selectedSignal)];
  const primarySubcategory = selectedSignal.signal.primary_product_category?.subcategories[0]?.label;
  const selectedProductCategories = selectedSignal.signal.product_categories.slice(0, 4);
  const deltaColor = selectedSignal.signal.trend.delta_pct >= 0 ? "#ff595e" : "#41d39d";
  const displayTitle = isAmlFocusView ? "AML" : selectedSignal.label;
  const displaySummary = isAmlFocusView
    ? "AML evidence is narrowed to sanctions, OFAC, AML/BSA, KYC, beneficial ownership, and illicit-finance indicators."
    : selectedSignal.oneLineSummary;
  const displayNarrative = isAmlFocusView
    ? "Active Signals remain unchanged. This category focus filters the evidence stream to explicit AML terms only, avoiding broad regulation, crypto, trade, or conflict matches."
    : selectedSignal.narrative;

  return (
    <div className="overflow-hidden rounded-md border border-[#252936] bg-[#080a10] text-[#f0f1f5] shadow-[0_24px_70px_rgba(0,0,0,0.55)]">
      <header className="flex flex-wrap items-center justify-between gap-4 border-b border-[#1d2029] px-5 py-3">
        <div className="flex min-w-0 items-center gap-3">
          <span className="inline-flex h-6 w-6 items-center justify-center rounded-sm bg-[#f2f2f2] text-[10px] font-black text-[#080a10]">IF</span>
          <div className="min-w-0">
            <h1 className="text-base font-bold text-[#f0f1f5]" style={{ letterSpacing: 0 }}>
              Intel Command
            </h1>
            <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[#666d7d]">
              {formatCompactDate(initialSignals.generatedAt)}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-5 text-xs">
          <span><b className="mr-1 text-lg text-[#ff595e]">{criticalCount}</b><span className="text-[10px] uppercase tracking-[0.12em] text-[#777d8e]">Critical</span></span>
          <span><b className="mr-1 text-lg text-[#ffbe3b]">{highCount}</b><span className="text-[10px] uppercase tracking-[0.12em] text-[#777d8e]">High</span></span>
          <span><b className="mr-1 text-lg text-[#f0f1f5]">{PRODUCT_CATEGORY_ORDER.length}</b><span className="text-[10px] uppercase tracking-[0.12em] text-[#777d8e]">Categories</span></span>
          <span><b className="mr-1 text-lg text-[#f0f1f5]">{sourceCount}</b><span className="text-[10px] uppercase tracking-[0.12em] text-[#777d8e]">Sources</span></span>
        </div>

        <div className="flex min-w-[280px] flex-1 justify-end gap-2">
          <input
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Query themes, tickers, narratives..."
            className="h-8 w-full max-w-[360px] rounded-sm border border-[#272b36] bg-[#0d1017] px-3 font-mono text-xs text-[#f0f1f5] placeholder:text-[#666d7d]"
          />
          <span className="hidden h-8 items-center rounded-sm border border-[#272b36] px-3 font-mono text-[10px] font-bold text-[#9aa1af] sm:inline-flex">CTRL K</span>
          <span className="inline-flex h-8 items-center rounded-sm bg-[#f2f2f2] px-4 text-[10px] font-black uppercase tracking-[0.18em] text-[#080a10]">Live</span>
        </div>
      </header>

      <section className="border-b border-[#1d2029] px-5 py-3">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex flex-wrap items-center gap-2">
            <span className="mr-1 text-[10px] font-bold uppercase tracking-[0.18em] text-[#777d8e]">Category</span>
            <FilterButton active={categoryFilter === "ALL"} onClick={() => selectCategory("ALL")}>
              All {profiles.length}
            </FilterButton>
            {categoryCounts.map((item) => (
              <FilterButton key={item.category} active={categoryFilter === item.category} onClick={() => selectCategory(item.category)}>
                {PRODUCT_CATEGORY_LABELS[item.category]} {item.count}
              </FilterButton>
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-2 border-l border-[#252936] pl-4">
            <span className="mr-1 text-[10px] font-bold uppercase tracking-[0.18em] text-[#777d8e]">Severity</span>
            {(["ALL", "CRITICAL", "HIGH", "NORMAL"] as SeverityFilter[]).map((severity) => (
              <FilterButton key={severity} active={severityFilter === severity} onClick={() => setSeverityFilter(severity)}>
                {severity}
              </FilterButton>
            ))}
          </div>
        </div>
      </section>

      <ThemeCommandStrip
        category={categoryFilter}
        activeThemes={selectedSignal.signal.normalized_theme_list}
        focusAreas={amlFocusAreas}
        selectedFocusId={selectedAmlFocusId}
        selectedTheme={selectedTheme}
        query={query}
        onSelectFocus={selectAmlFocus}
        onSelect={selectTheme}
      />

      <div className="grid min-h-[740px] grid-cols-1 xl:grid-cols-[280px_minmax(0,1fr)_340px]">
        <aside className="border-r border-[#1d2029] bg-[#090b10]">
          <div className="border-b border-[#1d2029] px-4 py-4">
            <div className="flex items-center justify-between">
              <p className="text-[10px] font-bold uppercase tracking-[0.22em] text-[#777d8e]">Active Signals</p>
              <span className="text-[10px] text-[#666d7d]">{visibleProfiles.length}</span>
            </div>
            <div className="mt-3 flex gap-2">
              {(["SEV", "MOM", "COV", "NEW"] as const).map((label) => (
                <span key={label} className="rounded-sm border border-[#272b36] bg-[#11141c] px-2 py-1 text-[9px] font-bold uppercase tracking-[0.12em] text-[#a2a6b3]">
                  {label}
                </span>
              ))}
            </div>
          </div>

          <div className="max-h-[660px] overflow-y-auto">
            {PRODUCT_CATEGORY_ORDER.map((group) => {
              const groupProfiles = visibleProfiles.filter((profile) => categoryForProfile(profile) === group);
              if (groupProfiles.length === 0) return null;

              return (
                <section key={group} className="border-b border-[#171a22] py-2">
                  <div className="flex items-center justify-between px-4 py-2">
                    <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#777d8e]">{PRODUCT_CATEGORY_LABELS[group]}</p>
                    <span className="text-[10px] text-[#555c6d]">{groupProfiles.length}</span>
                  </div>
                  {groupProfiles.map((profile) => (
                    <SignalRailItem
                      key={profile.id}
                      profile={profile}
                      trend={relatedTrendsForSignal(scoredTrends, profile)[0]}
                      active={profile.id === selectedSignal.id && selectedTheme === null}
                      onClick={() => selectSignal(profile)}
                    />
                  ))}
                </section>
              );
            })}

            {scoredTrends.length > 0 ? (
              <section className="py-3">
                <div className="px-4 pb-2">
                  <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#777d8e]">Fastest Trends</p>
                </div>
                {scoredTrends.slice(0, 6).map((trend) => (
                  <button
                    key={trend.id}
                    type="button"
                    onClick={() => selectTrend(trend)}
                    className={`w-full border-l-2 px-4 py-3 text-left transition-colors ${
                      selectedTrend?.id === trend.id
                        ? "border-l-[#ffbe3b] bg-[#12151e]"
                        : "border-l-transparent hover:border-l-[#515766] hover:bg-[#10131a]"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <p className="line-clamp-1 text-xs font-bold text-[#f0f1f5]">{trend.label}</p>
                      <span className="text-[10px] font-bold text-[#ffbe3b]">{formatPct(trend.growth_pct)}</span>
                    </div>
                    <p className="mt-1 text-[10px] text-[#666d7d]">{trend.total_mentions.toLocaleString()} mentions</p>
                  </button>
                ))}
              </section>
            ) : null}
          </div>
        </aside>

        <main className="bg-[#0b0d13] px-6 py-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                {isAmlFocusView ? (
                  <span className="inline-flex items-center rounded-sm border border-[#3a5066] bg-[#121926] px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.18em] text-[#b8daf1]">
                    Category Focus
                  </span>
                ) : (
                  <CommandBadge severity={selectedSignal.signal.severity} />
                )}
                <span className="rounded-sm border border-[#5a2530] bg-[#231117] px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.18em] text-[#ff595e]">
                  {isAmlFocusView ? (selectedAmlFocus?.label ?? "All AML") : formatTheme(selectedSignal.signal.trend.direction)}
                </span>
                <span className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#777d8e]">
                  {isAmlFocusView
                    ? `AML evidence focus - ${streamEvidence.length} matched articles`
                    : `SIG-${selectedSignal.signal.primary_driver?.rank ?? 1} - ${category}${primarySubcategory ? ` / ${primarySubcategory}` : ""} - first seen ${formatCompactDate(initialSignals.generatedAt)}`}
                </span>
              </div>
              <h2 className="mt-3 text-3xl font-bold leading-tight text-[#f2f2f2]" style={{ letterSpacing: 0 }}>
                {displayTitle}
              </h2>
              <p className="mt-3 max-w-3xl font-serif text-lg italic leading-7 text-[#d6d7dc]">
                {displaySummary}
              </p>
              <p className="mt-2 max-w-3xl text-sm leading-6 text-[#a1a7b5]">{displayNarrative}</p>
            </div>

            <div className="text-right">
              <p className="text-4xl font-black tabular-nums" style={{ color: isAmlFocusView ? "#4fd5ff" : deltaColor }}>
                {isAmlFocusView ? streamEvidence.length : formatPct(selectedSignal.signal.trend.delta_pct)}
              </p>
              <p className="mt-1 text-[10px] font-bold uppercase tracking-[0.18em] text-[#777d8e]">
                {isAmlFocusView ? "matched evidence" : "vs prior window"}
              </p>
            </div>
          </div>

          <div className="mt-7 grid gap-5 lg:grid-cols-[minmax(0,1fr)_255px]">
            <CoverageChart bars={coverageBars} />

            <div className="rounded-sm border border-[#242832] bg-[#10131b] p-4">
              <p className="text-[10px] font-bold uppercase tracking-[0.28em] text-[#777d8e]">Coverage Footprint</p>
              <div className="mt-5 grid grid-cols-2 gap-5">
                <div>
                  <p className="text-2xl font-black tabular-nums text-[#f2f2f2]">{coverageArticleCount}</p>
                  <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#777d8e]">Articles</p>
                </div>
                <div>
                  <p className="text-2xl font-black tabular-nums text-[#f2f2f2]">{coverageSourceCount}</p>
                  <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#777d8e]">Sources</p>
                </div>
                <div>
                  <p className="text-2xl font-black tabular-nums text-[#f2f2f2]">{selectedSignal.coverage.regionCount}</p>
                  <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#777d8e]">Regions</p>
                </div>
                <div>
                  <p className="text-2xl font-black tabular-nums text-[#f2f2f2]">{streamEvidence.length}</p>
                  <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#777d8e]">Evidence</p>
                </div>
              </div>
            </div>
          </div>

          <section className="mt-6">
            <p className="text-[10px] font-bold uppercase tracking-[0.28em] text-[#777d8e]">{isAmlFocusView ? "AML Focus Areas" : "Key Drivers"}</p>
            <div className="mt-3 grid gap-3 lg:grid-cols-3">
              {isAmlFocusView ? amlFocusCards.map((item, index) => (
                <button
                  key={item.focusArea.id}
                  type="button"
                  onClick={() => selectAmlFocus(item.focusArea.id)}
                  className={`min-h-[74px] rounded-sm border p-3 text-left transition-colors ${
                    selectedAmlFocusId === item.focusArea.id
                      ? "border-[#f2f2f2] bg-[#f2f2f2] text-[#0c0d12]"
                      : "border-[#272b36] bg-[#11141c] hover:border-[#4fd5ff]"
                  }`}
                >
                  <p className={`text-[9px] font-bold uppercase tracking-[0.18em] ${selectedAmlFocusId === item.focusArea.id ? "text-[#0c0d12]" : "text-[#4fd5ff]"}`}>
                    Focus {String(index + 1).padStart(2, "0")}
                  </p>
                  <p className={`mt-2 text-xs font-semibold leading-5 ${selectedAmlFocusId === item.focusArea.id ? "text-[#0c0d12]" : "text-[#f0f1f5]"}`}>{item.focusArea.label}</p>
                  <div className={`mt-2 flex items-center justify-between text-[10px] ${selectedAmlFocusId === item.focusArea.id ? "text-[#333741]" : "text-[#7b8190]"}`}>
                    <span>{item.focusArea.raw_patterns.slice(0, 2).join(", ")}</span>
                    <span className="font-bold tabular-nums">{item.count}</span>
                  </div>
                </button>
              )) : keyDrivers.map((driver, index) => (
                <DriverCard
                  key={driver.normalized_theme}
                  driver={driver}
                  profile={selectedSignal}
                  evidenceArticles={selectedProfileEvidence}
                  index={index}
                  onSelect={() => selectTheme(driver.normalized_theme)}
                />
              ))}
            </div>
          </section>

          <section className="mt-6">
            <div className="mb-3 flex items-center justify-between">
              <p className="text-[10px] font-bold uppercase tracking-[0.28em] text-[#777d8e]">{isAmlFocusView ? "AML Match Terms" : "Category Map"}</p>
              <span className="text-[10px] font-semibold text-[#777d8e]">
                {isAmlFocusView ? `${amlFocusAreas.reduce((sum, item) => sum + item.raw_patterns.length, 0)} strict terms` : `${selectedSignal.signal.product_categories.length} active categories`}
              </span>
            </div>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              {isAmlFocusView ? amlFocusAreas.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => selectAmlFocus(item.id)}
                  className="min-h-[96px] rounded-sm border border-[#272b36] bg-[#11141c] p-3 text-left transition-colors hover:border-[#4fd5ff]"
                >
                  <div className="flex items-start justify-between gap-3">
                    <p className="text-xs font-bold text-[#f0f1f5]">{item.label}</p>
                    <span className="font-mono text-xs font-bold text-[#4fd5ff]">{amlFocusCards.find((card) => card.focusArea.id === item.id)?.count ?? 0}</span>
                  </div>
                  <p className="mt-3 text-[10px] leading-4 text-[#8d94a5]">
                    {item.raw_patterns.join(", ")}
                  </p>
                </button>
              )) : selectedProductCategories.map((item) => (
                <button
                  key={item.category}
                  type="button"
                  onClick={() => selectCategory(item.category)}
                  className="min-h-[96px] rounded-sm border border-[#272b36] bg-[#11141c] p-3 text-left transition-colors hover:border-[#4fd5ff]"
                >
                  <div className="flex items-start justify-between gap-3">
                    <p className="text-xs font-bold text-[#f0f1f5]">{item.label}</p>
                    <span className="font-mono text-xs font-bold text-[#4fd5ff]">{item.contribution_pct.toFixed(1)}%</span>
                  </div>
                  <div className="mt-3 space-y-1">
                    {item.subcategories.slice(0, 2).map((subcategory) => (
                      <div key={subcategory.label} className="flex items-center justify-between gap-2 text-[10px] text-[#8d94a5]">
                        <span className="line-clamp-1">{subcategory.label}</span>
                        <span className="font-mono text-[#c8ccd5]">{subcategory.contribution_pct.toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                </button>
              ))}
            </div>
          </section>

          <section className="mt-6">
            <div className="mb-3 flex items-center justify-between">
              <p className="text-[10px] font-bold uppercase tracking-[0.28em] text-[#777d8e]">Market Read-Through</p>
              <span className="text-[10px] font-semibold text-[#777d8e]">{selectedSignal.signal.market_impacts.length} instruments</span>
            </div>
            <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-5">
              {selectedSignal.signal.market_impacts.slice(0, 5).map((impact) => (
                <MarketImpactCard key={`${impact.asset}-${impact.direction}`} impact={impact} />
              ))}
            </div>
          </section>

          <section className="mt-6 rounded-sm border border-[#272b36] bg-[#10131b] p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="text-[10px] font-bold uppercase tracking-[0.28em] text-[#777d8e]">
                  Promoted Evidence {selectedArticle ? "#01" : ""}
                </p>
                {selectedArticle ? (
                  <>
                    <h3 className="mt-3 text-lg font-bold text-[#f2f2f2]" style={{ letterSpacing: 0 }}>
                      {selectedArticle.headline}
                    </h3>
                    <p className="mt-2 text-sm text-[#4fd5ff]">-&gt; {selectedArticle.explanation}</p>
                    <p className="mt-3 max-w-4xl text-sm leading-6 text-[#9aa1af]">{selectedArticle.excerpt}</p>
                  </>
                ) : (
                  <p className="mt-3 text-sm text-[#9aa1af]">No evidence is available for this signal.</p>
                )}
              </div>
              {selectedArticle?.url ? (
                <a
                  href={selectedArticle.url}
                  target="_blank"
                  rel="noreferrer"
                  className="rounded-sm border border-[#272b36] px-3 py-2 text-[10px] font-bold uppercase tracking-[0.14em] text-[#f2f2f2] hover:border-[#4fd5ff]"
                >
                  {articleLinkLabel(selectedArticle)}
                </a>
              ) : (
                <span className="rounded-sm border border-[#272b36] px-3 py-2 text-[10px] font-bold uppercase tracking-[0.14em] text-[#777d8e]">
                  Source URL Unavailable
                </span>
              )}
            </div>
          </section>

          <section className="mt-6 rounded-sm border border-[#272b36] bg-[#10131b] p-4">
            <div className="mb-4 flex items-center justify-between">
              <p className="text-[10px] font-bold uppercase tracking-[0.28em] text-[#777d8e]">Signal Composition</p>
              <span className="text-[10px] text-[#777d8e]">{selectedSignal.signal.frequency_signals.length} contributing themes</span>
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              {selectedSignal.signal.frequency_signals.map((driver) => (
                <button key={driver.normalized_theme} type="button" onClick={() => selectTheme(driver.normalized_theme)} className="text-left">
                  <div className="flex items-center justify-between gap-3 text-[11px]">
                    <span className="font-bold uppercase tracking-[0.08em] text-[#f0f1f5]">{formatTheme(driver.normalized_theme)}</span>
                    <span className="font-bold tabular-nums text-[#f0f1f5]">{driver.contribution_pct.toFixed(1)}%</span>
                  </div>
                  <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-[#20242d]">
                    <div className="h-full rounded-full bg-[#ff595e]" style={{ width: `${Math.max(4, Math.min(100, driver.contribution_pct))}%` }} />
                  </div>
                </button>
              ))}
            </div>
          </section>
        </main>

        <aside className="border-l border-[#1d2029] bg-[#090b10]">
          <div className="border-b border-[#1d2029] px-4 py-4">
            <div className="flex items-center justify-between">
              <p className="text-[10px] font-bold uppercase tracking-[0.28em] text-[#777d8e]">Evidence Stream</p>
              <span className="text-[10px] text-[#777d8e]">{streamEvidence.length} of {evidence.length}</span>
            </div>
            <p className="mt-2 text-[11px] leading-4 text-[#8b91a0]">
              {isLoadingEvidence
                ? isAmlFocusView ? "Loading AML evidence across live signals..." : "Loading live GDELT article links..."
                : isAmlFocusView
                  ? hasLiveEvidence
                    ? amlEvidenceMeta?.source === "stored-news"
                      ? "Stored NewsAPI articles matched to strict AML terms."
                      : "Live AML evidence using strict sanctions, OFAC, AML/BSA, KYC, ownership, and illicit-finance terms."
                    : "No strict AML evidence matched yet."
                  : hasLiveEvidence
                  ? selectedEvidenceMeta?.source === "stored-news"
                    ? "Stored NewsAPI articles from the ingested corpus."
                    : selectedEvidenceMeta?.source === "gdelt-gkg"
                    ? "Live GDELT GKG articles with source URLs."
                    : "Live GDELT DOC 2.0 articles with source URLs."
                  : evidenceLoadError
                    ? `Seed evidence shown; GDELT issue: ${evidenceLoadError}`
                    : "Seed evidence shown until live GDELT links load."}
            </p>
          </div>

          <div className="max-h-[560px] overflow-y-auto">
            {streamEvidence.map((article, index) => (
              <EvidenceStreamItem
                key={article.id}
                article={article}
                index={index}
                active={selectedArticle?.id === article.id}
                onSelect={() => setSelectedArticleId(article.id)}
              />
            ))}
          </div>

          <section className="border-t border-[#1d2029] px-4 py-4">
            <p className="text-[10px] font-bold uppercase tracking-[0.28em] text-[#777d8e]">Related Signals</p>
            <div className="mt-3 space-y-2">
              {relatedProfiles.map((profile) => (
                <button
                  key={profile.id}
                  type="button"
                  onClick={() => selectSignal(profile)}
                  className="w-full rounded-sm border border-[#171a22] bg-[#10131b] px-3 py-2 text-left text-xs font-semibold text-[#f0f1f5] hover:border-[#4fd5ff]"
                >
                  {profile.label}
                </button>
              ))}
              {linkedTrends.slice(0, 3).map((trend) => (
                <button
                  key={trend.id}
                  type="button"
                  onClick={() => selectTrend(trend)}
                  className="w-full rounded-sm border border-[#171a22] bg-[#10131b] px-3 py-2 text-left text-xs font-semibold text-[#f0f1f5] hover:border-[#ffbe3b]"
                >
                  {trend.label}
                </button>
              ))}
            </div>
          </section>

          <section className="border-t border-[#1d2029] px-4 py-4">
            <p className="text-[10px] font-bold uppercase tracking-[0.28em] text-[#777d8e]">Top Sources</p>
            <div className="mt-3 space-y-2">
              {sourceDistribution.slice(0, 5).map((source) => (
                <div key={source.source} className="flex items-center justify-between gap-3 text-xs">
                  <span className="font-semibold text-[#d8dbe3]">{sourceLabel(source.source)}</span>
                  <span className="font-bold tabular-nums text-[#777d8e]">{source.count}</span>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
