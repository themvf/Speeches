"use client";

import { useEffect, useMemo, useState } from "react";
import type { TrendDocItem, TrendItem, TrendSparklinePoint, TrendsPayload } from "@/lib/server/types";

interface ApiEnvelope<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

type CategoryFilter =
  | "all"
  | "crypto"
  | "ai_tech"
  | "enforcement"
  | "aml"
  | "capital_formation"
  | "securities_regulation";

type RangeFilter = "30d" | "7d" | "90d";

interface CategoryDef {
  value: CategoryFilter;
  label: string;
  keywords: string[];
}

type ScoredTrend = TrendItem & { _score: number };

/* ── Category definitions ─────────────────────────────────────────────── */

const CATEGORIES: CategoryDef[] = [
  { value: "all", label: "All", keywords: [] },
  {
    value: "crypto",
    label: "Crypto Assets",
    keywords: [
      "crypto", "bitcoin", "ethereum", "blockchain", "token", "stablecoin",
      "defi", "digital asset", "digital-asset", "nft", "web3", "cryptocurrency",
      "crypto-asset", "crypto asset", "digital_asset",
    ],
  },
  {
    value: "ai_tech",
    label: "AI & Tech",
    keywords: [
      "artificial intelligence", "machine learning", "ai", "technology", "fintech",
      "cybersecurity", "cyber", "innovation", "algorithm", "automated", "automation",
      "predictive", "large language model", "llm",
    ],
  },
  {
    value: "enforcement",
    label: "Enforcement",
    keywords: [
      "enforcement", "litigation", "fraud", "sanction", "penalty", "settlement",
      "complaint", "violation", "misconduct", "fine", "cease-and-desist", "awc",
      "disciplinary", "conviction", "prosecution", "insider trading", "market manipulation",
    ],
  },
  {
    value: "aml",
    label: "AML",
    keywords: [
      "anti-money laundering", "aml", "bank secrecy", "bsa", "financial crime",
      "terrorist financing", "suspicious activity", "kyc", "know your customer",
      "fatf", "money laundering", "illicit finance", "sanctions",
    ],
  },
  {
    value: "capital_formation",
    label: "Capital Formation",
    keywords: [
      "capital formation", "capital-formation", "ipo", "spac", "private offering",
      "regulation a", "regulation crowdfunding", "small business", "emerging growth",
      "startup", "venture", "private market", "private-market", "fundraising",
      "public offering", "exempt offering",
    ],
  },
  {
    value: "securities_regulation",
    label: "Securities Regulation",
    keywords: [
      "securities regulation", "securities-regulation", "disclosure", "rulemaking",
      "regulation nms", "market structure", "equity market", "best execution",
      "broker-dealer", "investment adviser", "fiduciary", "suitability", "proxy",
      "shareholder", "corporate governance", "form pf", "edgar", "reporting",
      "registration", "investment company",
    ],
  },
];

const RANGE_FILTERS: { value: RangeFilter; label: string }[] = [
  { value: "7d", label: "Last 7 days" },
  { value: "30d", label: "Last 30 days" },
  { value: "90d", label: "Last 90 days" },
];

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
  uploaded: "Uploaded",
};

/* ── Helpers ──────────────────────────────────────────────────────────── */

function sourceLabel(kind: string): string {
  return SOURCE_KIND_LABELS[kind] ?? kind.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function fmtDate(value: string): string {
  if (!value) return "-";
  const d = new Date(`${value}T00:00:00Z`);
  return Number.isNaN(d.getTime())
    ? value
    : d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
}

function matchesCategory(trend: TrendItem, cat: CategoryDef): boolean {
  if (cat.value === "all") return true;
  const haystack = [trend.canonical_tag, ...trend.cluster_tags].join(" ").toLowerCase();
  return cat.keywords.some((kw) => haystack.includes(kw));
}

function applyCategoryFilter(trends: TrendItem[], filter: CategoryFilter): TrendItem[] {
  if (filter === "all") return trends;
  const def = CATEGORIES.find((c) => c.value === filter);
  if (!def) return trends;
  return trends.filter((t) => matchesCategory(t, def));
}

/**
 * Composite score 0–100 combining growth, volume, and momentum velocity.
 * Inputs:
 *   growth_pct   — capped at [-100, 300], weight 0.40
 *   total_mentions — log-scaled to [0,1] assuming 10 000 max, weight 0.35
 *   velocity     — recent_mentions / total_mentions, weight 0.25
 */
function computeTrendScore(trend: TrendItem): number {
  const growthNorm = (Math.min(Math.max(trend.growth_pct, -100), 300) + 100) / 400;
  const mentionNorm = Math.min(Math.log10(Math.max(trend.total_mentions, 1)) / 4, 1);
  const velocity = trend.total_mentions > 0
    ? Math.min(trend.recent_mentions / trend.total_mentions, 1)
    : 0;
  return Math.round((growthNorm * 0.4 + mentionNorm * 0.35 + velocity * 0.25) * 100);
}

interface Momentum {
  label: string;
  badgeClass: string;
  dotColor: string;
}

function getMomentum(pct: number): Momentum {
  if (pct >= 100) return {
    label: "Emerging",
    badgeClass: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
    dotColor: "#6ee7b7",
  };
  if (pct >= 50) return {
    label: "Surging",
    badgeClass: "bg-[rgba(79,213,255,0.2)] text-[#4fd5ff] border-[rgba(79,213,255,0.35)]",
    dotColor: "#4fd5ff",
  };
  if (pct >= 15) return {
    label: "Accelerating",
    badgeClass: "bg-[rgba(79,213,255,0.1)] text-[#4fd5ff] border-[rgba(79,213,255,0.2)]",
    dotColor: "#60c8f0",
  };
  if (pct >= -10) return {
    label: "Steady",
    badgeClass: "bg-[color:var(--surface-2)] text-[color:var(--ink-faint)] border-[color:var(--line)]",
    dotColor: "#6b7280",
  };
  if (pct >= -30) return {
    label: "Cooling Off",
    badgeClass: "bg-orange-500/10 text-orange-400 border-orange-500/20",
    dotColor: "#fb923c",
  };
  return {
    label: "Sharp Decline",
    badgeClass: "bg-red-500/10 text-red-400 border-red-500/20",
    dotColor: "#f87171",
  };
}

/* ── MiniSparkline ────────────────────────────────────────────────────── */

function MiniSparkline({ points, color }: { points: TrendSparklinePoint[]; color: string }) {
  if (points.length < 2) return null;

  const W = 64;
  const H = 24;
  const counts = points.map((p) => p.count);
  const min = Math.min(...counts);
  const max = Math.max(...counts);
  const range = max - min || 1;

  const xs = points.map((_, i) => (i / (points.length - 1)) * W);
  const ys = counts.map((c) => H - ((c - min) / range) * (H - 2) - 1);

  return (
    <svg width={W} height={H} className="shrink-0 overflow-visible" aria-hidden>
      <polyline
        points={xs.map((x, i) => `${x.toFixed(1)},${ys[i].toFixed(1)}`).join(" ")}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
        opacity="0.7"
      />
      <circle cx={xs[xs.length - 1]} cy={ys[ys.length - 1]} r="2" fill={color} opacity="0.9" />
    </svg>
  );
}

/* ── MoverCard ────────────────────────────────────────────────────────── */

function MoverCard({ trend, type }: { trend: ScoredTrend; type: "rising" | "declining" }) {
  const momentum = getMomentum(trend.growth_pct);
  const sign = trend.growth_pct > 0 ? "+" : "";
  const accentColor = type === "rising" ? "#4fd5ff" : "#f87171";

  return (
    <div
      className="rounded-xl border p-4 flex flex-col gap-2.5"
      style={{
        borderColor: type === "rising" ? "rgba(79,213,255,0.2)" : "rgba(248,113,113,0.2)",
        background: type === "rising" ? "rgba(79,213,255,0.03)" : "rgba(248,113,113,0.03)",
      }}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="text-sm font-semibold text-[color:var(--ink)] leading-snug">{trend.label}</span>
        <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold ${momentum.badgeClass}`}>
          {momentum.label}
        </span>
      </div>

      <div className="flex items-baseline gap-2.5">
        <span className="text-2xl font-bold tabular-nums leading-none" style={{ color: accentColor }}>
          {sign}{trend.growth_pct.toFixed(0)}%
        </span>
        <span className="text-xs text-[color:var(--ink-faint)]">
          {trend.total_mentions.toLocaleString()} mentions
        </span>
      </div>

      {trend.description && (
        <p className="text-xs leading-relaxed text-[color:var(--ink-faint)] line-clamp-2">
          {trend.description}
        </p>
      )}
    </div>
  );
}

/* ── RelatedDoc ───────────────────────────────────────────────────────── */

function RelatedDoc({ doc }: { doc: TrendDocItem }) {
  const inner = (
    <div className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(9,21,34,0.6)] px-3 py-2 transition-colors hover:border-[color:var(--line)] hover:bg-[color:rgba(15,34,54,0.7)]">
      <div className="flex items-start justify-between gap-2">
        <div className="flex flex-wrap items-baseline gap-1.5 min-w-0">
          <p className="text-xs font-medium leading-snug text-[color:var(--ink)] line-clamp-2">
            {doc.title || doc.id}
          </p>
          {doc.source_kind && (
            <span className="shrink-0 rounded-full border border-[color:var(--line)] px-1.5 py-0.5 text-[10px] text-[color:var(--ink-faint)]">
              {sourceLabel(doc.source_kind)}
            </span>
          )}
        </div>
        <span className="shrink-0 text-[10px] text-[color:var(--ink-faint)]">{fmtDate(doc.date)}</span>
      </div>
      {doc.summary && (
        <p className="mt-1 text-[11px] leading-snug text-[color:var(--ink-faint)] line-clamp-2">
          {doc.summary}
        </p>
      )}
    </div>
  );

  return (
    <li>
      {doc.url ? (
        <a href={doc.url} target="_blank" rel="noopener noreferrer" className="block">
          {inner}
        </a>
      ) : inner}
    </li>
  );
}

/* ── TrendRow ─────────────────────────────────────────────────────────── */

function TrendRow({
  trend,
  score,
  expanded,
  onToggle,
}: {
  trend: ScoredTrend;
  score: number;
  expanded: boolean;
  onToggle: () => void;
}) {
  const momentum = getMomentum(trend.growth_pct);
  const sign = trend.growth_pct > 0 ? "+" : "";

  return (
    <div className="border-b border-[color:var(--line)] last:border-b-0">
      <button
        type="button"
        onClick={onToggle}
        className="w-full text-left px-4 py-4 hover:bg-[color:rgba(79,213,255,0.04)] transition-colors"
      >
        <div className="flex items-start gap-3">
          {/* Momentum dot */}
          <div
            className="mt-[5px] h-2 w-2 shrink-0 rounded-full"
            style={{ backgroundColor: momentum.dotColor }}
          />

          {/* Main content */}
          <div className="min-w-0 flex-1">
            {/* Row 1: label + momentum + raw % */}
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <span className="text-sm font-semibold text-[color:var(--ink)]">{trend.label}</span>
              <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${momentum.badgeClass}`}>
                {momentum.label}
              </span>
              <span className="text-xs tabular-nums text-[color:var(--ink-faint)]">
                {sign}{trend.growth_pct.toFixed(0)}%
              </span>
            </div>

            {/* Row 2: stats line */}
            <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 mb-2 text-xs text-[color:var(--ink-faint)]">
              <span>{trend.total_mentions.toLocaleString()} mentions</span>
              <span aria-hidden>·</span>
              <span>{trend.recent_mentions} in last 30d</span>
              {trend.sources.length > 0 && (
                <>
                  <span aria-hidden>·</span>
                  <span>{trend.sources.map(sourceLabel).join(", ")}</span>
                </>
              )}
              <span aria-hidden>·</span>
              <span className="tabular-nums">Score {score}</span>
            </div>

            {/* Row 3: description (always visible) */}
            {trend.description && (
              <p className="text-sm text-[color:var(--ink-faint)] leading-relaxed">
                {trend.description}
              </p>
            )}
          </div>

          {/* Sparkline + chevron */}
          <div className="flex items-center gap-2 shrink-0 mt-0.5">
            {trend.sparkline.length >= 2 && (
              <MiniSparkline points={trend.sparkline} color={momentum.dotColor} />
            )}
            <span
              className={`text-xs text-[color:var(--ink-faint)] transition-transform duration-200 ${expanded ? "rotate-180" : ""}`}
              aria-hidden="true"
            >
              ▾
            </span>
          </div>
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-4 pt-2 space-y-3 border-t border-[color:var(--line)] bg-[color:rgba(9,21,34,0.5)]">
          {/* Related tags */}
          {trend.cluster_tags.length > 1 && (
            <div className="flex flex-wrap gap-1.5 items-center">
              <span className="text-xs text-[color:var(--ink-faint)] mr-1">Related:</span>
              {trend.cluster_tags
                .filter((t) => t !== trend.canonical_tag)
                .slice(0, 8)
                .map((tag) => (
                  <span
                    key={tag}
                    className="rounded-full border border-[color:var(--line)] px-2 py-0.5 text-xs text-[color:var(--ink-faint)]"
                  >
                    {tag}
                  </span>
                ))}
            </div>
          )}

          {/* Related documents */}
          {trend.top_docs.length > 0 && (
            <div>
              <p className="mb-1.5 text-xs font-semibold uppercase tracking-[0.07em] text-[color:var(--ink-faint)]">
                Related Content
              </p>
              <ul className="space-y-1.5">
                {trend.top_docs.map((doc) => (
                  <RelatedDoc key={doc.id} doc={doc} />
                ))}
              </ul>
            </div>
          )}

          {/* Date range */}
          <p className="text-xs text-[color:var(--ink-faint)]">
            First seen: {fmtDate(trend.first_seen)} &nbsp;&middot;&nbsp; Last seen: {fmtDate(trend.last_seen)}
          </p>
        </div>
      )}
    </div>
  );
}

/* ── TrendsDashboard ──────────────────────────────────────────────────── */

export function TrendsDashboard() {
  const [payload, setPayload] = useState<TrendsPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<CategoryFilter>("all");
  const [rangeFilter, setRangeFilter] = useState<RangeFilter>("30d");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  useEffect(() => {
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({ range: rangeFilter });
    fetch(`/api/trends?${params}`)
      .then((r) => r.json() as Promise<ApiEnvelope<TrendsPayload>>)
      .then((envelope) => {
        if (!envelope.ok || !envelope.data) {
          setError(envelope.error ?? "Failed to load trends");
        } else {
          setPayload(envelope.data);
        }
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [rangeFilter]);

  // Scored + sorted by trend score (category + search filter applied)
  const scoredTrends = useMemo<ScoredTrend[]>(() => {
    if (!payload) return [];
    const q = search.trim().toLowerCase();
    return applyCategoryFilter(payload.trends, categoryFilter)
      .filter((t) => {
        if (!q) return true;
        return (
          t.label.toLowerCase().includes(q) ||
          t.description?.toLowerCase().includes(q) ||
          t.cluster_tags.some((tag) => tag.toLowerCase().includes(q))
        );
      })
      .map((t) => ({ ...t, _score: computeTrendScore(t) }))
      .sort((a, b) => b._score - a._score);
  }, [payload, categoryFilter, search]);

  // Top movers (sorted separately from the main list)
  const topRisers = useMemo<ScoredTrend[]>(() =>
    [...scoredTrends].sort((a, b) => b.growth_pct - a.growth_pct).filter((t) => t.growth_pct > 0).slice(0, 3),
    [scoredTrends]
  );
  const topDecliners = useMemo<ScoredTrend[]>(() =>
    [...scoredTrends].sort((a, b) => a.growth_pct - b.growth_pct).filter((t) => t.growth_pct < -10).slice(0, 3),
    [scoredTrends]
  );
  const showMovers = topRisers.length > 0 || topDecliners.length > 0;

  const generatedAt = payload?.generated_at
    ? new Date(payload.generated_at).toLocaleString("en-US", {
        month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit",
      })
    : null;

  const activeCategoryLabel = CATEGORIES.find((c) => c.value === categoryFilter)?.label ?? "";

  return (
    <div className="space-y-6">
      {/* ── Filter bar ───────────────────────────────────────────────── */}
      <div className="space-y-2">
        {/* Category pills */}
        <div className="flex flex-wrap items-center gap-1.5">
          {CATEGORIES.map((cat) => (
            <button
              key={cat.value}
              type="button"
              onClick={() => { setCategoryFilter(cat.value); setExpandedId(null); }}
              className={`rounded-full border px-3 py-1 text-xs font-medium transition-colors ${
                categoryFilter === cat.value
                  ? "border-[color:rgba(79,213,255,0.4)] bg-[color:rgba(79,213,255,0.15)] text-[color:var(--ink)]"
                  : "border-[color:var(--line)] bg-transparent text-[color:var(--ink-faint)] hover:border-[color:var(--line-strong)] hover:text-[color:var(--ink)]"
              }`}
            >
              {cat.label}
            </button>
          ))}
        </div>

        {/* Search */}
        <div className="relative">
          <svg
            className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-[color:var(--ink-faint)]"
            viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"
            aria-hidden
          >
            <circle cx="6.5" cy="6.5" r="4.5" />
            <line x1="10" y1="10" x2="14" y2="14" />
          </svg>
          <input
            type="search"
            placeholder="Search trends…"
            value={search}
            onChange={(e) => { setSearch(e.target.value); setExpandedId(null); }}
            className="w-full rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)] py-1.5 pl-8 pr-3 text-xs text-[color:var(--ink)] placeholder-[color:var(--ink-faint)] outline-none focus:border-[color:rgba(79,213,255,0.35)] focus:ring-0"
          />
        </div>

        {/* Range + metadata */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1 rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)] p-1">
            {RANGE_FILTERS.map((f) => (
              <button
                key={f.value}
                type="button"
                onClick={() => setRangeFilter(f.value)}
                className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                  rangeFilter === f.value
                    ? "bg-[color:rgba(79,213,255,0.18)] text-[color:var(--ink)]"
                    : "text-[color:var(--ink-faint)] hover:text-[color:var(--ink)]"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
          <div className="ml-auto text-xs text-[color:var(--ink-faint)]">
            {generatedAt && <span>Updated {generatedAt}</span>}
          </div>
        </div>
      </div>

      {/* ── Loading ───────────────────────────────────────────────────── */}
      {loading && (
        <div className="flex items-center justify-center py-16 text-sm text-[color:var(--ink-faint)]">
          Loading trends...
        </div>
      )}

      {/* ── Error ────────────────────────────────────────────────────── */}
      {!loading && error && (
        <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* ── Empty ────────────────────────────────────────────────────── */}
      {!loading && !error && scoredTrends.length === 0 && (
        <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)] p-8 text-center">
          <p className="text-sm text-[color:var(--ink-faint)]">
            {payload && payload.trend_count > 0
              ? "No trends matched this category in the selected time window."
              : "The trends pipeline may not have run yet. Check back after the daily cron completes."}
          </p>
        </div>
      )}

      {!loading && !error && scoredTrends.length > 0 && (
        <>
          {/* ── Top Movers ─────────────────────────────────────────── */}
          {showMovers && (
            <div>
              <p className="mb-3 text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
                Top Movers
              </p>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                {/* Rising */}
                {topRisers.length > 0 && (
                  <div>
                    <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.12em]" style={{ color: "#4fd5ff" }}>
                      Rising
                    </p>
                    <div className="space-y-2">
                      {topRisers.map((t) => (
                        <MoverCard key={t.id} trend={t} type="rising" />
                      ))}
                    </div>
                  </div>
                )}

                {/* Declining */}
                {topDecliners.length > 0 && (
                  <div>
                    <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.12em]" style={{ color: "#f87171" }}>
                      Declining
                    </p>
                    <div className="space-y-2">
                      {topDecliners.map((t) => (
                        <MoverCard key={t.id} trend={t} type="declining" />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── All Trends list ────────────────────────────────────── */}
          <div>
            {showMovers && (
              <p className="mb-3 text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
                All Trends
              </p>
            )}
            <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
              {/* Summary header */}
              <div className="flex items-center justify-between border-b border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)] px-4 py-2">
                <span className="text-xs text-[color:var(--ink-faint)]">
                  {scoredTrends.length} trend{scoredTrends.length !== 1 ? "s" : ""}
                  {categoryFilter !== "all" && (
                    <span className="ml-1 text-[color:var(--accent)]">· {activeCategoryLabel}</span>
                  )}
                </span>
                <span className="text-xs text-[color:var(--ink-faint)]">Sorted by Trend Score</span>
              </div>

              {scoredTrends.map((trend) => (
                <TrendRow
                  key={trend.id}
                  trend={trend}
                  score={trend._score}
                  expanded={expandedId === trend.id}
                  onToggle={() => setExpandedId((prev) => (prev === trend.id ? null : trend.id))}
                />
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
