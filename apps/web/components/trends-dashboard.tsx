"use client";

import { useEffect, useMemo, useState } from "react";
import { SparklineChart } from "@/components/sparkline-chart";
import type { TrendItem, TrendsPayload } from "@/lib/server/types";

interface ApiEnvelope<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

type GrowthFilter = "all" | "rising" | "surging" | "new";
type RangeFilter = "30d" | "7d" | "90d";

const GROWTH_FILTERS: { value: GrowthFilter; label: string }[] = [
  { value: "all", label: "All Trends" },
  { value: "rising", label: "Rising (>10%)" },
  { value: "surging", label: "Surging (>50%)" },
  { value: "new", label: "New (100%+)" },
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
  finra_comment_letter: "FINRA",
  doj_usao_press_release: "DOJ",
  cftc_press_release: "CFTC",
  newsapi_article: "News",
  reddit_post: "Reddit",
  uploaded: "Uploaded",
};

function sourceLabel(kind: string): string {
  return SOURCE_KIND_LABELS[kind] || kind.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function fmtDate(value: string): string {
  if (!value) return "-";
  const d = new Date(value + "T00:00:00Z");
  return Number.isNaN(d.getTime())
    ? value
    : d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
}

function GrowthBadge({ pct }: { pct: number }) {
  let colorClass: string;
  let prefix: string;

  if (pct >= 100) {
    colorClass = "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
    prefix = "+";
  } else if (pct > 0) {
    colorClass = "bg-[color:rgba(79,213,255,0.15)] text-[color:var(--accent)] border-[color:rgba(79,213,255,0.25)]";
    prefix = "+";
  } else if (pct < 0) {
    colorClass = "bg-red-500/10 text-red-400 border-red-500/20";
    prefix = "";
  } else {
    colorClass = "bg-[color:var(--surface-2)] text-[color:var(--ink-faint)] border-[color:var(--line)]";
    prefix = "";
  }

  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-semibold tabular-nums ${colorClass}`}>
      {prefix}{pct.toFixed(0)}%
    </span>
  );
}

function SizeBadge({ count }: { count: number }) {
  return (
    <span className="inline-flex items-center rounded-full border border-[color:var(--line)] bg-[color:var(--surface-2)] px-2 py-0.5 text-xs font-medium tabular-nums text-[color:var(--ink-faint)]">
      {count.toLocaleString()} mentions
    </span>
  );
}

function TrendRow({ trend, expanded, onToggle }: { trend: TrendItem; expanded: boolean; onToggle: () => void }) {
  return (
    <div className="border-b border-[color:var(--line)] last:border-b-0">
      <button
        type="button"
        onClick={onToggle}
        className="w-full text-left px-4 py-4 hover:bg-[color:rgba(79,213,255,0.04)] transition-colors"
      >
        <div className="flex items-start gap-3">
          {/* Sparkline */}
          <div className="mt-0.5 shrink-0">
            <SparklineChart data={trend.sparkline} />
          </div>

          {/* Main content */}
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <span className="text-sm font-semibold text-[color:var(--ink)]">{trend.label}</span>
              <GrowthBadge pct={trend.growth_pct} />
              <SizeBadge count={trend.total_mentions} />
            </div>
            <p className="text-sm text-[color:var(--ink-faint)] line-clamp-2">{trend.description}</p>
          </div>

          {/* Chevron */}
          <span
            className={`ml-2 mt-1 shrink-0 text-[color:var(--ink-faint)] transition-transform duration-200 ${expanded ? "rotate-180" : ""}`}
            aria-hidden="true"
          >
            ▾
          </span>
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-4 pt-1 space-y-3 border-t border-[color:var(--line)] bg-[color:rgba(9,21,34,0.5)]">
          {/* Sources */}
          {trend.sources.length > 0 && (
            <div className="flex flex-wrap gap-1.5 items-center">
              <span className="text-xs text-[color:var(--ink-faint)] mr-1">Sources:</span>
              {trend.sources.map((s) => (
                <span
                  key={s}
                  className="rounded-full border border-[color:var(--line)] bg-[color:var(--surface-2)] px-2 py-0.5 text-xs text-[color:var(--ink-faint)]"
                >
                  {sourceLabel(s)}
                </span>
              ))}
            </div>
          )}

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

          {/* Date range */}
          <p className="text-xs text-[color:var(--ink-faint)]">
            First seen: {fmtDate(trend.first_seen)} &nbsp;&middot;&nbsp; Last seen: {fmtDate(trend.last_seen)} &nbsp;&middot;&nbsp; Recent: {trend.recent_mentions} mentions (30d)
          </p>
        </div>
      )}
    </div>
  );
}

function applyGrowthFilter(trends: TrendItem[], filter: GrowthFilter): TrendItem[] {
  if (filter === "rising") return trends.filter((t) => t.growth_pct > 10);
  if (filter === "surging") return trends.filter((t) => t.growth_pct > 50);
  if (filter === "new") return trends.filter((t) => t.growth_pct >= 100);
  return trends;
}

export function TrendsDashboard() {
  const [payload, setPayload] = useState<TrendsPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [growthFilter, setGrowthFilter] = useState<GrowthFilter>("all");
  const [rangeFilter, setRangeFilter] = useState<RangeFilter>("30d");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({ range: rangeFilter });
    fetch(`/api/trends?${params}`)
      .then((r) => r.json() as Promise<ApiEnvelope<TrendsPayload>>)
      .then((envelope) => {
        if (!envelope.ok || !envelope.data) {
          setError(envelope.error || "Failed to load trends");
        } else {
          setPayload(envelope.data);
        }
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [rangeFilter]);

  const filteredTrends = useMemo(() => {
    if (!payload) return [];
    return applyGrowthFilter(payload.trends, growthFilter);
  }, [payload, growthFilter]);

  const generatedAt = payload?.generated_at
    ? new Date(payload.generated_at).toLocaleString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
        hour: "numeric",
        minute: "2-digit",
      })
    : null;

  return (
    <div className="space-y-4">
      {/* Filters bar */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Range filter */}
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

        {/* Growth filter */}
        <div className="flex items-center gap-1 rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)] p-1">
          {GROWTH_FILTERS.map((f) => (
            <button
              key={f.value}
              type="button"
              onClick={() => setGrowthFilter(f.value)}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                growthFilter === f.value
                  ? "bg-[color:rgba(79,213,255,0.18)] text-[color:var(--ink)]"
                  : "text-[color:var(--ink-faint)] hover:text-[color:var(--ink)]"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>

        {/* Metadata */}
        <div className="ml-auto text-xs text-[color:var(--ink-faint)]">
          {generatedAt && <span>Updated {generatedAt}</span>}
        </div>
      </div>

      {/* Content */}
      {loading && (
        <div className="flex items-center justify-center py-16 text-[color:var(--ink-faint)] text-sm">
          Loading trends...
        </div>
      )}

      {!loading && error && (
        <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-400">
          {error}
        </div>
      )}

      {!loading && !error && filteredTrends.length === 0 && (
        <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)] p-8 text-center">
          <p className="text-sm text-[color:var(--ink-faint)]">
            No trends found for the selected filters.
          </p>
          {payload && payload.trend_count === 0 && (
            <p className="mt-2 text-xs text-[color:var(--ink-faint)]">
              The trends pipeline may not have run yet. Check back after the daily cron completes.
            </p>
          )}
        </div>
      )}

      {!loading && !error && filteredTrends.length > 0 && (
        <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)] overflow-hidden">
          {/* Summary line */}
          <div className="flex items-center justify-between px-4 py-2 border-b border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)]">
            <span className="text-xs text-[color:var(--ink-faint)]">
              {filteredTrends.length} trend{filteredTrends.length !== 1 ? "s" : ""}
            </span>
          </div>

          {filteredTrends.map((trend) => (
            <TrendRow
              key={trend.id}
              trend={trend}
              expanded={expandedId === trend.id}
              onToggle={() => setExpandedId((prev) => (prev === trend.id ? null : trend.id))}
            />
          ))}
        </div>
      )}
    </div>
  );
}
