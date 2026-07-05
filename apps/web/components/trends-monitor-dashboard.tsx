"use client";

import { useEffect, useMemo, useState } from "react";
import type { TrendDocItem, TrendItem, TrendsPayload } from "@/lib/server/types";

interface ApiEnvelope<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

type RangeFilter = "all" | "7d" | "30d" | "90d";
type MonitorLane = "all" | "rising" | "cooling" | "high_volume" | "new";
type SortKey = "score" | "growth" | "mentions" | "recent" | "last_seen";

type ScoredTrend = TrendItem & {
  score: number;
  lane: Exclude<MonitorLane, "all"> | "steady";
};

const RANGE_FILTERS: Array<{ value: RangeFilter; label: string }> = [
  { value: "all", label: "All" },
  { value: "7d", label: "7D" },
  { value: "30d", label: "30D" },
  { value: "90d", label: "90D" },
];

const LANES: Array<{ value: MonitorLane; label: string }> = [
  { value: "all", label: "All" },
  { value: "rising", label: "Rising" },
  { value: "cooling", label: "Cooling" },
  { value: "high_volume", label: "High Volume" },
  { value: "new", label: "New" },
];

const SORTS: Array<{ value: SortKey; label: string }> = [
  { value: "score", label: "Score" },
  { value: "growth", label: "Growth" },
  { value: "mentions", label: "Mentions" },
  { value: "recent", label: "Recent" },
  { value: "last_seen", label: "Last Seen" },
];

const SOURCE_KIND_LABELS: Record<string, string> = {
  sec_speech: "SEC",
  sec_enforcement_litigation: "SEC Enforcement",
  sec_press_release_rss: "SEC",
  sec_administrative_proceeding: "SEC Proceedings",
  sec_trading_suspension: "SEC Suspensions",
  sec_federal_register: "SEC Federal Register",
  sec_pcaob_rulemaking: "SEC PCAOB",
  sec_tm_faq: "SEC TM FAQ",
  finra_regulatory_notice: "FINRA",
  finra_awc: "FINRA AWC",
  doj_usao_press_release: "DOJ",
  federal_reserve_speech_testimony: "Federal Reserve",
  cisa_cybersecurity_advisory: "CISA",
  cftc_press_release: "CFTC",
  cftc_public_statement_remark: "CFTC",
  pcaob_update: "PCAOB",
  msrb_press_release: "MSRB",
  treasury_featured_story: "Treasury",
  treasury_press_release: "Treasury",
  treasury_statement_remark: "Treasury",
  sifma_news_item: "SIFMA",
  ici_news_item: "ICI",
  isda_news_item: "ISDA",
  mfa_news_item: "MFA",
  fia_news_item: "FIA",
  aba_news_item: "ABA",
  bpi_news_item: "BPI",
  icba_news_item: "ICBA",
  lsta_news_item: "LSTA",
  bloomberg_public_article: "Bloomberg",
  bloomberg_apify_article: "Bloomberg",
  substack_public_article: "Substack",
  jdsupra_article: "JD Supra",
  investmentnews_article: "InvestmentNews",
  citywire_article: "Citywire",
  therecord_media_article: "The Record",
  krebs_on_security_article: "Krebs on Security",
  the_hacker_news_article: "The Hacker News",
  welivesecurity_article: "WeLiveSecurity",
  sophos_security_operations_article: "Sophos",
  flashpoint_blog_article: "Flashpoint",
  recorded_future_article: "Recorded Future",
  intel471_blog_article: "Intel 471",
  securityweek_article: "SecurityWeek",
  dark_reading_article: "Dark Reading",
  wired_article: "WIRED",
  tripwire_article: "Tripwire",
  akamai_blog_article: "Akamai Blog",
  ritholtz_article: "The Big Picture",
  ft_portfolios_market_commentary: "First Trust",
  liberty_street_economics_article: "Liberty Street Economics",
  wealth_of_common_sense_article: "A Wealth of Common Sense",
  prnewswire_article: "PR Newswire",
  google_news_ponzi_investor_fraud_article: "Google News: Ponzi & Fraud",
  google_news_senate_committee_article: "Google News: Senate",
  coindesk_article: "CoinDesk",
  cointelegraph_article: "Cointelegraph",
  decrypt_article: "Decrypt",
  the_block_article: "The Block",
  congress_crs_product: "CRS",
  senate_committee_site: "Senate",
  wsj_dow_jones: "WSJ",
  newsapi_article: "News",
  rss_news_feed: "News Feed",
  reddit_post: "Reddit",
  hedge_fund_letter: "Hedge Fund Letters",
};

function sourceLabel(kind: string): string {
  return SOURCE_KIND_LABELS[kind] ?? kind.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function fmtDate(value: string): string {
  if (!value) return "-";
  const date = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}

function daysAgo(value: string): number {
  const date = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return Number.POSITIVE_INFINITY;
  return Math.max(0, Math.floor((Date.now() - date.getTime()) / 86_400_000));
}

function trendScore(trend: TrendItem): number {
  const growthNorm = (Math.min(Math.max(trend.growth_pct, -100), 300) + 100) / 400;
  const mentionNorm = Math.min(Math.log10(Math.max(trend.total_mentions, 1)) / 4, 1);
  const velocity = trend.total_mentions > 0 ? Math.min(trend.recent_mentions / trend.total_mentions, 1) : 0;
  return Math.round((growthNorm * 0.42 + mentionNorm * 0.33 + velocity * 0.25) * 100);
}

function classifyLane(trend: TrendItem): ScoredTrend["lane"] {
  if (trend.growth_pct >= 75 || trend.recent_mentions >= Math.max(8, trend.total_mentions * 0.55)) return "rising";
  if (trend.growth_pct <= -20) return "cooling";
  if (trend.total_mentions >= 50 || trend.recent_mentions >= 20) return "high_volume";
  if (daysAgo(trend.first_seen) <= 14 || trend.growth_pct >= 100) return "new";
  return "steady";
}

function signedPct(value: number): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(0)}%`;
}

function laneTone(lane: ScoredTrend["lane"]): string {
  if (lane === "rising") return "border-emerald-400/35 bg-emerald-400/10 text-emerald-200";
  if (lane === "cooling") return "border-orange-400/35 bg-orange-400/10 text-orange-200";
  if (lane === "high_volume") return "border-sky-300/35 bg-sky-300/10 text-sky-100";
  if (lane === "new") return "border-amber-300/35 bg-amber-300/10 text-amber-100";
  return "border-[color:var(--line)] bg-[color:rgba(12,24,38,0.72)] text-[color:var(--ink-faint)]";
}

function laneLabel(lane: ScoredTrend["lane"]): string {
  if (lane === "high_volume") return "High Volume";
  if (lane === "rising") return "Rising";
  if (lane === "cooling") return "Cooling";
  if (lane === "new") return "New";
  return "Steady";
}

function MiniBars({ trend }: { trend: TrendItem }) {
  const points = trend.sparkline.slice(-18);
  const max = Math.max(1, ...points.map((point) => point.count));
  return (
    <div className="flex h-9 w-24 items-end gap-0.5" aria-hidden="true">
      {points.map((point) => (
        <span
          key={point.date}
          className="w-1 rounded-t-sm bg-[color:var(--accent)]/55"
          style={{ height: `${Math.max(3, (point.count / max) * 34)}px` }}
        />
      ))}
    </div>
  );
}

function MetricTile({ label, value, active, onClick }: { label: string; value: number; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`min-h-20 rounded-lg border px-4 py-3 text-left transition-colors ${
        active
          ? "border-[color:var(--accent)]/55 bg-[color:rgba(79,213,255,0.14)]"
          : "border-[color:var(--line)] bg-[color:rgba(8,18,30,0.72)] hover:border-[color:var(--line-strong)]"
      }`}
    >
      <span className="block text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">{label}</span>
      <span className="mt-2 block text-2xl font-semibold tabular-nums text-[color:var(--ink)]">{value.toLocaleString()}</span>
    </button>
  );
}

function TopDoc({ doc }: { doc: TrendDocItem }) {
  const content = (
    <div className="rounded-md border border-[color:var(--line-soft)] bg-[color:rgba(8,18,30,0.68)] p-3 hover:border-[color:var(--line-strong)]">
      <div className="flex items-start justify-between gap-3">
        <p className="min-w-0 text-sm font-medium leading-snug text-[color:var(--ink)]">{doc.title || doc.id}</p>
        <span className="shrink-0 text-xs tabular-nums text-[color:var(--ink-faint)]">{fmtDate(doc.date)}</span>
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-[color:var(--ink-faint)]">
        {doc.source_kind ? <span>{sourceLabel(doc.source_kind)}</span> : null}
        {doc.summary ? <span className="line-clamp-1">{doc.summary}</span> : null}
      </div>
    </div>
  );
  return doc.url ? (
    <a href={doc.url} target="_blank" rel="noopener noreferrer" className="block">
      {content}
    </a>
  ) : content;
}

function DetailPanel({ trend }: { trend: ScoredTrend | null }) {
  if (!trend) {
    return (
      <aside className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.72)] p-5">
        <p className="text-sm text-[color:var(--ink-faint)]">Select a trend to inspect its source mix, related tags, and linked documents.</p>
      </aside>
    );
  }

  return (
    <aside className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.82)]">
      <div className="border-b border-[color:var(--line)] p-5">
        <div className="flex flex-wrap items-center gap-2">
          <span className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${laneTone(trend.lane)}`}>
            {laneLabel(trend.lane)}
          </span>
          <span className="text-xs text-[color:var(--ink-faint)]">Score {trend.score}</span>
        </div>
        <h2 className="mt-3 text-lg font-semibold leading-snug text-[color:var(--ink)]">{trend.label}</h2>
        {trend.description ? <p className="mt-3 text-sm leading-6 text-[color:var(--ink-soft)]">{trend.description}</p> : null}
      </div>

      <div className="grid grid-cols-3 border-b border-[color:var(--line)] text-center">
        <div className="p-3">
          <p className="text-xs text-[color:var(--ink-faint)]">Growth</p>
          <p className="mt-1 text-base font-semibold tabular-nums text-[color:var(--ink)]">{signedPct(trend.growth_pct)}</p>
        </div>
        <div className="border-x border-[color:var(--line)] p-3">
          <p className="text-xs text-[color:var(--ink-faint)]">Mentions</p>
          <p className="mt-1 text-base font-semibold tabular-nums text-[color:var(--ink)]">{trend.total_mentions.toLocaleString()}</p>
        </div>
        <div className="p-3">
          <p className="text-xs text-[color:var(--ink-faint)]">Recent</p>
          <p className="mt-1 text-base font-semibold tabular-nums text-[color:var(--ink)]">{trend.recent_mentions.toLocaleString()}</p>
        </div>
      </div>

      <div className="space-y-5 p-5">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Sources</p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {trend.sources.map((source) => (
              <span key={source} className="rounded-full border border-[color:var(--line)] px-2 py-1 text-xs text-[color:var(--ink-soft)]">
                {sourceLabel(source)}
              </span>
            ))}
          </div>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Related Tags</p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {trend.cluster_tags.slice(0, 14).map((tag) => (
              <span key={tag} className="rounded-md border border-[color:var(--line-soft)] px-2 py-1 text-xs text-[color:var(--ink-faint)]">
                {tag}
              </span>
            ))}
          </div>
        </div>

        <div>
          <div className="mb-2 flex items-center justify-between gap-3">
            <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Top Documents</p>
            <span className="text-xs text-[color:var(--ink-faint)]">{fmtDate(trend.first_seen)} to {fmtDate(trend.last_seen)}</span>
          </div>
          <div className="space-y-2">
            {trend.top_docs.slice(0, 6).map((doc) => (
              <TopDoc key={doc.id} doc={doc} />
            ))}
          </div>
        </div>
      </div>
    </aside>
  );
}

export function TrendsMonitorDashboard() {
  const [payload, setPayload] = useState<TrendsPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [range, setRange] = useState<RangeFilter>("all");
  const [lane, setLane] = useState<MonitorLane>("all");
  const [sort, setSort] = useState<SortKey>("score");
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<string>("");

  useEffect(() => {
    setLoading(true);
    setError(null);
    const params = new URLSearchParams();
    if (range !== "all") {
      params.set("range", range);
    }
    const query = params.toString();
    fetch(`/api/trends${query ? `?${query}` : ""}`)
      .then((res) => res.json() as Promise<ApiEnvelope<TrendsPayload>>)
      .then((envelope) => {
        if (!envelope.ok || !envelope.data) {
          setError(envelope.error ?? "Failed to load trends");
          setPayload(null);
          return;
        }
        setPayload(envelope.data);
      })
      .catch((err) => {
        setError(String(err));
        setPayload(null);
      })
      .finally(() => setLoading(false));
  }, [range]);

  const trends = useMemo<ScoredTrend[]>(() => {
    if (!payload) return [];
    return payload.trends.map((trend) => ({
      ...trend,
      score: trendScore(trend),
      lane: classifyLane(trend),
    }));
  }, [payload]);

  const counts = useMemo(() => ({
    all: trends.length,
    rising: trends.filter((trend) => trend.lane === "rising").length,
    cooling: trends.filter((trend) => trend.lane === "cooling").length,
    high_volume: trends.filter((trend) => trend.lane === "high_volume").length,
    new: trends.filter((trend) => trend.lane === "new").length,
  }), [trends]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    const selected = trends
      .filter((trend) => lane === "all" || trend.lane === lane)
      .filter((trend) => {
        if (!q) return true;
        return (
          trend.label.toLowerCase().includes(q) ||
          trend.description.toLowerCase().includes(q) ||
          trend.cluster_tags.some((tag) => tag.toLowerCase().includes(q)) ||
          trend.sources.some((source) => sourceLabel(source).toLowerCase().includes(q))
        );
      });

    const sorted = [...selected].sort((a, b) => {
      if (sort === "growth") return b.growth_pct - a.growth_pct;
      if (sort === "mentions") return b.total_mentions - a.total_mentions;
      if (sort === "recent") return b.recent_mentions - a.recent_mentions;
      if (sort === "last_seen") return new Date(b.last_seen).getTime() - new Date(a.last_seen).getTime();
      return b.score - a.score;
    });
    return sorted;
  }, [lane, search, sort, trends]);

  useEffect(() => {
    if (filtered.length === 0) {
      setSelectedId("");
      return;
    }
    if (!filtered.some((trend) => trend.id === selectedId)) {
      setSelectedId(filtered[0].id);
    }
  }, [filtered, selectedId]);

  const selected = filtered.find((trend) => trend.id === selectedId) ?? filtered[0] ?? null;
  const generatedAt = payload?.generated_at
    ? new Date(payload.generated_at).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })
    : "";

  return (
    <div className="space-y-5">
      <div className="grid gap-3 md:grid-cols-5">
        {LANES.map((item) => (
          <MetricTile
            key={item.value}
            label={item.label}
            value={counts[item.value]}
            active={lane === item.value}
            onClick={() => setLane(item.value)}
          />
        ))}
      </div>

      <div className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.7)] p-3">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
          <div className="flex items-center gap-1 rounded-lg border border-[color:var(--line)] bg-[color:rgba(5,12,19,0.62)] p-1">
            {RANGE_FILTERS.map((item) => (
              <button
                key={item.value}
                type="button"
                onClick={() => setRange(item.value)}
                className={`min-h-9 rounded-md px-3 text-sm font-semibold ${range === item.value ? "bg-[color:rgba(79,213,255,0.18)] text-[color:var(--ink)]" : "text-[color:var(--ink-faint)] hover:text-[color:var(--ink)]"}`}
              >
                {item.label}
              </button>
            ))}
          </div>

          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search label, source, or tag"
            className="min-h-10 flex-1 rounded-md border border-[color:var(--line)] bg-[color:rgba(5,12,19,0.62)] px-3 text-sm text-[color:var(--ink)] placeholder:text-[color:var(--ink-faint)]"
          />

          <label className="flex items-center gap-2 text-sm text-[color:var(--ink-faint)]">
            Sort
            <select
              value={sort}
              onChange={(event) => setSort(event.target.value as SortKey)}
              className="min-h-10 rounded-md border border-[color:var(--line)] bg-[color:rgba(5,12,19,0.9)] px-3 text-sm text-[color:var(--ink)]"
            >
              {SORTS.map((item) => (
                <option key={item.value} value={item.value}>{item.label}</option>
              ))}
            </select>
          </label>
        </div>
      </div>

      {loading ? (
        <div className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.72)] p-8 text-center text-sm text-[color:var(--ink-faint)]">
          Loading trend monitor...
        </div>
      ) : null}

      {!loading && error ? (
        <div className="rounded-lg border border-red-400/30 bg-red-400/10 p-4 text-sm text-red-200">{error}</div>
      ) : null}

      {!loading && !error ? (
        <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_420px]">
          <section className="overflow-hidden rounded-lg border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.72)]">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[color:var(--line)] px-4 py-3">
              <div>
                <p className="text-sm font-semibold text-[color:var(--ink)]">{filtered.length.toLocaleString()} monitored trends</p>
                <p className="text-xs text-[color:var(--ink-faint)]">{generatedAt ? `Updated ${generatedAt}` : "Waiting for trend data"}</p>
              </div>
              <p className="text-xs text-[color:var(--ink-faint)]">Click a row for detail</p>
            </div>

            {filtered.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="min-w-[900px] w-full border-collapse text-left">
                  <thead className="bg-[color:rgba(13,31,48,0.92)] text-xs uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
                    <tr>
                      <th className="px-4 py-3 font-semibold">Trend</th>
                      <th className="px-3 py-3 font-semibold">State</th>
                      <th className="px-3 py-3 text-right font-semibold">Score</th>
                      <th className="px-3 py-3 text-right font-semibold">Growth</th>
                      <th className="px-3 py-3 text-right font-semibold">Mentions</th>
                      <th className="px-3 py-3 text-right font-semibold">Recent</th>
                      <th className="px-3 py-3 font-semibold">Sources</th>
                      <th className="px-4 py-3 font-semibold">30D</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((trend) => {
                      const active = selected?.id === trend.id;
                      return (
                        <tr
                          key={trend.id}
                          className={`cursor-pointer border-t border-[color:var(--line-soft)] ${active ? "bg-[color:rgba(79,213,255,0.1)]" : "hover:bg-[color:rgba(79,213,255,0.05)]"}`}
                          onClick={() => setSelectedId(trend.id)}
                        >
                          <td className="max-w-[360px] px-4 py-3 align-top">
                            <button
                              type="button"
                              onClick={() => setSelectedId(trend.id)}
                              aria-pressed={active}
                              className="block w-full text-left"
                            >
                              <span className="block text-sm font-semibold leading-snug text-[color:var(--ink)]">{trend.label}</span>
                              <span className="mt-1 block truncate text-xs text-[color:var(--ink-faint)]">{trend.description}</span>
                            </button>
                          </td>
                          <td className="px-3 py-3 align-top">
                            <span className={`inline-flex rounded-full border px-2 py-0.5 text-xs font-semibold ${laneTone(trend.lane)}`}>
                              {laneLabel(trend.lane)}
                            </span>
                          </td>
                          <td className="px-3 py-3 text-right align-top text-sm font-semibold tabular-nums text-[color:var(--ink)]">{trend.score}</td>
                          <td className={`px-3 py-3 text-right align-top text-sm font-semibold tabular-nums ${trend.growth_pct < 0 ? "text-orange-200" : "text-emerald-200"}`}>{signedPct(trend.growth_pct)}</td>
                          <td className="px-3 py-3 text-right align-top text-sm tabular-nums text-[color:var(--ink-soft)]">{trend.total_mentions.toLocaleString()}</td>
                          <td className="px-3 py-3 text-right align-top text-sm tabular-nums text-[color:var(--ink-soft)]">{trend.recent_mentions.toLocaleString()}</td>
                          <td className="max-w-[180px] px-3 py-3 align-top text-xs text-[color:var(--ink-faint)]">
                            {trend.sources.slice(0, 3).map(sourceLabel).join(", ")}
                          </td>
                          <td className="px-4 py-3 align-top"><MiniBars trend={trend} /></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="p-8 text-center text-sm text-[color:var(--ink-faint)]">No trends match the current monitor filters.</div>
            )}
          </section>

          <div className="xl:sticky xl:top-20 xl:self-start">
            <DetailPanel trend={selected} />
          </div>
        </div>
      ) : null}
    </div>
  );
}
