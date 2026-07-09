"use client";

import { useEffect, useRef, useState } from "react";
import { JobStatusBadge } from "@/components/job-status-badge";
import {
  TOPIC_RULE_RECOMMENDATION_BY_KEY,
  formatTopicRuleKeywords,
  type TopicRuleRecommendation,
} from "@/lib/topic-rule-recommendations";

/* ─── Knowledge Index types ────────────────────────────────────────── */
type OrgIndexStatus = {
  org_key: string;
  org_label: string;
  vector_store_id: string | null;
  corpus_count: number;
  indexed_count: number;
  pending: number;
  updated_at: string | null;
  last_sync: { sync_mode: string; status: string; uploaded: number; deleted: number; failed_count: number } | null;
};

/* ─── RSS Feed types ───────────────────────────────────────────────── */
type RssFeed = { id: number; label: string; feed_url: string; feed_key: string; active: boolean; refresh_interval_minutes?: number; last_refresh_at?: string | null; last_error?: string | null; consecutive_failures?: number };
type XAccountFeed = RssFeed & { username: string };
type TopicRule = { id: number; topic_key: string; label: string; keywords: string; active: boolean; sort_order: number };
type YouTubeChannelConfig = {
  id: string;
  label: string;
  channel_ref: string;
  active: boolean;
  extraction_limit: number;
  enrich_limit: number;
  max_pages: number;
  connector?: "sec_youtube_video" | "youtube_video";
  last_run_at?: string;
  last_status?: string;
  last_error?: string;
};

/* ─── Ticker types ─────────────────────────────────────────────────── */
type TickerEntry = { symbol: string; name: string };
type ValidationResult =
  | { valid: false; error: string }
  | { valid: true; symbol: string; name: string; price: number; change: number; pct: number; up: boolean };

const MAX = 10;

/* ─── Workflow field types ─────────────────────────────────────────── */
type FieldDef =
  | { name: string; label: string; type: "text"; default?: string; placeholder?: string }
  | { name: string; label: string; type: "number"; default?: string; placeholder?: string }
  | { name: string; label: string; type: "select"; default?: string; options: { value: string; label: string }[] }
  | { name: string; label: string; type: "boolean"; default?: "true" | "false" };

type AdminJobState = {
  job_id: string;
  provider?: string;
  workflow?: string;
  status: "queued" | "running" | "success" | "failed" | "unknown";
  status_url?: string;
  github_run_id?: number;
  html_url?: string;
  created_at?: string;
  started_at?: string;
  updated_at?: string;
  finished_at?: string;
  conclusion?: string;
  artifacts?: string[];
};

type MetricsData = {
  connector_audit?: {
    newsapi?: {
      total: number;
      in_feed: number;
      recent_24h: number;
      recent_7d: number;
      recent_30d: number;
      newest: null | {
        title: string;
        url: string;
        source_name: string;
        published_at: string;
        extraction_mode: string;
      };
      by_source: Array<{ source_name: string; count: number }>;
    };
    feed_documents?: {
      total: number;
      by_source_kind: Array<{ source_kind: string; count: number }>;
    };
  };
  recent_ingest?: {
    last_run_at: string;
    processed_count: number;
    failed_count: number;
  };
};

type SourceHealthCounts = {
  discovered?: number;
  processed?: number;
  saved_new?: number;
  saved_updates?: number;
  failed?: number;
  enriched?: number;
  fallback_enriched?: number;
};

type SourceHealthSource = {
  source_key: string;
  last_run_at?: string;
  last_status?: string;
  last_error_category?: string;
  last_error?: string;
  last_workflow?: string;
  last_run_id?: string;
  last_counts?: SourceHealthCounts;
  last_success_at?: string;
  consecutive_failures?: number;
};

type SourceHealthRun = {
  id?: string;
  source_key: string;
  command?: string;
  workflow?: string;
  run_id?: string;
  status?: string;
  ran_at?: string;
  error_category?: string;
  sample_error?: string;
  discovered_count?: number;
  processed_count?: number;
  saved_new?: number;
  saved_updates?: number;
  failed_count?: number;
  enriched_count?: number;
  fallback_enriched_count?: number;
};

type SourceHealthReport = {
  generated_at?: string;
  recent_run_count?: number;
  recent_failed_run_count?: number;
  recent_partial_run_count?: number;
  failing_sources?: SourceHealthSource[];
  stale_sources?: SourceHealthSource[];
  quiet_sources?: SourceHealthSource[];
  error_categories?: Record<string, number>;
  recent_runs?: SourceHealthRun[];
};

type SourceHealthData = {
  updated_at: string;
  sources: SourceHealthSource[];
  runs: SourceHealthRun[];
  latest_report?: {
    generated_at?: string;
    title?: string;
    ai_review?: string;
    report?: SourceHealthReport;
  } | null;
};

/* ─── Workflow definitions ─────────────────────────────────────────── */
const POLICY_EXTRACTION_FIELDS: FieldDef[] = [
  {
    name: "connector",
    label: "Connector",
    type: "select",
    default: "sec_enforcement_litigation",
    options: [
      { value: "sec_speech", label: "SEC Speech" },
      { value: "sec_tm_faq", label: "SEC TM FAQ" },
      { value: "sec_rule_comment", label: "SEC Rule Release + Public Comments" },
      { value: "sec_enforcement_litigation", label: "SEC Enforcement Litigation" },
      { value: "finra_regulatory_notice", label: "FINRA Regulatory Notice" },
      { value: "finra_comment_letter", label: "FINRA Rule Comment Letter" },
      { value: "finra_awc", label: "FINRA AWC Disciplinary Actions" },
      { value: "doj_usao_press_release", label: "DOJ USAO Press Release" },
      { value: "federal_reserve_speech_testimony", label: "Federal Reserve Speech / Testimony" },
      { value: "cftc_press_release", label: "CFTC Press Release" },
      { value: "cftc_public_statement_remark", label: "CFTC Public Statement / Remark" },
      { value: "sec_press_release_rss", label: "SEC Press Release RSS" },
      { value: "sec_administrative_proceeding", label: "SEC Administrative Proceeding" },
      { value: "sec_trading_suspension", label: "SEC Trading Suspension" },
      { value: "sec_federal_register", label: "SEC Federal Register" },
      { value: "sec_pcaob_rulemaking", label: "SEC PCAOB Rulemaking" },
      { value: "pcaob_update", label: "PCAOB Update" },
      { value: "msrb_press_release", label: "MSRB Press Release" },
      { value: "congress_crs_product", label: "Congress CRS Product" },
      { value: "bloomberg_public_latest", label: "Bloomberg Latest (Public RSS)" },
      { value: "bloomberg_public_article", label: "Bloomberg Article (Public)" },
      { value: "substack_public_article", label: "Substack Public Article" },
      { value: "treasury_featured_story", label: "Treasury Featured Story" },
      { value: "treasury_press_release", label: "Treasury Press Release" },
      { value: "treasury_statement_remark", label: "Treasury Statement / Remark" },
      { value: "sifma_news_item", label: "SIFMA News" },
      { value: "ici_news_item", label: "ICI News" },
      { value: "isda_news_item", label: "ISDA News" },
      { value: "mfa_news_item", label: "Managed Funds Association News" },
      { value: "fia_news_item", label: "FIA News" },
      { value: "aba_news_item", label: "American Bankers Association Press Releases" },
      { value: "bpi_news_item", label: "Bank Policy Institute News" },
      { value: "icba_news_item", label: "ICBA News" },
      { value: "lsta_news_item", label: "LSTA News" },
      { value: "jdsupra_article", label: "Trade Media: JD Supra" },
      { value: "investmentnews_article", label: "Trade Media: InvestmentNews" },
      { value: "citywire_article", label: "Trade Media: Citywire" },
      { value: "therecord_media_article", label: "Trade Media: The Record" },
      { value: "wired_article", label: "Trade Media: WIRED" },
      { value: "tripwire_article", label: "Trade Media: Tripwire" },
      { value: "akamai_blog_article", label: "Trade Media: Akamai Blog" },
      { value: "ritholtz_article", label: "Trade Media: The Big Picture" },
      { value: "ft_portfolios_market_commentary", label: "Trade Media: First Trust Market Commentary" },
      { value: "liberty_street_economics_article", label: "Trade Media: Liberty Street Economics" },
      { value: "wealth_of_common_sense_article", label: "Trade Media: A Wealth of Common Sense" },
      { value: "wsj_dow_jones", label: "WSJ / Dow Jones RSS" },
      { value: "reddit_post", label: "Reddit Post" },
      { value: "hedge_fund_letter", label: "Hedge Fund Letters" },
    ],
  },
  {
    name: "selection",
    label: "Selection",
    type: "select",
    default: "new_or_updated",
    options: [
      { value: "new_or_updated", label: "New or Updated" },
      { value: "all", label: "All (re-extract)" },
    ],
  },
  { name: "extraction_limit", label: "Extraction limit", type: "number", default: "25" },
  { name: "max_pages", label: "Listing pages to scan", type: "number", default: "5" },
  { name: "exclude_terms", label: "Exclude terms", type: "text", placeholder: "Comma-separated phrases (DOJ only)" },
  { name: "base_url", label: "Override index URL", type: "text", placeholder: "Optional" },
  { name: "include_pdfs", label: "Include PDFs (SEC TM FAQ)", type: "boolean", default: "true" },
  { name: "include_rss", label: "Use RSS supplement (FINRA)", type: "boolean", default: "true" },
];

const NEWS_INGEST_FIELDS: FieldDef[] = [
  { name: "ingest_limit", label: "Max articles to ingest", type: "number", default: "10" },
  { name: "lookback_days", label: "Lookback days override", type: "number", placeholder: "Leave blank for default" },
  { name: "query", label: "NewsAPI query override", type: "text", placeholder: "Optional" },
  { name: "max_pages", label: "Pages override", type: "number", placeholder: "Optional" },
  { name: "page_size", label: "Page size override", type: "number", placeholder: "Optional" },
  { name: "target_count", label: "Discovery target override", type: "number", placeholder: "Optional" },
  { name: "domains", label: "Domains override", type: "text", placeholder: "Optional" },
  { name: "tags_csv", label: "Tags override", type: "text", placeholder: "Optional" },
  {
    name: "selection",
    label: "Selection",
    type: "select",
    default: "new_or_updated",
    options: [
      { value: "new_or_updated", label: "New or Updated" },
      { value: "all", label: "All" },
    ],
  },
];

const YOUTUBE_VIDEO_FIELDS: FieldDef[] = [
  { name: "video_url", label: "Individual YouTube video URL", type: "text", placeholder: "https://www.youtube.com/watch?v=..." },
];

const RULE_COMMENT_FIELDS: FieldDef[] = [
  {
    name: "source_type",
    label: "Source type",
    type: "select",
    default: "sec_rule_page",
    options: [
      { value: "sec_rule_page", label: "SEC Rule Page URL" },
      { value: "sec_comment_url", label: "SEC Comment File URL" },
      { value: "finra_rule_page", label: "FINRA Rule Page URL" },
      { value: "finra_comment_url", label: "FINRA Comment File URL" },
    ],
  },
  { name: "source_url", label: "Rule or comment URL", type: "text", placeholder: "https://www.sec.gov/... or https://www.finra.org/..." },
  { name: "monitor_days", label: "Monitor days", type: "number", default: "95" },
  { name: "extraction_limit", label: "Extraction limit", type: "number", default: "50" },
  { name: "enrich_limit", label: "Enrichment limit", type: "number", default: "50" },
];


const TRENDS_FIELDS: FieldDef[] = [
  { name: "min_mentions", label: "Min tag mentions", type: "number", default: "5" },
  { name: "dry_run", label: "Dry run (skip OpenAI calls)", type: "boolean", default: "false" },
];

function fmtNumber(value: number | undefined): string {
  return Number(value || 0).toLocaleString();
}

function fmtDateTime(value: string | undefined): string {
  if (!value) {
    return "Never";
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? value
    : parsed.toLocaleString("en-US", { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" });
}

function sourceHealthStatusClass(status: string | undefined): string {
  if (status === "success") {
    return "border-[color:rgba(65,211,157,0.45)] bg-[color:rgba(65,211,157,0.08)] text-[#41d39d]";
  }
  if (status === "partial") {
    return "border-[color:rgba(255,199,95,0.45)] bg-[color:rgba(255,199,95,0.08)] text-[color:var(--warn)]";
  }
  if (status === "failed") {
    return "border-[color:rgba(255,107,127,0.45)] bg-[color:rgba(255,107,127,0.08)] text-[color:var(--danger)]";
  }
  return "border-[color:var(--line)] bg-[color:rgba(255,255,255,0.03)] text-[color:var(--ink-faint)]";
}

function sourceHealthAction(source: SourceHealthSource): string {
  const category = source.last_error_category || "";
  if (category === "blocked_403") return "Try proxy first, then rotate or replace source.";
  if (category === "rate_limited_429") return "Back off cadence or add source-specific throttling.";
  if (category === "stale_404") return "Refresh the URL or retire the connector.";
  if (category === "proxy_tunnel" || category === "network_tls") return "Check proxy/TLS path and retry.";
  if (category === "parser") return "Fix parser or feed format handling.";
  if (category === "auth") return "Check API credentials.";
  if (category === "model_access") return "Check model name and DeepSeek access.";
  if (category === "no_discovery" || category === "no_new_items") return "Watch if this source is expected to be active.";
  if (source.last_status === "failed") return "Investigate latest error.";
  if (source.last_status === "partial") return "Review failed item sample.";
  return "No action.";
}

/* ─── Knowledge Index ──────────────────────────────────────────────── */
function KnowledgeIndexSection() {
  const [orgs, setOrgs] = useState<OrgIndexStatus[]>([]);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dispatching, setDispatching] = useState<string | null>(null);
  const [dispatchStatus, setDispatchStatus] = useState<Record<string, "ok" | "error">>({});

  useEffect(() => {
    fetch("/api/admin/knowledge-index/status")
      .then((r) => r.json())
      .then((d) => {
        if (d.ok) { setOrgs(d.data.orgs); setUpdatedAt(d.data.updated_at); }
        else setError(d.error);
      })
      .catch(() => setError("Network error"))
      .finally(() => setLoading(false));
  }, []);

  async function dispatch(workflow: string, inputs: Record<string, string>) {
    const key = `${workflow}:${inputs.org ?? "all"}:${inputs.force_rebuild ?? "false"}`;
    setDispatching(key);
    try {
      const res = await fetch("/api/admin/workflow", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workflow, inputs }),
      });
      const d = await res.json().catch(() => ({ ok: false })) as { ok: boolean };
      setDispatchStatus((p) => ({ ...p, [key]: (res.ok && d.ok) ? "ok" : "error" }));
    } catch {
      setDispatchStatus((p) => ({ ...p, [key]: "error" }));
    } finally {
      setDispatching(null);
    }
  }

  const totalCorpus = orgs.reduce((s, o) => s + o.corpus_count, 0);
  const totalIndexed = orgs.reduce((s, o) => s + o.indexed_count, 0);
  const totalPending = orgs.reduce((s, o) => s + o.pending, 0);

  return (
    <section className="mb-8">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Knowledge Index</h2>
      <p className="mb-3 text-xs text-[color:var(--ink-faint)]">OpenAI vector store sync status per regulatory source.</p>
      <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.88)] px-4 py-4">
        {loading && <p className="text-xs text-[color:var(--ink-faint)]">Loading…</p>}
        {error && <p className="text-xs text-[color:var(--danger)]">{error}</p>}
        {!loading && !error && (
          <>
            {/* Summary row */}
            <div className="mb-4 flex flex-wrap gap-6 border-b border-[color:var(--line)] pb-4">
              <div className="flex flex-col gap-0.5">
                <span className="text-xl font-bold tabular-nums text-[color:var(--ink)]">{totalCorpus.toLocaleString()}</span>
                <span className="text-xs text-[color:var(--ink-faint)]">Corpus Docs</span>
              </div>
              <div className="flex flex-col gap-0.5">
                <span className="text-xl font-bold tabular-nums text-[#41d39d]">{totalIndexed.toLocaleString()}</span>
                <span className="text-xs text-[color:var(--ink-faint)]">Indexed</span>
              </div>
              <div className="flex flex-col gap-0.5">
                <span className={`text-xl font-bold tabular-nums ${totalPending > 0 ? "text-[color:var(--accent)]" : "text-[color:var(--ink-faint)]"}`}>{totalPending.toLocaleString()}</span>
                <span className="text-xs text-[color:var(--ink-faint)]">Pending</span>
              </div>
              {updatedAt && (
                <div className="ml-auto flex flex-col gap-0.5 text-right">
                  <span className="text-xs text-[color:var(--ink-faint)]">Last sync</span>
                  <span className="text-xs text-[color:var(--ink)]">{new Date(updatedAt).toLocaleString()}</span>
                </div>
              )}
            </div>

            {/* Per-org table */}
            <div className="mb-4 overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-[color:var(--line)] text-left text-[color:var(--ink-faint)]">
                    <th className="pb-2 pr-4 font-semibold">Org</th>
                    <th className="pb-2 pr-4 font-semibold">Vector Store ID</th>
                    <th className="pb-2 pr-4 text-right font-semibold">Corpus</th>
                    <th className="pb-2 pr-4 text-right font-semibold">Indexed</th>
                    <th className="pb-2 pr-4 text-right font-semibold">Pending</th>
                    <th className="pb-2 pr-4 font-semibold">Last Sync</th>
                    <th className="pb-2 font-semibold"></th>
                  </tr>
                </thead>
                <tbody>
                  {orgs.map((org) => {
                    const syncKey = `knowledge-index-sync.yml:${org.org_key}:false`;
                    const rebuildKey = `knowledge-index-sync.yml:${org.org_key}:true`;
                    return (
                      <tr key={org.org_key} className="border-b border-[color:rgba(255,255,255,0.04)]">
                        <td className="py-2 pr-4 font-medium text-[color:var(--ink)]">{org.org_label}</td>
                        <td className="py-2 pr-4 font-mono text-[color:var(--ink-faint)]">
                          {org.vector_store_id ? (
                            <span title={org.vector_store_id}>{org.vector_store_id.slice(0, 18)}…</span>
                          ) : <span className="italic">none</span>}
                        </td>
                        <td className="py-2 pr-4 text-right tabular-nums text-[color:var(--ink)]">{org.corpus_count}</td>
                        <td className="py-2 pr-4 text-right tabular-nums text-[#41d39d]">{org.indexed_count}</td>
                        <td className={`py-2 pr-4 text-right tabular-nums ${org.pending > 0 ? "text-[color:var(--accent)]" : "text-[color:var(--ink-faint)]"}`}>{org.pending}</td>
                        <td className="py-2 pr-4 text-[color:var(--ink-faint)]">
                          {org.last_sync ? (
                            <span className={org.last_sync.status === "completed" ? "text-[#41d39d]" : "text-[color:var(--warn)]"}>
                              {org.last_sync.sync_mode} · +{org.last_sync.uploaded} -{org.last_sync.deleted}
                              {org.last_sync.failed_count > 0 && <span className="text-[color:var(--danger)]"> · {org.last_sync.failed_count} failed</span>}
                            </span>
                          ) : <span className="italic">never</span>}
                        </td>
                        <td className="py-2">
                          <div className="flex gap-2">
                            <button
                              type="button"
                              disabled={dispatching === syncKey}
                              onClick={() => dispatch("knowledge-index-sync.yml", { org: org.org_key, force_rebuild: "false" })}
                              className="rounded-lg border border-[color:var(--line)] px-2 py-1 text-xs hover:border-[color:var(--accent)] disabled:opacity-40"
                              title="Incremental sync"
                            >
                              {dispatching === syncKey ? "…" : dispatchStatus[syncKey] === "ok" ? "✓" : "Sync"}
                            </button>
                            <button
                              type="button"
                              disabled={dispatching === rebuildKey}
                              onClick={() => dispatch("knowledge-index-sync.yml", { org: org.org_key, force_rebuild: "true" })}
                              className="rounded-lg border border-[color:rgba(255,107,127,0.4)] px-2 py-1 text-xs text-[color:var(--danger)] hover:bg-[color:rgba(255,107,127,0.1)] disabled:opacity-40"
                              title="Force rebuild"
                            >
                              {dispatching === rebuildKey ? "…" : "Rebuild"}
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Sync all buttons */}
            <div className="flex flex-wrap items-center gap-3 border-t border-[color:var(--line)] pt-4">
              <button
                type="button"
                disabled={!!dispatching}
                onClick={() => dispatch("knowledge-index-sync.yml", { org: "", force_rebuild: "false" })}
                className="btn-solid rounded-xl px-5 py-2 text-sm font-semibold disabled:opacity-40"
              >
                {dispatching?.startsWith("knowledge-index-sync.yml::false") ? "Dispatching…" : "Sync All Orgs"}
              </button>
              <button
                type="button"
                disabled={!!dispatching}
                onClick={() => dispatch("knowledge-index-sync.yml", { org: "", force_rebuild: "true" })}
                className="rounded-xl border border-[color:rgba(255,107,127,0.4)] bg-[color:rgba(255,107,127,0.1)] px-5 py-2 text-sm font-semibold text-[color:var(--danger)] hover:bg-[color:rgba(255,107,127,0.2)] disabled:opacity-40"
              >
                Force Rebuild All
              </button>
              <span className="ml-auto">
                <JobStatusBadge workflowFile="knowledge-index-sync.yml" />
              </span>
            </div>
          </>
        )}
      </div>
    </section>
  );
}

/* ─── RSS Feed Manager ─────────────────────────────────────────────── */
function ConnectorAuditSection() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/metrics")
      .then((res) => res.json())
      .then((payload) => {
        if (payload?.ok) {
          setMetrics(payload.data as MetricsData);
          setError("");
        } else {
          setError(String(payload?.error || "Failed to load metrics."));
        }
      })
      .catch(() => setError("Network error while loading connector metrics."))
      .finally(() => setLoading(false));
  }, []);

  const newsapi = metrics?.connector_audit?.newsapi;
  const feedDocs = metrics?.connector_audit?.feed_documents;
  const newest = newsapi?.newest;
  const newestMs = Date.parse(newest?.published_at || "");
  const stale = Number.isFinite(newestMs) && Date.now() - newestMs > 7 * 24 * 60 * 60 * 1000;

  return (
    <section className="mb-8">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Connector Audit</h2>
      <p className="mb-3 text-xs text-[color:var(--ink-faint)]">Live corpus checks for NewsAPI presence, freshness, feed inclusion, and publisher mix.</p>

      <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.88)] px-4 py-4">
        {loading ? <p className="text-xs text-[color:var(--ink-faint)]">Loading connector metrics...</p> : null}
        {error ? <p className="text-xs text-[color:var(--danger)]">{error}</p> : null}
        {!loading && !error ? (
          <div className="grid gap-4">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
              <article className="rounded-lg border border-[color:var(--line)] px-3 py-3">
                <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[color:var(--ink-faint)]">NewsAPI Docs</p>
                <p className="mt-1 text-xl font-semibold text-[color:var(--ink)]">{fmtNumber(newsapi?.total)}</p>
              </article>
              <article className="rounded-lg border border-[color:var(--line)] px-3 py-3">
                <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[color:var(--ink-faint)]">In Feed</p>
                <p className="mt-1 text-xl font-semibold text-[#41d39d]">{fmtNumber(newsapi?.in_feed)}</p>
              </article>
              <article className="rounded-lg border border-[color:var(--line)] px-3 py-3">
                <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[color:var(--ink-faint)]">24h</p>
                <p className="mt-1 text-xl font-semibold text-[color:var(--ink)]">{fmtNumber(newsapi?.recent_24h)}</p>
              </article>
              <article className="rounded-lg border border-[color:var(--line)] px-3 py-3">
                <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[color:var(--ink-faint)]">7d</p>
                <p className="mt-1 text-xl font-semibold text-[color:var(--ink)]">{fmtNumber(newsapi?.recent_7d)}</p>
              </article>
              <article className="rounded-lg border border-[color:var(--line)] px-3 py-3">
                <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[color:var(--ink-faint)]">30d</p>
                <p className="mt-1 text-xl font-semibold text-[color:var(--ink)]">{fmtNumber(newsapi?.recent_30d)}</p>
              </article>
            </div>

            <div className={`rounded-lg border px-3 py-3 ${stale ? "border-[color:rgba(255,199,95,0.42)] bg-[color:rgba(255,199,95,0.08)]" : "border-[color:var(--line)]"}`}>
              <div className="flex flex-wrap items-center gap-2">
                <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">Newest NewsAPI Article</p>
                <JobStatusBadge workflowFile="financial-news-daily.yml" />
              </div>
              {newest ? (
                <div className="mt-2 grid gap-1 text-sm">
                  <a href={newest.url} target="_blank" rel="noopener noreferrer" className="font-semibold text-[color:var(--ink)] hover:text-[color:var(--accent)]">
                    {newest.title || "Untitled article"}
                  </a>
                  <p className="text-xs text-[color:var(--ink-faint)]">
                    {newest.source_name || "Unknown source"} - {fmtDateTime(newest.published_at)} - {newest.extraction_mode || "unknown extraction mode"}
                  </p>
                  {stale ? <p className="text-xs text-[color:var(--warn)]">Newest stored NewsAPI article is more than 7 days old. Check the daily workflow run and GCS data source.</p> : null}
                </div>
              ) : (
                <p className="mt-2 text-xs text-[color:var(--danger)]">No NewsAPI documents found in the corpus.</p>
              )}
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <p className="mb-2 text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">Top NewsAPI Publishers</p>
                <div className="grid gap-1 text-xs">
                  {(newsapi?.by_source || []).slice(0, 8).map((row) => (
                    <div key={row.source_name || "Unknown"} className="flex justify-between gap-3 rounded border border-[color:rgba(255,255,255,0.05)] px-2 py-1.5">
                      <span className="truncate text-[color:var(--ink)]">{row.source_name || "Unknown"}</span>
                      <span className="tabular-nums text-[color:var(--ink-faint)]">{fmtNumber(row.count)}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <p className="mb-2 text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">Feed Document Mix</p>
                <div className="grid gap-1 text-xs">
                  {(feedDocs?.by_source_kind || []).slice(0, 8).map((row) => (
                    <div key={row.source_kind} className="flex justify-between gap-3 rounded border border-[color:rgba(255,255,255,0.05)] px-2 py-1.5">
                      <span className="truncate text-[color:var(--ink)]">{row.source_kind}</span>
                      <span className="tabular-nums text-[color:var(--ink-faint)]">{fmtNumber(row.count)}</span>
                    </div>
                  ))}
                </div>
                <p className="mt-2 text-[10px] text-[color:var(--ink-faint)]">Feed document set: {fmtNumber(feedDocs?.total)} corpus-backed items selected for display.</p>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </section>
  );
}

function SourceHealthSection() {
  const [data, setData] = useState<SourceHealthData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/admin/source-health")
      .then((res) => res.json())
      .then((payload: { ok: boolean; data?: SourceHealthData; error?: string }) => {
        if (payload?.ok && payload.data) {
          setData({
            ...payload.data,
            sources: Array.isArray(payload.data.sources) ? payload.data.sources : [],
            runs: Array.isArray(payload.data.runs) ? payload.data.runs : [],
          });
          setError("");
        } else {
          setError(String(payload?.error || "Failed to load source health."));
        }
      })
      .catch(() => setError("Network error while loading source health."))
      .finally(() => setLoading(false));
  }, []);

  const sources = data?.sources || [];
  const report = data?.latest_report?.report;
  const sourceTime = (value: string | undefined) => {
    const parsed = Date.parse(value || "");
    return Number.isFinite(parsed) ? parsed : 0;
  };
  const failingCount = report?.failing_sources?.length ?? sources.filter((source) => source.last_status === "failed" || Number(source.consecutive_failures || 0) > 0).length;
  const staleCount = report?.stale_sources?.length ?? 0;
  const quietCount = report?.quiet_sources?.length ?? 0;
  const recentRunCount = report?.recent_run_count ?? data?.runs.length ?? 0;
  const recentFailedCount = report?.recent_failed_run_count ?? data?.runs.filter((run) => run.status === "failed").length ?? 0;
  const rankedSources = [...sources]
    .sort((a, b) => {
      const aScore = a.last_status === "failed" ? 3 : a.last_status === "partial" ? 2 : Number(a.consecutive_failures || 0) > 0 ? 1 : 0;
      const bScore = b.last_status === "failed" ? 3 : b.last_status === "partial" ? 2 : Number(b.consecutive_failures || 0) > 0 ? 1 : 0;
      return bScore - aScore || sourceTime(b.last_run_at) - sourceTime(a.last_run_at) || a.source_key.localeCompare(b.source_key);
    })
    .slice(0, 15);
  const reportRuns = report?.recent_runs?.length ? report.recent_runs : data?.runs || [];
  const runsNeedingReview = reportRuns
    .filter((run) => run.status === "failed" || run.status === "partial" || (run.error_category && run.error_category !== "none"))
    .slice(0, 8);
  const categories = report?.error_categories || {};

  return (
    <section className="mb-8">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Source Health</h2>
      <p className="mb-3 text-xs text-[color:var(--ink-faint)]">Daily source failure log with structured categories, recent run counts, and DeepSeek review notes.</p>

      <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.88)] px-4 py-4">
        {loading ? <p className="text-xs text-[color:var(--ink-faint)]">Loading source health...</p> : null}
        {error ? <p className="text-xs text-[color:var(--danger)]">{error}</p> : null}
        {!loading && !error && data ? (
          <div className="grid gap-4">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
              <article className="rounded-lg border border-[color:var(--line)] px-3 py-3">
                <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[color:var(--ink-faint)]">Tracked Sources</p>
                <p className="mt-1 text-xl font-semibold text-[color:var(--ink)]">{fmtNumber(sources.length)}</p>
              </article>
              <article className="rounded-lg border border-[color:var(--line)] px-3 py-3">
                <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[color:var(--ink-faint)]">Recent Runs</p>
                <p className="mt-1 text-xl font-semibold text-[color:var(--ink)]">{fmtNumber(recentRunCount)}</p>
              </article>
              <article className="rounded-lg border border-[color:rgba(255,107,127,0.35)] px-3 py-3">
                <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[color:var(--ink-faint)]">Failing</p>
                <p className={`mt-1 text-xl font-semibold ${failingCount ? "text-[color:var(--danger)]" : "text-[color:var(--ink-faint)]"}`}>{fmtNumber(failingCount)}</p>
              </article>
              <article className="rounded-lg border border-[color:rgba(255,199,95,0.35)] px-3 py-3">
                <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[color:var(--ink-faint)]">Stale</p>
                <p className={`mt-1 text-xl font-semibold ${staleCount ? "text-[color:var(--warn)]" : "text-[color:var(--ink-faint)]"}`}>{fmtNumber(staleCount)}</p>
              </article>
              <article className="rounded-lg border border-[color:var(--line)] px-3 py-3">
                <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[color:var(--ink-faint)]">Quiet</p>
                <p className="mt-1 text-xl font-semibold text-[color:var(--ink-faint)]">{fmtNumber(quietCount)}</p>
              </article>
            </div>

            <div className="flex flex-wrap items-center gap-3 rounded-lg border border-[color:var(--line)] px-3 py-2 text-xs text-[color:var(--ink-faint)]">
              <span>Log updated: {fmtDateTime(data.updated_at)}</span>
              <span>Recent failures: {fmtNumber(recentFailedCount)}</span>
              {Object.keys(categories).length ? (
                <span>
                  Categories: {Object.entries(categories).map(([key, count]) => `${key} ${count}`).join(", ")}
                </span>
              ) : null}
            </div>

            {data.latest_report?.ai_review ? (
              <div className="rounded-lg border border-[color:rgba(83,210,255,0.25)] bg-[color:rgba(83,210,255,0.05)] px-3 py-3">
                <p className="mb-2 text-[10px] font-bold uppercase tracking-[0.12em] text-[color:var(--accent)]">DeepSeek Review</p>
                <p className="whitespace-pre-wrap text-xs leading-relaxed text-[color:var(--ink)]">{data.latest_report.ai_review}</p>
              </div>
            ) : null}

            <div className="overflow-x-auto">
              <table className="w-full min-w-[860px] text-xs">
                <thead>
                  <tr className="border-b border-[color:var(--line)] text-left text-[color:var(--ink-faint)]">
                    <th className="pb-2 pr-4 font-semibold">Source</th>
                    <th className="pb-2 pr-4 font-semibold">Status</th>
                    <th className="pb-2 pr-4 font-semibold">Last Run</th>
                    <th className="pb-2 pr-4 text-right font-semibold">Counts</th>
                    <th className="pb-2 pr-4 font-semibold">Category</th>
                    <th className="pb-2 font-semibold">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {rankedSources.map((source) => {
                    const counts = source.last_counts || {};
                    return (
                      <tr key={source.source_key} className="border-b border-[color:rgba(255,255,255,0.04)]">
                        <td className="py-2 pr-4 font-mono text-[color:var(--ink)]">{source.source_key}</td>
                        <td className="py-2 pr-4">
                          <span className={`rounded-full border px-2 py-1 text-[10px] font-bold uppercase tracking-[0.08em] ${sourceHealthStatusClass(source.last_status)}`}>
                            {source.last_status || "unknown"}
                          </span>
                        </td>
                        <td className="py-2 pr-4 text-[color:var(--ink-faint)]">{fmtDateTime(source.last_run_at)}</td>
                        <td className="py-2 pr-4 text-right tabular-nums text-[color:var(--ink-faint)]">
                          {fmtNumber(counts.discovered)} found / {fmtNumber(counts.processed)} processed / {fmtNumber(counts.failed)} failed
                        </td>
                        <td className="py-2 pr-4 font-mono text-[color:var(--ink-faint)]">{source.last_error_category || "none"}</td>
                        <td className="py-2 text-[color:var(--ink-faint)]">{sourceHealthAction(source)}</td>
                      </tr>
                    );
                  })}
                  {rankedSources.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="py-6 text-center text-[color:var(--ink-faint)]">No source health records yet.</td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>

            {runsNeedingReview.length ? (
              <div className="rounded-lg border border-[color:var(--line)] px-3 py-3">
                <p className="mb-2 text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">Recent Runs Needing Review</p>
                <div className="grid gap-2">
                  {runsNeedingReview.map((run) => (
                    <div key={`${run.id || run.source_key}-${run.ran_at || ""}`} className="grid gap-1 rounded border border-[color:rgba(255,255,255,0.05)] px-2 py-2 text-xs">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-mono text-[color:var(--ink)]">{run.source_key}</span>
                        <span className={`rounded-full border px-2 py-0.5 text-[10px] font-bold uppercase ${sourceHealthStatusClass(run.status)}`}>{run.status || "unknown"}</span>
                        <span className="text-[color:var(--ink-faint)]">{fmtDateTime(run.ran_at)}</span>
                        <span className="font-mono text-[color:var(--ink-faint)]">{run.error_category || "none"}</span>
                      </div>
                      {run.sample_error ? <p className="line-clamp-2 text-[color:var(--ink-faint)]">{run.sample_error}</p> : null}
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </section>
  );
}

function FeedManagerSection() {
  const [feeds, setFeeds] = useState<RssFeed[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [newLabel, setNewLabel] = useState("");
  const [newUrl, setNewUrl] = useState("");
  const [adding, setAdding] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/admin/feeds")
      .then((r) => r.json())
      .then((d) => { if (d.ok) setFeeds(d.data.feeds); else setError(d.error); })
      .catch(() => setError("Network error"))
      .finally(() => setLoading(false));
  }, []);

  async function handleAdd() {
    if (!newLabel.trim() || !newUrl.trim()) return;
    setAdding(true);
    setAddError(null);
    try {
      const res = await fetch("/api/admin/feeds", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: newLabel.trim(), feedUrl: newUrl.trim() }),
      });
      const d = await res.json();
      if (d.ok) { setFeeds((p) => [...p, d.data.feed]); setNewLabel(""); setNewUrl(""); }
      else setAddError(d.error);
    } catch { setAddError("Network error"); }
    finally { setAdding(false); }
  }

  async function handleToggle(feed: RssFeed) {
    try {
      const res = await fetch(`/api/admin/feeds/${feed.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ active: !feed.active }),
      });
      if (!res.ok) { const d = await res.json().catch(() => ({})); setError(d.error ?? "Toggle failed"); return; }
      setFeeds((p) => p.map((f) => f.id === feed.id ? { ...f, active: !f.active } : f));
    } catch { setError("Network error"); }
  }

  async function handleDelete(id: number) {
    try {
      const res = await fetch(`/api/admin/feeds/${id}`, { method: "DELETE" });
      if (!res.ok) { const d = await res.json().catch(() => ({})); setError(d.error ?? "Delete failed"); return; }
      setFeeds((p) => p.filter((f) => f.id !== id));
    } catch { setError("Network error"); }
  }

  return (
    <section className="mb-8">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">RSS Feeds</h2>
      <p className="mb-3 text-xs text-[color:var(--ink-faint)]">Manage Intel Feed RSS sources. Changes apply when each source is due on the scheduled refresh.</p>
      <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.88)] px-4 py-4">
        {loading && <p className="text-xs text-[color:var(--ink-faint)]">Loading…</p>}
        {error && <p className="text-xs text-[color:var(--danger)]">{error}</p>}
        {!loading && feeds.length === 0 && <p className="text-xs text-[color:var(--ink-faint)]">No feeds configured.</p>}
        <ul className="space-y-2">
          {feeds.map((f) => (
            <li key={f.id} className="flex items-center gap-3">
              <label className="flex cursor-pointer items-center gap-2">
                <input
                  type="checkbox"
                  checked={f.active}
                  onChange={() => handleToggle(f)}
                  className="h-4 w-4 rounded accent-[color:var(--accent)]"
                />
              </label>
              <span className="min-w-0 flex-1">
                <span className="block text-sm font-medium text-[color:var(--ink)]">{f.label}</span>
                <span className="block truncate text-xs text-[color:var(--ink-faint)]">{f.feed_url}</span>
                <span className="block text-xs text-[color:var(--ink-faint)]">
                  Every {f.refresh_interval_minutes ?? 10} min{f.last_refresh_at ? ` | Last refresh ${new Date(f.last_refresh_at).toLocaleString()}` : ""}
                </span>
                {!!f.last_error && (
                  <span className="block truncate text-xs text-[color:var(--danger)]" title={f.last_error}>
                    {f.consecutive_failures ? `${f.consecutive_failures} failed refresh(es): ` : "Last refresh failed: "}
                    {f.last_error}
                  </span>
                )}
              </span>
              <button
                type="button"
                onClick={() => handleDelete(f.id)}
                className="flex-shrink-0 rounded-lg border border-[color:rgba(255,107,127,0.4)] bg-[color:rgba(255,107,127,0.1)] px-3 py-1 text-xs font-semibold text-[color:var(--danger)] hover:bg-[color:rgba(255,107,127,0.2)]"
              >
                Remove
              </button>
            </li>
          ))}
        </ul>
        <div className="mt-4 border-t border-[color:var(--line)] pt-4">
          <p className="mb-2 text-xs font-semibold text-[color:var(--ink-faint)]">Add Feed</p>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-[1fr_2fr_auto]">
            <input
              type="text"
              value={newLabel}
              onChange={(e) => setNewLabel(e.target.value)}
              placeholder="Label (e.g. WSJ Tech)"
              className="form-control px-2 py-1.5 text-sm"
            />
            <input
              type="url"
              value={newUrl}
              onChange={(e) => setNewUrl(e.target.value)}
              placeholder="Feed URL"
              className="form-control px-2 py-1.5 text-sm"
            />
            <button
              type="button"
              onClick={handleAdd}
              disabled={adding || !newLabel.trim() || !newUrl.trim()}
              className="btn-solid rounded-xl px-4 py-1.5 text-sm disabled:opacity-40"
            >
              {adding ? "Adding…" : "Add"}
            </button>
          </div>
          {addError && <p className="mt-1 text-xs text-[color:var(--danger)]">{addError}</p>}
        </div>
      </div>
    </section>
  );
}

function YouTubeChannelManagerSection() {
  const [channels, setChannels] = useState<YouTubeChannelConfig[]>([]);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [newLabel, setNewLabel] = useState("");
  const [newRef, setNewRef] = useState("");
  const [newExtractionLimit, setNewExtractionLimit] = useState("2");
  const [newEnrichLimit, setNewEnrichLimit] = useState("2");
  const [adding, setAdding] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);
  const [dispatching, setDispatching] = useState<string | null>(null);
  const [dispatchStatus, setDispatchStatus] = useState<Record<string, "ok" | "error">>({});
  const [dispatchError, setDispatchError] = useState<Record<string, string>>({});

  async function loadChannels() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/admin/youtube-channels");
      const data = await res.json() as {
        ok: boolean;
        data?: { channels: YouTubeChannelConfig[]; updated_at?: string };
        error?: string;
      };
      if (data.ok && data.data) {
        setChannels(data.data.channels || []);
        setUpdatedAt(data.data.updated_at || null);
      } else {
        setError(data.error || "Failed to load YouTube channels.");
      }
    } catch {
      setError("Network error");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadChannels();
  }, []);

  async function handleAdd() {
    if (!newLabel.trim() || !newRef.trim()) return;
    setAdding(true);
    setAddError(null);
    try {
      const res = await fetch("/api/admin/youtube-channels", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          label: newLabel.trim(),
          channelRef: newRef.trim(),
          extractionLimit: newExtractionLimit,
          enrichLimit: newEnrichLimit,
          maxPages: 1,
        }),
      });
      const data = await res.json() as { ok: boolean; data?: { channels: YouTubeChannelConfig[]; updated_at?: string }; error?: string };
      if (data.ok && data.data) {
        setChannels(data.data.channels || []);
        setUpdatedAt(data.data.updated_at || null);
        setNewLabel("");
        setNewRef("");
        setNewExtractionLimit("2");
        setNewEnrichLimit("2");
      } else {
        setAddError(data.error || "Failed to add YouTube channel.");
      }
    } catch {
      setAddError("Network error");
    } finally {
      setAdding(false);
    }
  }

  async function handleToggle(channel: YouTubeChannelConfig) {
    setError(null);
    try {
      const res = await fetch(`/api/admin/youtube-channels/${encodeURIComponent(channel.id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ active: !channel.active }),
      });
      const data = await res.json() as { ok: boolean; data?: { channels: YouTubeChannelConfig[]; updated_at?: string }; error?: string };
      if (data.ok && data.data) {
        setChannels(data.data.channels || []);
        setUpdatedAt(data.data.updated_at || null);
      } else {
        setError(data.error || "Toggle failed");
      }
    } catch {
      setError("Network error");
    }
  }

  async function handleDelete(channel: YouTubeChannelConfig) {
    setError(null);
    try {
      const res = await fetch(`/api/admin/youtube-channels/${encodeURIComponent(channel.id)}`, { method: "DELETE" });
      const data = await res.json() as { ok: boolean; data?: { channels: YouTubeChannelConfig[]; updated_at?: string }; error?: string };
      if (data.ok && data.data) {
        setChannels(data.data.channels || []);
        setUpdatedAt(data.data.updated_at || null);
      } else {
        setError(data.error || "Delete failed");
      }
    } catch {
      setError("Network error");
    }
  }

  async function dispatchYouTubeRun(key: string, inputs: Record<string, string>) {
    setDispatching(key);
    setDispatchStatus((prev) => ({ ...prev, [key]: "ok" }));
    setDispatchError((prev) => ({ ...prev, [key]: "" }));
    try {
      const res = await fetch("/api/admin/workflow", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workflow: "sec-youtube-videos-daily.yml", inputs }),
      });
      const data = await res.json().catch(() => ({ ok: false })) as { ok?: boolean; error?: string };
      if (!res.ok || !data.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      setDispatchStatus((prev) => ({ ...prev, [key]: "ok" }));
    } catch (err) {
      setDispatchStatus((prev) => ({ ...prev, [key]: "error" }));
      setDispatchError((prev) => ({ ...prev, [key]: err instanceof Error ? err.message : "Dispatch failed" }));
    } finally {
      setDispatching(null);
    }
  }

  function runSingleChannel(channel: YouTubeChannelConfig) {
    void dispatchYouTubeRun(channel.id, {
      channel_ref: channel.channel_ref,
      extraction_limit: String(channel.extraction_limit || 2),
      enrich_limit: String(channel.enrich_limit || 2),
      max_pages: String(channel.max_pages || 1),
    });
  }

  const activeCount = channels.filter((channel) => channel.active).length;

  return (
    <section className="mb-8">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
        YouTube Continuous Channels
      </h2>
      <p className="mb-3 text-xs text-[color:var(--ink-faint)]">
        Add channel URLs, handles, channel IDs, or uploads RSS URLs here. Active channels are scanned by the daily YouTube workflow.
      </p>
      <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.88)] px-4 py-4">
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => dispatchYouTubeRun("all", {})}
            disabled={dispatching === "all" || activeCount === 0}
            className="btn-solid rounded-xl px-4 py-2 text-sm font-semibold disabled:opacity-40"
          >
            {dispatching === "all" ? "Dispatching..." : `Run All Active Channels (${activeCount})`}
          </button>
          <span className="text-xs text-[color:var(--ink-faint)]">
            Daily schedule uses this same active list{updatedAt ? ` | Config saved ${new Date(updatedAt).toLocaleString()}` : ""}.
          </span>
          <span className="ml-auto">
            <JobStatusBadge workflowFile="sec-youtube-videos-daily.yml" />
          </span>
        </div>
        {dispatchStatus.all === "error" && (
          <p className="mb-3 text-xs text-[color:var(--danger)]">Run failed: {dispatchError.all || "Dispatch failed"}</p>
        )}
        {loading && <p className="text-xs text-[color:var(--ink-faint)]">Loading...</p>}
        {error && <p className="text-xs text-[color:var(--danger)]">{error}</p>}
        {!loading && channels.length === 0 && <p className="text-xs text-[color:var(--ink-faint)]">No YouTube channels configured.</p>}
        <ul className="space-y-2">
          {channels.map((channel) => (
            <li key={channel.id} className="flex flex-col gap-2 rounded-lg border border-[color:rgba(82,120,160,0.25)] bg-[color:rgba(4,13,24,0.45)] px-3 py-3 sm:flex-row sm:items-center">
              <label className="flex cursor-pointer items-center gap-2">
                <input
                  type="checkbox"
                  checked={channel.active}
                  onChange={() => handleToggle(channel)}
                  className="h-4 w-4 rounded accent-[color:var(--accent)]"
                />
              </label>
              <span className="min-w-0 flex-1">
                <span className="block text-sm font-medium text-[color:var(--ink)]">{channel.label}</span>
                <span className="block truncate text-xs text-[color:var(--ink-faint)]">{channel.channel_ref}</span>
                <span className="block text-xs text-[color:var(--ink-faint)]">
                  {channel.extraction_limit || 2} latest videos | enrich {channel.enrich_limit || 2}
                  {channel.last_run_at ? ` | Last run ${new Date(channel.last_run_at).toLocaleString()}` : ""}
                  {channel.last_status ? ` | ${channel.last_status}` : ""}
                </span>
                {channel.last_error ? <span className="block text-xs text-[color:var(--danger)]">{channel.last_error}</span> : null}
                {dispatchStatus[channel.id] === "error" ? (
                  <span className="block text-xs text-[color:var(--danger)]">Run failed: {dispatchError[channel.id] || "Dispatch failed"}</span>
                ) : null}
              </span>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => runSingleChannel(channel)}
                  disabled={dispatching === channel.id}
                  className="rounded-lg border border-[color:rgba(79,213,255,0.35)] bg-[color:rgba(79,213,255,0.08)] px-3 py-1 text-xs font-semibold text-[color:var(--accent)] hover:bg-[color:rgba(79,213,255,0.15)] disabled:opacity-40"
                >
                  {dispatching === channel.id ? "Running..." : "Run Now"}
                </button>
                <button
                  type="button"
                  onClick={() => handleDelete(channel)}
                  disabled={channel.id === "sec_views"}
                  className="rounded-lg border border-[color:rgba(255,107,127,0.4)] bg-[color:rgba(255,107,127,0.1)] px-3 py-1 text-xs font-semibold text-[color:var(--danger)] hover:bg-[color:rgba(255,107,127,0.2)] disabled:cursor-not-allowed disabled:opacity-35"
                  title={channel.id === "sec_views" ? "Deactivate the default SEC channel instead of removing it." : undefined}
                >
                  Remove
                </button>
              </div>
            </li>
          ))}
        </ul>
        <div className="mt-4 border-t border-[color:var(--line)] pt-4">
          <p className="mb-2 text-xs font-semibold text-[color:var(--ink-faint)]">Add Continuous Channel</p>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-[1fr_2fr_120px_120px_auto]">
            <input
              type="text"
              value={newLabel}
              onChange={(e) => setNewLabel(e.target.value)}
              placeholder="Label"
              className="form-control px-2 py-1.5 text-sm"
            />
            <input
              type="text"
              value={newRef}
              onChange={(e) => setNewRef(e.target.value)}
              placeholder="Channel URL, @handle, channel ID, or uploads RSS"
              className="form-control px-2 py-1.5 text-sm"
            />
            <input
              type="number"
              value={newExtractionLimit}
              onChange={(e) => setNewExtractionLimit(e.target.value)}
              min={1}
              max={50}
              title="Latest videos to scan per run"
              className="form-control px-2 py-1.5 text-sm"
            />
            <input
              type="number"
              value={newEnrichLimit}
              onChange={(e) => setNewEnrichLimit(e.target.value)}
              min={1}
              max={50}
              title="New transcript docs to enrich per run"
              className="form-control px-2 py-1.5 text-sm"
            />
            <button
              type="button"
              onClick={handleAdd}
              disabled={adding || !newLabel.trim() || !newRef.trim()}
              className="btn-solid rounded-xl px-4 py-1.5 text-sm disabled:opacity-40"
            >
              {adding ? "Adding..." : "Add"}
            </button>
          </div>
          <p className="mt-1 text-[10px] text-[color:var(--ink-faint)]">
            The numeric fields are latest videos to scan and new transcripts to enrich per run. Defaults are 2 and 2.
          </p>
          {addError && <p className="mt-1 text-xs text-[color:var(--danger)]">{addError}</p>}
        </div>
      </div>
    </section>
  );
}

/* ─── Topic Rules Manager ──────────────────────────────────────────── */
type XRefreshResult = {
  inserted: number;
  feeds: Array<{ feedKey: string; label: string; fetched: number; matched: number; filtered: number; inserted: number; provider?: string; error?: string }>;
  analysis: { selected_count: number; saved_count: number; failed_count: number };
};

function XAccountManagerSection() {
  const [accounts, setAccounts] = useState<XAccountFeed[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [newAccount, setNewAccount] = useState("");
  const [newInterval, setNewInterval] = useState("180");
  const [adding, setAdding] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshLimit, setRefreshLimit] = useState("20");
  const [analysisLimit, setAnalysisLimit] = useState("10");
  const [refreshResult, setRefreshResult] = useState<XRefreshResult | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);

  async function loadAccounts() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/admin/x-accounts");
      const data = await res.json();
      if (data.ok) setAccounts(data.data.accounts);
      else setError(data.error || "Failed to load X accounts.");
    } catch {
      setError("Network error");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadAccounts();
  }, []);

  async function handleAdd() {
    if (!newAccount.trim()) return;
    setAdding(true);
    setAddError(null);
    try {
      const res = await fetch("/api/admin/x-accounts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account: newAccount.trim(), refreshIntervalMinutes: newInterval }),
      });
      const data = await res.json();
      if (data.ok) {
        setAccounts((prev) => {
          const next = data.data.account as XAccountFeed;
          const rest = prev.filter((item) => item.feed_key !== next.feed_key);
          return [...rest, next].sort((a, b) => a.label.localeCompare(b.label));
        });
        setNewAccount("");
      } else {
        setAddError(data.error || "Failed to add account.");
      }
    } catch {
      setAddError("Network error");
    } finally {
      setAdding(false);
    }
  }

  async function handleToggle(account: XAccountFeed) {
    try {
      const res = await fetch(`/api/admin/feeds/${account.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ active: !account.active }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setError(data.error ?? "Toggle failed");
        return;
      }
      setAccounts((prev) => prev.map((item) => item.id === account.id ? { ...item, active: !item.active } : item));
    } catch {
      setError("Network error");
    }
  }

  async function handleDelete(account: XAccountFeed) {
    try {
      const res = await fetch(`/api/admin/feeds/${account.id}`, { method: "DELETE" });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setError(data.error ?? "Delete failed");
        return;
      }
      setAccounts((prev) => prev.filter((item) => item.id !== account.id));
    } catch {
      setError("Network error");
    }
  }

  async function handleRefresh() {
    setRefreshing(true);
    setRefreshResult(null);
    setRefreshError(null);
    try {
      const res = await fetch("/api/admin/x-accounts/refresh", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ limit: refreshLimit, analysisLimit }),
      });
      const data = await res.json();
      if (data.ok) {
        setRefreshResult(data.data);
        await loadAccounts();
      } else {
        setRefreshError(data.error || "Refresh failed.");
      }
    } catch {
      setRefreshError("Network error");
    } finally {
      setRefreshing(false);
    }
  }

  const activeCount = accounts.filter((account) => account.active).length;

  return (
    <section className="mb-8">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">X Accounts</h2>
      <p className="mb-3 text-xs text-[color:var(--ink-faint)]">Stored account timelines use source chips like X: @SECGov and category matching from active Topic Rules.</p>
      <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.88)] px-4 py-4">
        {loading && <p className="text-xs text-[color:var(--ink-faint)]">Loading...</p>}
        {error && <p className="text-xs text-[color:var(--danger)]">{error}</p>}
        {!loading && accounts.length === 0 && <p className="text-xs text-[color:var(--ink-faint)]">No X accounts configured.</p>}
        <ul className="space-y-2">
          {accounts.map((account) => (
            <li key={account.id} className="flex items-center gap-3">
              <label className="flex cursor-pointer items-center gap-2">
                <input
                  type="checkbox"
                  checked={account.active}
                  onChange={() => handleToggle(account)}
                  className="h-4 w-4 rounded accent-[color:var(--accent)]"
                />
              </label>
              <span className="min-w-0 flex-1">
                <span className="block text-sm font-medium text-[color:var(--ink)]">{account.label}</span>
                <span className="block truncate text-xs text-[color:var(--ink-faint)]">{account.feed_url}</span>
                <span className="block text-xs text-[color:var(--ink-faint)]">
                  Every {account.refresh_interval_minutes ?? 180} min{account.last_refresh_at ? ` | Last refresh ${new Date(account.last_refresh_at).toLocaleString()}` : ""}
                </span>
              </span>
              <button
                type="button"
                onClick={() => handleDelete(account)}
                className="flex-shrink-0 rounded-lg border border-[color:rgba(255,107,127,0.4)] bg-[color:rgba(255,107,127,0.1)] px-3 py-1 text-xs font-semibold text-[color:var(--danger)] hover:bg-[color:rgba(255,107,127,0.2)]"
              >
                Remove
              </button>
            </li>
          ))}
        </ul>

        <div className="mt-4 border-t border-[color:var(--line)] pt-4">
          <p className="mb-2 text-xs font-semibold text-[color:var(--ink-faint)]">Add Account</p>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-[1fr_8rem_auto]">
            <input
              type="text"
              value={newAccount}
              onChange={(e) => setNewAccount(e.target.value)}
              placeholder="@SECGov"
              className="form-control px-2 py-1.5 text-sm"
            />
            <input
              type="number"
              min={15}
              max={1440}
              value={newInterval}
              onChange={(e) => setNewInterval(e.target.value)}
              className="form-control px-2 py-1.5 text-sm"
              aria-label="Refresh interval minutes"
            />
            <button
              type="button"
              onClick={handleAdd}
              disabled={adding || !newAccount.trim()}
              className="btn-solid rounded-xl px-4 py-1.5 text-sm disabled:opacity-40"
            >
              {adding ? "Adding..." : "Add"}
            </button>
          </div>
          {addError && <p className="mt-1 text-xs text-[color:var(--danger)]">{addError}</p>}
        </div>

        <div className="mt-4 border-t border-[color:var(--line)] pt-4">
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-[8rem_8rem_auto_1fr]">
            <input
              type="number"
              min={1}
              max={50}
              value={refreshLimit}
              onChange={(e) => setRefreshLimit(e.target.value)}
              className="form-control px-2 py-1.5 text-sm"
              aria-label="Posts per account"
            />
            <input
              type="number"
              min={0}
              max={50}
              value={analysisLimit}
              onChange={(e) => setAnalysisLimit(e.target.value)}
              className="form-control px-2 py-1.5 text-sm"
              aria-label="DeepSeek analysis limit"
            />
            <button
              type="button"
              onClick={handleRefresh}
              disabled={refreshing || activeCount === 0}
              className="btn-solid rounded-xl px-4 py-1.5 text-sm disabled:opacity-40"
            >
              {refreshing ? "Refreshing..." : "Refresh Active"}
            </button>
            <span className="self-center text-xs text-[color:var(--ink-faint)]">{activeCount} active</span>
          </div>
          {refreshError && <p className="mt-2 text-xs text-[color:var(--danger)]">{refreshError}</p>}
          {refreshResult && (
            <p className="mt-2 text-xs text-[color:var(--ink-faint)]">
              Inserted {refreshResult.inserted}; DeepSeek saved {refreshResult.analysis.saved_count}/{refreshResult.analysis.selected_count}.
              {refreshResult.feeds[0]?.provider ? ` Provider: ${refreshResult.feeds[0].provider}.` : ""}
              {refreshResult.feeds.some((feed) => feed.error) ? " Some X requests returned errors." : ""}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

type TopicReviewRow = {
  rule: TopicRule;
  recommendation?: TopicRuleRecommendation;
  keywordCount: number;
  broadTerms: string[];
  missingSuggestions: string[];
};

function normalizeKeywordToken(value: string): string {
  return value.toLowerCase().replace(/[\u2018\u2019]/g, "'").replace(/[\u201c\u201d]/g, "\"").replace(/[^a-z0-9]+/g, " ").trim();
}

function splitTopicKeywords(value: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of value.split(/[\n,]+/)) {
    const item = raw.trim();
    const key = normalizeKeywordToken(item);
    if (!item || seen.has(key)) continue;
    seen.add(key);
    out.push(item);
  }
  return out;
}

function getTopicReviewRow(rule: TopicRule, keywords: string): TopicReviewRow {
  const recommendation = TOPIC_RULE_RECOMMENDATION_BY_KEY[rule.topic_key];
  const currentKeywords = splitTopicKeywords(keywords);
  const currentKeys = new Set(currentKeywords.map(normalizeKeywordToken));
  const broadTerms = recommendation?.broadTerms.filter((term) => currentKeys.has(normalizeKeywordToken(term))) ?? [];
  const missingSuggestions =
    recommendation?.suggestedKeywords.filter((term) => !currentKeys.has(normalizeKeywordToken(term))).slice(0, 4) ?? [];

  return {
    rule,
    recommendation,
    keywordCount: currentKeywords.length,
    broadTerms,
    missingSuggestions,
  };
}

function TopicRuleRecommendationPanel({
  rule,
  onStage,
}: {
  rule: TopicRule;
  onStage: (rule: TopicRule) => void;
}) {
  const recommendation = TOPIC_RULE_RECOMMENDATION_BY_KEY[rule.topic_key];
  if (!recommendation) return null;

  return (
    <div className="mb-3 border-b border-[color:var(--line)] pb-3">
      <div className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <p className="mb-1 text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Review Guidance</p>
          <p className="text-sm leading-6 text-[color:var(--ink-soft)]">{recommendation.focus}</p>
        </div>
        <button
          type="button"
          onClick={() => onStage(rule)}
          className="rounded-lg border border-[color:var(--line)] px-3 py-1.5 text-xs font-semibold text-[color:var(--ink)] hover:border-[color:var(--accent)]"
        >
          Stage Suggested Keywords
        </button>
      </div>
      <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
        <div>
          <p className="mb-1 text-xs font-semibold text-[color:var(--ink-faint)]">Avoid As Standalone Signals</p>
          <p className="text-xs leading-5 text-[color:var(--ink-soft)]">{recommendation.broadTerms.join(", ")}</p>
        </div>
        <div>
          <p className="mb-1 text-xs font-semibold text-[color:var(--ink-faint)]">Suggested Precision Terms</p>
          <p className="text-xs leading-5 text-[color:var(--ink-soft)]">{recommendation.suggestedKeywords.slice(0, 10).join(", ")}</p>
        </div>
      </div>
      <ul className="mt-3 space-y-1">
        {recommendation.notes.map((note) => (
          <li key={note} className="text-xs leading-5 text-[color:var(--ink-faint)]">{note}</li>
        ))}
      </ul>
    </div>
  );
}

function TopicRulesReviewPanel({
  loading,
  rules,
  rowsWithRecommendations,
  rowsWithBroadTerms,
  totalKeywordCount,
  priorityRows,
  onStage,
}: {
  loading: boolean;
  rules: TopicRule[];
  rowsWithRecommendations: TopicReviewRow[];
  rowsWithBroadTerms: TopicReviewRow[];
  totalKeywordCount: number;
  priorityRows: TopicReviewRow[];
  onStage: (rule: TopicRule) => void;
}) {
  if (loading || rules.length === 0) return null;

  return (
    <div className="mb-4 border-b border-[color:var(--line)] pb-4">
      <div className="mb-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div className="flex flex-col gap-0.5">
          <span className="text-xl font-bold tabular-nums text-[color:var(--ink)]">{rules.length}</span>
          <span className="text-xs text-[color:var(--ink-faint)]">Rules</span>
        </div>
        <div className="flex flex-col gap-0.5">
          <span className="text-xl font-bold tabular-nums text-[#41d39d]">{rowsWithRecommendations.length}</span>
          <span className="text-xs text-[color:var(--ink-faint)]">Reviewed</span>
        </div>
        <div className="flex flex-col gap-0.5">
          <span className={`text-xl font-bold tabular-nums ${rowsWithBroadTerms.length ? "text-[color:var(--danger)]" : "text-[color:var(--ink)]"}`}>
            {rowsWithBroadTerms.length}
          </span>
          <span className="text-xs text-[color:var(--ink-faint)]">Broad-Term Risk</span>
        </div>
        <div className="flex flex-col gap-0.5">
          <span className="text-xl font-bold tabular-nums text-[color:var(--ink)]">{totalKeywordCount}</span>
          <span className="text-xs text-[color:var(--ink-faint)]">Keywords</span>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[0.8fr_1.2fr]">
        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Topic Quality Review</p>
          <p className="text-sm leading-6 text-[color:var(--ink-soft)]">
            Tighten broad single-word triggers, prefer phrase-level regulatory terms, and keep agency names from acting as topic signals by themselves.
          </p>
        </div>
        <div className="space-y-3">
          {priorityRows.length === 0 && (
            <p className="text-sm text-[#41d39d]">No priority keyword issues found in the current rules.</p>
          )}
          {priorityRows.map((row) => (
            <div key={row.rule.id} className="border-l border-[color:var(--line)] pl-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-semibold text-[color:var(--ink)]">{row.rule.label}</span>
                <span className="font-mono text-[10px] uppercase text-[color:var(--ink-faint)]">{row.rule.topic_key}</span>
                {row.broadTerms.length > 0 && (
                  <span className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[color:var(--danger)]">Broad terms</span>
                )}
              </div>
              {row.broadTerms.length > 0 && (
                <p className="mt-1 text-xs text-[color:var(--ink-faint)]">Review: {row.broadTerms.join(", ")}</p>
              )}
              {row.missingSuggestions.length > 0 && (
                <p className="mt-1 text-xs text-[color:var(--ink-faint)]">Add: {row.missingSuggestions.join(", ")}</p>
              )}
              {row.recommendation && (
                <button
                  type="button"
                  onClick={() => onStage(row.rule)}
                  className="mt-2 rounded-lg border border-[color:var(--line)] px-3 py-1 text-xs font-semibold text-[color:var(--ink)] hover:border-[color:var(--accent)]"
                >
                  Stage Suggested Keywords
                </button>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function TopicRulesSection() {
  const [rules, setRules] = useState<TopicRule[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [drafts, setDrafts] = useState<Record<number, Partial<TopicRule>>>({});
  const [saving, setSaving] = useState<number | null>(null);
  const [newRule, setNewRule] = useState({ label: "", topicKey: "", keywords: "", sortOrder: "100" });
  const [adding, setAdding] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/admin/topic-rules")
      .then((r) => r.json())
      .then((d) => { if (d.ok) setRules(d.data.rules); else setError(d.error); })
      .catch(() => setError("Network error"))
      .finally(() => setLoading(false));
  }, []);

  function draft(id: number, patch: Partial<TopicRule>) {
    setDrafts((p) => ({ ...p, [id]: { ...p[id], ...patch } }));
  }

  function getValue<K extends keyof TopicRule>(rule: TopicRule, key: K): TopicRule[K] {
    return (key in (drafts[rule.id] ?? {})) ? (drafts[rule.id] as TopicRule)[key] : rule[key];
  }

  async function handleSave(rule: TopicRule) {
    setSaving(rule.id);
    const d = drafts[rule.id] ?? {};
    const payload = {
      label: d.label,
      keywords: d.keywords,
      active: d.active,
      sortOrder: d.sort_order,
    };
    try {
      const res = await fetch(`/api/admin/topic-rules/${rule.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) { const j = await res.json().catch(() => ({})); setError(j.error ?? "Save failed"); return; }
      setRules((p) => p.map((r) => r.id === rule.id ? { ...r, ...d } : r));
      setDrafts((p) => { const n = { ...p }; delete n[rule.id]; return n; });
    } catch { setError("Network error"); }
    finally { setSaving(null); }
  }

  async function handleDelete(id: number) {
    try {
      const res = await fetch(`/api/admin/topic-rules/${id}`, { method: "DELETE" });
      if (!res.ok) { const j = await res.json().catch(() => ({})); setError(j.error ?? "Delete failed"); return; }
      setRules((p) => p.filter((r) => r.id !== id));
      if (expanded === id) setExpanded(null);
    } catch { setError("Network error"); }
  }

  async function handleAdd() {
    if (!newRule.label.trim() || !newRule.topicKey.trim()) return;
    setAdding(true);
    setAddError(null);
    try {
      const res = await fetch("/api/admin/topic-rules", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          label: newRule.label.trim(),
          topicKey: newRule.topicKey.trim(),
          keywords: newRule.keywords.trim(),
          sortOrder: parseInt(newRule.sortOrder) || 100,
          active: true,
        }),
      });
      const d = await res.json();
      if (d.ok) {
        setRules((p) => [...p, d.data.rule]);
        setNewRule({ label: "", topicKey: "", keywords: "", sortOrder: "100" });
      } else setAddError(d.error);
    } catch { setAddError("Network error"); }
    finally { setAdding(false); }
  }

  function stageSuggestedKeywords(rule: TopicRule) {
    const recommendation = TOPIC_RULE_RECOMMENDATION_BY_KEY[rule.topic_key];
    if (!recommendation) return;
    draft(rule.id, { keywords: formatTopicRuleKeywords(recommendation.suggestedKeywords) });
    setExpanded(rule.id);
  }

  const reviewRows = rules.map((rule) => getTopicReviewRow(rule, String(getValue(rule, "keywords") || "")));
  const rowsWithRecommendations = reviewRows.filter((row) => row.recommendation);
  const rowsWithBroadTerms = reviewRows.filter((row) => row.broadTerms.length > 0);
  const rowsMissingSuggestions = reviewRows.filter((row) => row.missingSuggestions.length > 0);
  const totalKeywordCount = reviewRows.reduce((sum, row) => sum + row.keywordCount, 0);
  const priorityRows = [
    ...rowsWithBroadTerms,
    ...rowsMissingSuggestions.filter((row) => row.broadTerms.length === 0),
  ].slice(0, 5);

  return (
    <section className="mb-8">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Topic Rules</h2>
      <p className="mb-3 text-xs text-[color:var(--ink-faint)]">Keywords that route Intel Feed articles into sidebar topics.</p>
      <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.88)] px-4 py-4">
        {loading && <p className="text-xs text-[color:var(--ink-faint)]">Loading…</p>}
        {error && <p className="text-xs text-[color:var(--danger)]">{error}</p>}
        <TopicRulesReviewPanel
          loading={loading}
          rules={rules}
          rowsWithRecommendations={rowsWithRecommendations}
          rowsWithBroadTerms={rowsWithBroadTerms}
          totalKeywordCount={totalKeywordCount}
          priorityRows={priorityRows}
          onStage={stageSuggestedKeywords}
        />
        <ul className="space-y-2">
          {rules.map((rule) => (
            <li key={rule.id} className="rounded-lg border border-[color:var(--line)]">
              <button
                type="button"
                onClick={() => setExpanded(expanded === rule.id ? null : rule.id)}
                className="flex w-full items-center gap-3 px-3 py-2.5 text-left"
              >
                <span className={`inline-block h-1.5 w-1.5 flex-shrink-0 rounded-full ${getValue(rule, "active") ? "bg-[#41d39d]" : "bg-[color:var(--ink-faint)]"}`} />
                <span className="flex-1 text-sm font-medium text-[color:var(--ink)]">{getValue(rule, "label")}</span>
                <span className="font-mono text-xs text-[color:var(--ink-faint)]">{rule.topic_key}</span>
                <span className="text-xs text-[color:var(--ink-faint)]">{expanded === rule.id ? "▲" : "▼"}</span>
              </button>
              {expanded === rule.id && (
                <div className="border-t border-[color:var(--line)] px-3 py-3">
                  <TopicRuleRecommendationPanel rule={rule} onStage={stageSuggestedKeywords} />
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <div className="flex flex-col gap-1">
                      <label className="text-xs text-[color:var(--ink-faint)]">Label</label>
                      <input
                        type="text"
                        value={getValue(rule, "label") as string}
                        onChange={(e) => draft(rule.id, { label: e.target.value })}
                        className="form-control px-2 py-1.5 text-sm"
                      />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-xs text-[color:var(--ink-faint)]">Sort Order</label>
                      <input
                        type="number"
                        value={getValue(rule, "sort_order") as number}
                        onChange={(e) => draft(rule.id, { sort_order: parseInt(e.target.value) || 0 })}
                        className="form-control px-2 py-1.5 text-sm"
                      />
                    </div>
                    <div className="flex flex-col gap-1 sm:col-span-2">
                      <label className="text-xs text-[color:var(--ink-faint)]">Keywords (comma or newline separated)</label>
                      <textarea
                        rows={3}
                        value={getValue(rule, "keywords") as string}
                        onChange={(e) => draft(rule.id, { keywords: e.target.value })}
                        className="form-control px-2 py-1.5 text-sm"
                      />
                    </div>
                  </div>
                  <div className="mt-3 flex items-center gap-4">
                    <label className="flex cursor-pointer items-center gap-2">
                      <input
                        type="checkbox"
                        checked={getValue(rule, "active") as boolean}
                        onChange={(e) => draft(rule.id, { active: e.target.checked })}
                        className="h-4 w-4 rounded accent-[color:var(--accent)]"
                      />
                      <span className="text-xs text-[color:var(--ink-faint)]">Active</span>
                    </label>
                    <button
                      type="button"
                      onClick={() => handleSave(rule)}
                      disabled={saving === rule.id || !(rule.id in drafts)}
                      className="btn-solid rounded-lg px-4 py-1.5 text-xs font-semibold disabled:opacity-40"
                    >
                      {saving === rule.id ? "Saving…" : "Save"}
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDelete(rule.id)}
                      className="ml-auto rounded-lg border border-[color:rgba(255,107,127,0.4)] bg-[color:rgba(255,107,127,0.1)] px-3 py-1.5 text-xs font-semibold text-[color:var(--danger)] hover:bg-[color:rgba(255,107,127,0.2)]"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              )}
            </li>
          ))}
        </ul>
        <div className="mt-4 border-t border-[color:var(--line)] pt-4">
          <p className="mb-2 text-xs font-semibold text-[color:var(--ink-faint)]">Add Topic Rule</p>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <input type="text" value={newRule.label} onChange={(e) => setNewRule((p) => ({ ...p, label: e.target.value }))} placeholder="Label (e.g. AI Regulation)" className="form-control px-2 py-1.5 text-sm" />
            <input type="text" value={newRule.topicKey} onChange={(e) => setNewRule((p) => ({ ...p, topicKey: e.target.value }))} placeholder="Key (e.g. AI_REGULATION)" className="form-control px-2 py-1.5 text-sm" />
            <textarea rows={2} value={newRule.keywords} onChange={(e) => setNewRule((p) => ({ ...p, keywords: e.target.value }))} placeholder="Keywords, comma separated" className="form-control px-2 py-1.5 text-sm sm:col-span-2" />
            <input type="number" value={newRule.sortOrder} onChange={(e) => setNewRule((p) => ({ ...p, sortOrder: e.target.value }))} placeholder="Sort order" className="form-control px-2 py-1.5 text-sm" />
            <button type="button" onClick={handleAdd} disabled={adding || !newRule.label.trim() || !newRule.topicKey.trim()} className="btn-solid rounded-xl px-4 py-1.5 text-sm disabled:opacity-40">
              {adding ? "Adding…" : "Add Rule"}
            </button>
          </div>
          {addError && <p className="mt-1 text-xs text-[color:var(--danger)]">{addError}</p>}
        </div>
      </div>
    </section>
  );
}

/* ─── WorkflowPanel component ──────────────────────────────────────── */
function WorkflowPanel({
  title,
  description,
  workflowFile,
  fields,
}: {
  title: string;
  description: string;
  workflowFile: string;
  fields: FieldDef[];
}) {
  const [values, setValues] = useState<Record<string, string>>(
    Object.fromEntries(fields.map((f) => [f.name, f.default ?? ""]))
  );
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<"idle" | "ok" | "error">("idle");
  const [error, setError] = useState<string | null>(null);

  function setValue(name: string, val: string) {
    setValues((prev) => ({ ...prev, [name]: val }));
    setStatus("idle");
  }

  async function handleRun() {
    setRunning(true);
    setStatus("idle");
    setError(null);
    try {
      const res = await fetch("/api/admin/workflow", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workflow: workflowFile, inputs: values }),
      });
      const data = await res.json();
      if (data.ok) {
        setStatus("ok");
      } else {
        setError(data.error ?? `HTTP ${res.status}`);
        setStatus("error");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Network error");
      setStatus("error");
    } finally {
      setRunning(false);
    }
  }

  const regularFields = fields.filter((f) => f.type !== "boolean");
  const booleanFields = fields.filter((f) => f.type === "boolean");

  return (
    <section className="mb-8">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
        {title}
      </h2>
      <p className="mb-3 text-xs text-[color:var(--ink-faint)]">{description}</p>

      <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.88)] px-4 py-4">
        {/* Regular fields */}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {regularFields.map((field) => (
            <div key={field.name} className="flex flex-col gap-1">
              <label className="text-xs text-[color:var(--ink-faint)]">{field.label}</label>
              {field.type === "select" ? (
                <select
                  value={values[field.name]}
                  onChange={(e) => setValue(field.name, e.target.value)}
                  className="form-control px-2 py-1.5 text-sm"
                >
                  {field.options.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type={field.type === "number" ? "number" : "text"}
                  value={values[field.name]}
                  onChange={(e) => setValue(field.name, e.target.value)}
                  placeholder={"placeholder" in field ? field.placeholder : undefined}
                  className="form-control px-2 py-1.5 text-sm"
                />
              )}
            </div>
          ))}
        </div>

        {/* Boolean toggles */}
        {booleanFields.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-5">
            {booleanFields.map((field) => (
              <label key={field.name} className="flex cursor-pointer items-center gap-2">
                <input
                  type="checkbox"
                  checked={values[field.name] === "true"}
                  onChange={(e) => setValue(field.name, e.target.checked ? "true" : "false")}
                  className="h-4 w-4 rounded accent-[color:var(--accent)]"
                />
                <span className="text-xs text-[color:var(--ink-faint)]">{field.label}</span>
              </label>
            ))}
          </div>
        )}

        {/* Run button + status */}
        <div className="mt-4 flex flex-wrap items-center gap-4 border-t border-[color:var(--line)] pt-4">
          <button
            type="button"
            onClick={handleRun}
            disabled={running}
            className="btn-solid rounded-xl px-5 py-2 text-sm font-semibold disabled:opacity-40"
          >
            {running ? "Dispatching…" : "Run Workflow"}
          </button>
          {status === "ok" && (
            <span className="text-sm text-[color:var(--ok)]">
              Dispatched
            </span>
          )}
          {status === "error" && (
            <span className="text-sm text-[color:var(--danger)]">
              Failed{error ? `: ${error}` : " — try again"}
            </span>
          )}
          <span className="ml-auto">
            <JobStatusBadge workflowFile={workflowFile} />
          </span>
        </div>
      </div>
    </section>
  );
}

/* ─── Enrichment Pipeline types ────────────────────────────────────── */
function BloombergOnDemandSection() {
  const [limit, setLimit] = useState("10");
  const [maxPages, setMaxPages] = useState("10");
  const [selection, setSelection] = useState<"new_or_updated" | "all">("new_or_updated");
  const [baseUrl, setBaseUrl] = useState("");
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<"idle" | "ok" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<AdminJobState | null>(null);

  async function runBloombergLatest() {
    setRunning(true);
    setStatus("idle");
    setError(null);
    setJob(null);

    try {
      const res = await fetch("/api/jobs/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          connector: "bloomberg_public_latest",
          selection,
          limit: Math.max(1, Number.parseInt(limit || "10", 10) || 10),
          max_pages: Math.max(1, Number.parseInt(maxPages || "10", 10) || 10),
          base_url: baseUrl.trim(),
          include_pdfs: "true",
          include_rss: "true",
        }),
      });
      const payload = await res.json().catch(() => null) as { ok?: boolean; data?: AdminJobState; error?: string } | null;
      if (!res.ok || !payload?.ok || !payload.data?.job_id) {
        throw new Error(payload?.error || `HTTP ${res.status}`);
      }
      setJob(payload.data);
      setStatus("ok");
    } catch (e) {
      setStatus("error");
      setError(e instanceof Error ? e.message : "Network error");
    } finally {
      setRunning(false);
    }
  }

  useEffect(() => {
    if (!job?.job_id || ["success", "failed"].includes(job.status)) return;
    let cancelled = false;
    const timer = window.setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${encodeURIComponent(job.job_id)}`, { cache: "no-store" });
        const payload = await res.json().catch(() => null) as { ok?: boolean; data?: AdminJobState } | null;
        if (!cancelled && res.ok && payload?.ok && payload.data) {
          setJob(payload.data);
        }
      } catch {
        // Keep the last known status visible.
      }
    }, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [job?.job_id, job?.status]);

  const latestStatus = job?.status || "idle";
  const isActive = latestStatus === "queued" || latestStatus === "running";

  return (
    <section className="mb-8">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
        Bloomberg Public Pull
      </h2>
      <p className="mb-3 text-xs text-[color:var(--ink-faint)]">
        Discover recent Bloomberg URLs from public RSS feeds, extract public article text when available, and save them into the corpus.
      </p>
      <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.88)] px-4 py-4">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-[color:var(--ink-faint)]">Selection</label>
            <select
              value={selection}
              onChange={(e) => setSelection(e.target.value === "all" ? "all" : "new_or_updated")}
              className="form-control px-2 py-1.5 text-sm"
            >
              <option value="new_or_updated">New or Updated</option>
              <option value="all">All (re-extract)</option>
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-[color:var(--ink-faint)]">Article limit</label>
            <input
              type="number"
              min={1}
              max={50}
              value={limit}
              onChange={(e) => setLimit(e.target.value)}
              className="form-control px-2 py-1.5 text-sm"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-[color:var(--ink-faint)]">Discovery limit</label>
            <input
              type="number"
              min={1}
              max={100}
              value={maxPages}
              onChange={(e) => setMaxPages(e.target.value)}
              className="form-control px-2 py-1.5 text-sm"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-[color:var(--ink-faint)]">Section URL override</label>
            <input
              type="text"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="Optional Bloomberg section URL"
              className="form-control px-2 py-1.5 text-sm"
            />
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-4 border-t border-[color:var(--line)] pt-4">
          <button
            type="button"
            onClick={runBloombergLatest}
            disabled={running || isActive}
            className="btn-solid rounded-xl px-5 py-2 text-sm font-semibold disabled:opacity-40"
          >
            {running ? "Dispatching..." : isActive ? "Running..." : "Pull Latest Bloomberg"}
          </button>
          {status === "error" && (
            <span className="text-sm text-[color:var(--danger)]">
              Failed{error ? `: ${error}` : ""}
            </span>
          )}
          {job ? (
            <div className="flex flex-wrap items-center gap-3 text-sm">
              <span
                className={
                  job.status === "success"
                    ? "text-[color:var(--ok)]"
                    : job.status === "failed"
                      ? "text-[color:var(--danger)]"
                      : "text-[color:var(--accent)]"
                }
              >
                {job.status}
              </span>
              {job.html_url ? (
                <a href={job.html_url} target="_blank" rel="noopener noreferrer" className="link-inline text-xs">
                  Open GitHub run
                </a>
              ) : null}
              {job.artifacts?.length ? (
                <span className="text-xs text-[color:var(--ink-faint)]">Artifact: {job.artifacts.join(", ")}</span>
              ) : null}
            </div>
          ) : null}
          <span className="ml-auto">
            <JobStatusBadge workflowFile="policy-extraction.yml" />
          </span>
        </div>
      </div>
    </section>
  );
}

type OrgEnrichmentStatus = { org_key: string; org_label: string; total: number; enriched: number; failed: number; pending: number };
type FailedDoc = { doc_id: string; title: string; org_key: string; org_label: string; error: string; updated_at: string };
type EnrichmentStatusData = {
  total: number; enriched: number; failed: number; pending: number; updated_at: string | null;
  by_org: OrgEnrichmentStatus[];
  failed_docs: FailedDoc[];
};

/* ─── Enrichment Pipeline ──────────────────────────────────────────── */
function EnrichmentPipelineSection() {
  const [data, setData] = useState<EnrichmentStatusData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [showFailed, setShowFailed] = useState(false);

  const [enrich, setEnrich] = useState({ source_kind: "newsapi_article", enrich_limit: "25", mode: "only_missing_or_failed", heuristic_only: "false", provider: "deepseek", model: "" });
  const [dispatching, setDispatching] = useState(false);
  const [dispatchStatus, setDispatchStatus] = useState<"idle" | "ok" | "error">("idle");
  const [dispatchError, setDispatchError] = useState<string | null>(null);
  const [analysisLimit, setAnalysisLimit] = useState("10");
  const [analysisRunning, setAnalysisRunning] = useState(false);
  const [analysisMessage, setAnalysisMessage] = useState<string | null>(null);
  const [rssAnalysisLimit, setRssAnalysisLimit] = useState("10");
  const [rssAnalysisRunning, setRssAnalysisRunning] = useState(false);
  const [rssAnalysisMessage, setRssAnalysisMessage] = useState<string | null>(null);

  const [cancelling, setCancelling] = useState(false);
  const [cancelMsg, setCancelMsg] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/admin/enrichment-status")
      .then((r) => r.json())
      .then((json: { ok: boolean; data?: EnrichmentStatusData; error?: string }) => {
        if (json.ok && json.data) setData(json.data);
        else setError(json.error ?? "Failed to load");
      })
      .catch(() => setError("Network error"))
      .finally(() => setLoading(false));
  }, []);

  async function handleDispatch() {
    setDispatching(true);
    setDispatchStatus("idle");
    setDispatchError(null);
    try {
      const res = await fetch("/api/admin/workflow", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workflow: "financial-news-enrich.yml", inputs: enrich }),
      });
      const d = await res.json();
      if (d.ok) setDispatchStatus("ok");
      else { setDispatchError(d.error); setDispatchStatus("error"); }
    } catch { setDispatchStatus("error"); }
    finally { setDispatching(false); }
  }

  async function handleCancel() {
    setCancelling(true);
    setCancelMsg(null);
    try {
      const res = await fetch("/api/admin/workflow/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workflow: "financial-news-enrich.yml" }),
      });
      const d = await res.json();
      setCancelMsg(d.ok ? "Cancelled" : (d.error ?? "Failed"));
    } catch { setCancelMsg("Network error"); }
    finally { setCancelling(false); }
  }

  async function handleEnforcementAnalysisBatch() {
    setAnalysisRunning(true);
    setAnalysisMessage(null);
    try {
      const res = await fetch("/api/admin/enforcement-analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ limit: analysisLimit, mode: "missing" }),
      });
      const d = await res.json();
      if (!d.ok) throw new Error(d.error || "Failed");
      setAnalysisMessage(`Saved ${d.data.saved_count} analysis item${d.data.saved_count === 1 ? "" : "s"}${d.data.failed_count ? `; ${d.data.failed_count} failed` : ""}.`);
    } catch (err) {
      setAnalysisMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setAnalysisRunning(false);
    }
  }

  async function handleRssAnalysisBatch() {
    setRssAnalysisRunning(true);
    setRssAnalysisMessage(null);
    try {
      const res = await fetch("/api/admin/rss-analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ limit: rssAnalysisLimit }),
      });
      const d = await res.json();
      if (!d.ok) throw new Error(d.error || "Failed");
      setRssAnalysisMessage(`Saved ${d.data.saved_count} RSS analysis item${d.data.saved_count === 1 ? "" : "s"}${d.data.failed_count ? `; ${d.data.failed_count} failed` : ""}.`);
    } catch (err) {
      setRssAnalysisMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setRssAnalysisRunning(false);
    }
  }

  return (
    <section className="mb-8">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Enrichment Pipeline</h2>
      <p className="mb-3 text-xs text-[color:var(--ink-faint)]">Enrichment coverage across all corpus documents, and controls to run or stop the enrichment workflow.</p>
      <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.88)] px-4 py-4">
        {loading && <p className="text-xs text-[color:var(--ink-faint)]">Loading…</p>}
        {error && <p className="text-xs text-[color:var(--danger)]">{error}</p>}
        {data && (
          <>
            {/* Aggregate stats */}
            <div className="mb-4 flex flex-wrap gap-6 border-b border-[color:var(--line)] pb-4">
              <Stat label="Total Docs" value={data.total} />
              <Stat label="Enriched" value={data.enriched} color="text-[#41d39d]" />
              <Stat label="Pending" value={data.pending} color="text-[color:var(--accent)]" />
              <Stat label="Failed" value={data.failed} color={data.failed > 0 ? "text-[color:var(--danger)]" : undefined} />
              {data.total > 0 && (
                <div className="flex w-full flex-col gap-1">
                  <div className="flex h-1.5 w-full overflow-hidden rounded-full bg-[color:rgba(255,255,255,0.06)]">
                    <div className="h-full bg-[#41d39d]" style={{ width: `${(data.enriched / data.total) * 100}%` }} />
                    <div className="h-full bg-[color:var(--danger)] opacity-70" style={{ width: `${(data.failed / data.total) * 100}%` }} />
                  </div>
                  <p className="text-[10px] text-[color:var(--ink-faint)]">
                    {Math.round((data.enriched / data.total) * 100)}% enriched
                    {data.updated_at ? ` · updated ${new Date(data.updated_at).toLocaleDateString()}` : ""}
                  </p>
                </div>
              )}
            </div>

            {/* Per-org breakdown table */}
            {data.by_org.length > 0 && (
              <div className="mb-4 overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-[color:var(--line)] text-left text-[color:var(--ink-faint)]">
                      <th className="pb-2 pr-4 font-semibold">Org</th>
                      <th className="pb-2 pr-4 text-right font-semibold">Total</th>
                      <th className="pb-2 pr-4 text-right font-semibold">Enriched</th>
                      <th className="pb-2 pr-4 text-right font-semibold">Failed</th>
                      <th className="pb-2 pr-4 text-right font-semibold">Pending</th>
                      <th className="pb-2 text-right font-semibold">%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.by_org.map((org) => (
                      <tr key={org.org_key} className="border-b border-[color:rgba(255,255,255,0.04)]">
                        <td className="py-2 pr-4 font-medium text-[color:var(--ink)]">{org.org_label}</td>
                        <td className="py-2 pr-4 text-right tabular-nums text-[color:var(--ink)]">{org.total.toLocaleString()}</td>
                        <td className="py-2 pr-4 text-right tabular-nums text-[#41d39d]">{org.enriched.toLocaleString()}</td>
                        <td className={`py-2 pr-4 text-right tabular-nums ${org.failed > 0 ? "text-[color:var(--danger)]" : "text-[color:var(--ink-faint)]"}`}>{org.failed.toLocaleString()}</td>
                        <td className={`py-2 pr-4 text-right tabular-nums ${org.pending > 0 ? "text-[color:var(--accent)]" : "text-[color:var(--ink-faint)]"}`}>{org.pending.toLocaleString()}</td>
                        <td className="py-2 text-right tabular-nums text-[color:var(--ink-faint)]">
                          {org.total > 0 ? `${Math.round((org.enriched / org.total) * 100)}%` : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Failed docs collapsible */}
            {data.failed_docs.length > 0 && (
              <div className="mb-4">
                <button
                  type="button"
                  onClick={() => setShowFailed((p) => !p)}
                  className="flex items-center gap-2 text-xs font-semibold text-[color:var(--danger)] hover:underline"
                >
                  <span className="inline-block h-1.5 w-1.5 rounded-full bg-[color:var(--danger)]" />
                  {data.failed_docs.length} Failed Doc{data.failed_docs.length !== 1 ? "s" : ""}
                  <span className="text-[color:var(--ink-faint)]">{showFailed ? "▲" : "▼"}</span>
                </button>
                {showFailed && (
                  <div className="mt-2 max-h-64 overflow-y-auto rounded-lg border border-[color:var(--line)]">
                    <table className="w-full text-xs">
                      <thead className="sticky top-0 bg-[color:rgba(9,22,36,0.95)]">
                        <tr className="border-b border-[color:var(--line)] text-left text-[color:var(--ink-faint)]">
                          <th className="px-3 py-2 font-semibold">Title</th>
                          <th className="px-3 py-2 font-semibold">Org</th>
                          <th className="px-3 py-2 font-semibold">Error</th>
                          <th className="px-3 py-2 font-semibold">Updated</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.failed_docs.map((doc) => (
                          <tr key={doc.doc_id} className="border-b border-[color:rgba(255,255,255,0.04)]">
                            <td className="max-w-[200px] truncate px-3 py-2 font-medium text-[color:var(--ink)]" title={doc.title}>{doc.title}</td>
                            <td className="px-3 py-2 text-[color:var(--ink-faint)]">{doc.org_label}</td>
                            <td className="max-w-[240px] truncate px-3 py-2 text-[color:var(--danger)]" title={doc.error}>{doc.error}</td>
                            <td className="whitespace-nowrap px-3 py-2 text-[color:var(--ink-faint)]">{doc.updated_at ? new Date(doc.updated_at).toLocaleDateString() : "—"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </>
        )}

        <div className={`${data ? "border-t border-[color:var(--line)] pt-4" : ""}`}>
          <p className="mb-2 text-xs font-semibold text-[color:var(--ink-faint)]">RSS Feed Analysis</p>
          <p className="mb-3 text-xs text-[color:var(--ink-faint)]">Generate and persist DeepSeek analysis for RSS Feed articles missing saved keywords, individuals, and entities. Saved results render automatically on Feed cards and populate the shared mention index.</p>
          <div className="mb-4 flex flex-wrap items-end gap-3">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-[color:var(--ink-faint)]">Batch size</span>
              <input
                type="number"
                min="1"
                max="50"
                value={rssAnalysisLimit}
                onChange={(e) => setRssAnalysisLimit(e.target.value)}
                className="form-control w-28 px-2 py-1.5 text-sm"
              />
            </label>
            <button
              type="button"
              onClick={handleRssAnalysisBatch}
              disabled={rssAnalysisRunning}
              className="btn-solid rounded-xl px-4 py-2 text-sm font-semibold disabled:opacity-40"
            >
              {rssAnalysisRunning ? "Generating..." : "Generate RSS Analysis"}
            </button>
            {rssAnalysisMessage ? <span className="text-xs text-[color:var(--ink-faint)]">{rssAnalysisMessage}</span> : null}
          </div>

          <p className="mb-2 text-xs font-semibold text-[color:var(--ink-faint)]">SEC Enforcement Analysis</p>
          <p className="mb-3 text-xs text-[color:var(--ink-faint)]">Generate DeepSeek analysis for recent SEC litigation releases missing saved analysis. Saved results render automatically on Enforcement cards.</p>
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-[color:var(--ink-faint)]">Batch size</span>
              <input
                type="number"
                min="1"
                max="25"
                value={analysisLimit}
                onChange={(e) => setAnalysisLimit(e.target.value)}
                className="form-control w-28 px-2 py-1.5 text-sm"
              />
            </label>
            <button
              type="button"
              onClick={handleEnforcementAnalysisBatch}
              disabled={analysisRunning}
              className="btn-solid rounded-xl px-4 py-2 text-sm font-semibold disabled:opacity-40"
            >
              {analysisRunning ? "Generating..." : "Generate Missing Analysis"}
            </button>
            {analysisMessage ? <span className="text-xs text-[color:var(--ink-faint)]">{analysisMessage}</span> : null}
          </div>
        </div>

        {/* Dispatch controls */}
        <div className="mt-4 border-t border-[color:var(--line)] pt-4">
          <p className="mb-3 text-xs font-semibold text-[color:var(--ink-faint)]">Run Enrichment</p>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-[color:var(--ink-faint)]">Source kind</label>
              <input
                type="text"
                value={enrich.source_kind}
                onChange={(e) => setEnrich((p) => ({ ...p, source_kind: e.target.value }))}
                className="form-control px-2 py-1.5 text-sm"
                placeholder="e.g. newsapi_article"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-[color:var(--ink-faint)]">Max to enrich</label>
              <input
                type="number"
                value={enrich.enrich_limit}
                onChange={(e) => setEnrich((p) => ({ ...p, enrich_limit: e.target.value }))}
                className="form-control px-2 py-1.5 text-sm"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-[color:var(--ink-faint)]">Mode</label>
              <select
                value={enrich.mode}
                onChange={(e) => setEnrich((p) => ({ ...p, mode: e.target.value }))}
                className="form-control px-2 py-1.5 text-sm"
              >
                <option value="only_missing_or_failed">Missing / Failed only</option>
                <option value="all">All (re-enrich)</option>
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-[color:var(--ink-faint)]">Provider</label>
              <select
                value={enrich.provider}
                onChange={(e) => setEnrich((p) => ({ ...p, provider: e.target.value === "openai" ? "openai" : "deepseek" }))}
                className="form-control px-2 py-1.5 text-sm"
              >
                <option value="deepseek">DeepSeek</option>
                <option value="openai">OpenAI</option>
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-[color:var(--ink-faint)]">Model override</label>
              <input
                type="text"
                value={enrich.model}
                onChange={(e) => setEnrich((p) => ({ ...p, model: e.target.value }))}
                placeholder="e.g. deepseek-v4-pro (leave blank for default)"
                className="form-control px-2 py-1.5 text-sm"
              />
            </div>
          </div>
          <div className="mt-3">
            <label className="flex cursor-pointer items-center gap-2">
              <input
                type="checkbox"
                checked={enrich.heuristic_only === "true"}
                onChange={(e) => setEnrich((p) => ({ ...p, heuristic_only: e.target.checked ? "true" : "false" }))}
                className="h-4 w-4 rounded accent-[color:var(--accent)]"
              />
              <span className="text-xs text-[color:var(--ink-faint)]">Skip hosted model (heuristic only)</span>
            </label>
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={handleDispatch}
              disabled={dispatching}
              className="btn-solid rounded-xl px-5 py-2 text-sm font-semibold disabled:opacity-40"
            >
              {dispatching ? "Dispatching…" : "Run Enrichment"}
            </button>
            <button
              type="button"
              onClick={handleCancel}
              disabled={cancelling}
              className="rounded-xl border border-[color:rgba(255,107,127,0.4)] bg-[color:rgba(255,107,127,0.1)] px-4 py-2 text-sm font-semibold text-[color:var(--danger)] hover:bg-[color:rgba(255,107,127,0.2)] disabled:opacity-40"
            >
              {cancelling ? "Cancelling…" : "Cancel Active Run"}
            </button>
            {dispatchStatus === "ok" && <span className="text-sm text-[color:var(--ok)]">Dispatched</span>}
            {dispatchStatus === "error" && <span className="text-sm text-[color:var(--danger)]">Failed{dispatchError ? `: ${dispatchError}` : ""}</span>}
            {cancelMsg && <span className={`text-sm ${cancelMsg === "Cancelled" ? "text-[color:var(--ok)]" : "text-[color:var(--danger)]"}`}>{cancelMsg}</span>}
            <span className="ml-auto">
              <JobStatusBadge workflowFile="financial-news-enrich.yml" />
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}

function Stat({ label, value, color }: { label: string; value: number; color?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className={`text-xl font-bold tabular-nums ${color ?? "text-[color:var(--ink)]"}`}>{value.toLocaleString()}</span>
      <span className="text-xs text-[color:var(--ink-faint)]">{label}</span>
    </div>
  );
}

/* ─── Document Library ─────────────────────────────────────────────── */
type DocItem = {
  document_id: string; title: string; organization: string; source_kind: string;
  doc_type: string; speaker: string; url: string; date: string; word_count: number;
  tags: string[]; keywords: string[]; enrichment_status: string; review_decision: string;
  sentiment_label: string; sentiment_score: number;
};
type DocFacets = { sources: string[]; organizations: string[]; statuses: string[] };

function fmtSourceKind(sk: string): string {
  return sk.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function EnrichBadge({ status }: { status: string }) {
  const cls =
    status === "enriched" ? "text-[#41d39d]" :
    status === "failed"   ? "text-[color:var(--danger)]" :
    status === "pending"  ? "text-[color:var(--accent)]" :
    "text-[color:var(--ink-faint)]";
  const label =
    status === "enriched" ? "Enriched" :
    status === "failed"   ? "Failed" :
    status === "pending"  ? "Pending" : "Not Enriched";
  return <span className={`text-xs font-medium ${cls}`}>{label}</span>;
}

function DocumentLibrarySection() {
  const PAGE_SIZE = 25;
  const [pendingQ, setPendingQ] = useState("");
  const [filters, setFilters] = useState({ q: "", source_kind: "", status: "", sort: "date_desc" });
  const [page, setPage] = useState(1);

  const [items, setItems] = useState<DocItem[]>([]);
  const [total, setTotal] = useState(0);
  const [facets, setFacets] = useState<DocFacets | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [initialized, setInitialized] = useState(false);

  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState<string | null>(null);

  async function fetchDocs(f: typeof filters, pg: number) {
    setLoading(true);
    setError(null);
    try {
      const sp = new URLSearchParams({ sort: f.sort, page: String(pg), page_size: String(PAGE_SIZE) });
      if (f.q) sp.set("q", f.q);
      if (f.source_kind) sp.set("source_kind", f.source_kind);
      if (f.status) sp.set("status", f.status);
      const res = await fetch(`/api/documents?${sp.toString()}`);
      const d = await res.json() as { ok: boolean; data?: { items: DocItem[]; total: number; facets: DocFacets }; error?: string };
      if (d.ok && d.data) {
        setItems(d.data.items);
        setTotal(d.data.total);
        if (pg === 1) setFacets(d.data.facets);
        setInitialized(true);
      } else setError(d.error ?? "Failed to load");
    } catch { setError("Network error"); }
    finally { setLoading(false); }
  }

  function applyFilters(patch: Partial<typeof filters>, resetPage = true) {
    const next = { ...filters, ...patch };
    setFilters(next);
    const pg = resetPage ? 1 : page;
    if (resetPage) setPage(1);
    fetchDocs(next, pg);
  }

  function handleSearch() {
    applyFilters({ q: pendingQ.trim() });
  }

  function handlePage(delta: number) {
    const next = page + delta;
    setPage(next);
    fetchDocs(filters, next);
  }

  async function handleDelete(doc: DocItem) {
    if (!window.confirm(`Delete "${doc.title || doc.document_id}"?\n\nThis removes it from the GCS corpus. If it came from an extraction workflow it may be re-added on the next run.`)) return;
    setDeleting(doc.document_id);
    try {
      const res = await fetch(`/api/admin/documents/${encodeURIComponent(doc.document_id)}`, { method: "DELETE" });
      const d = await res.json() as { ok: boolean; error?: string };
      if (d.ok) {
        setItems((p) => p.filter((i) => i.document_id !== doc.document_id));
        setTotal((p) => Math.max(0, p - 1));
      } else {
        alert(d.error ?? "Delete failed");
      }
    } catch { alert("Network error"); }
    finally { setDeleting(null); }
  }

  function toggleExpand(id: string) {
    setExpanded((p) => { const n = new Set(p); n.has(id) ? n.delete(id) : n.add(id); return n; });
  }

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const start = (page - 1) * PAGE_SIZE + 1;
  const end = Math.min(page * PAGE_SIZE, total);

  return (
    <section className="mb-8">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Document Library</h2>
      <p className="mb-3 text-xs text-[color:var(--ink-faint)]">Browse, search, and manage all corpus documents with enrichment status.</p>

      <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.88)] px-4 py-4">
        {/* Filter bar */}
        <div className="mb-4 flex flex-wrap gap-2">
          <div className="flex flex-1 gap-2">
            <input
              type="text"
              value={pendingQ}
              onChange={(e) => setPendingQ(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="Search title or text…"
              className="form-control min-w-0 flex-1 px-2 py-1.5 text-sm"
            />
            <button
              type="button"
              onClick={handleSearch}
              className="btn-solid rounded-xl px-4 py-1.5 text-sm font-semibold"
            >
              {loading && initialized ? "…" : "Search"}
            </button>
          </div>
          <select
            value={filters.source_kind}
            onChange={(e) => applyFilters({ source_kind: e.target.value })}
            disabled={!initialized}
            className="form-control px-2 py-1.5 text-sm disabled:opacity-40"
          >
            <option value="">All sources</option>
            {(facets?.sources ?? []).map((s) => (
              <option key={s} value={s}>{fmtSourceKind(s)}</option>
            ))}
          </select>
          <select
            value={filters.status}
            onChange={(e) => applyFilters({ status: e.target.value })}
            disabled={!initialized}
            className="form-control px-2 py-1.5 text-sm disabled:opacity-40"
          >
            <option value="">All statuses</option>
            {(facets?.statuses ?? []).map((s) => (
              <option key={s} value={s}>{s.replace(/_/g, " ")}</option>
            ))}
          </select>
          <select
            value={filters.sort}
            onChange={(e) => applyFilters({ sort: e.target.value })}
            className="form-control px-2 py-1.5 text-sm"
          >
            <option value="date_desc">Newest first</option>
            <option value="date_asc">Oldest first</option>
            <option value="updated_desc">Recently updated</option>
          </select>
        </div>

        {!initialized && !loading && (
          <div className="py-8 text-center">
            <button
              type="button"
              onClick={() => fetchDocs(filters, 1)}
              className="btn-solid rounded-xl px-6 py-2 text-sm font-semibold"
            >
              Load Documents
            </button>
            <p className="mt-2 text-xs text-[color:var(--ink-faint)]">Loads the full corpus from GCS — may take a few seconds.</p>
          </div>
        )}

        {error && <p className="text-xs text-[color:var(--danger)]">{error}</p>}

        {initialized && (
          <>
            {/* Results summary */}
            <div className="mb-2 flex items-center justify-between text-xs text-[color:var(--ink-faint)]">
              <span>{total === 0 ? "No documents" : `${start.toLocaleString()}–${end.toLocaleString()} of ${total.toLocaleString()} documents`}</span>
              {loading && <span className="animate-pulse">Loading…</span>}
            </div>

            {/* Table */}
            <div className="overflow-x-auto rounded-lg border border-[color:var(--line)]">
              <table className="w-full text-xs">
                <thead className="bg-[color:rgba(255,255,255,0.02)]">
                  <tr className="border-b border-[color:var(--line)] text-left text-[color:var(--ink-faint)]">
                    <th className="w-6 px-2 py-2" />
                    <th className="px-3 py-2 font-semibold">Title</th>
                    <th className="px-3 py-2 font-semibold">Source</th>
                    <th className="px-3 py-2 font-semibold">Status</th>
                    <th className="px-3 py-2 font-semibold">Date</th>
                    <th className="px-3 py-2 text-right font-semibold">Words</th>
                    <th className="px-3 py-2" />
                  </tr>
                </thead>
                <tbody>
                  {items.map((doc) => {
                    const isExpanded = expanded.has(doc.document_id);
                    return (
                      <>
                        <tr key={doc.document_id} className="border-b border-[color:rgba(255,255,255,0.04)] hover:bg-[color:rgba(255,255,255,0.02)]">
                          <td className="px-2 py-2 text-center">
                            <button type="button" onClick={() => toggleExpand(doc.document_id)} className="text-[color:var(--ink-faint)] hover:text-[color:var(--ink)]">
                              {isExpanded ? "▲" : "▼"}
                            </button>
                          </td>
                          <td className="max-w-[260px] px-3 py-2">
                            {doc.url ? (
                              <a href={doc.url} target="_blank" rel="noopener noreferrer" className="font-medium text-[color:var(--ink)] hover:underline" title={doc.title}>{doc.title || doc.document_id}</a>
                            ) : (
                              <span className="font-medium text-[color:var(--ink)]" title={doc.title}>{doc.title || doc.document_id}</span>
                            )}
                            {doc.speaker && <span className="block text-[color:var(--ink-faint)]">{doc.speaker}</span>}
                          </td>
                          <td className="px-3 py-2 text-[color:var(--ink-faint)]">
                            <span title={doc.source_kind}>{doc.organization}</span>
                            <span className="ml-1 text-[10px] opacity-60">{fmtSourceKind(doc.source_kind)}</span>
                          </td>
                          <td className="px-3 py-2"><EnrichBadge status={doc.enrichment_status} /></td>
                          <td className="whitespace-nowrap px-3 py-2 text-[color:var(--ink-faint)]">{doc.date || "—"}</td>
                          <td className="px-3 py-2 text-right tabular-nums text-[color:var(--ink-faint)]">{doc.word_count > 0 ? doc.word_count.toLocaleString() : "—"}</td>
                          <td className="px-3 py-2">
                            <button
                              type="button"
                              onClick={() => handleDelete(doc)}
                              disabled={deleting === doc.document_id}
                              className="rounded border border-[color:rgba(255,107,127,0.4)] bg-[color:rgba(255,107,127,0.1)] px-2 py-0.5 text-[10px] font-semibold text-[color:var(--danger)] hover:bg-[color:rgba(255,107,127,0.2)] disabled:opacity-40"
                            >
                              {deleting === doc.document_id ? "…" : "Delete"}
                            </button>
                          </td>
                        </tr>
                        {isExpanded && (
                          <tr key={`${doc.document_id}-detail`} className="border-b border-[color:rgba(255,255,255,0.04)] bg-[color:rgba(255,255,255,0.015)]">
                            <td />
                            <td colSpan={6} className="px-3 py-3">
                              <div className="flex flex-col gap-2">
                                <p className="text-[10px] font-semibold uppercase tracking-wider text-[color:var(--ink-faint)]">Document ID: <span className="font-mono normal-case">{doc.document_id}</span></p>
                                {doc.tags.length > 0 && (
                                  <div className="flex flex-wrap gap-1">
                                    {doc.tags.slice(0, 12).map((t) => (
                                      <span key={t} className="rounded-full bg-[color:rgba(255,255,255,0.06)] px-2 py-0.5 text-[10px] text-[color:var(--ink-faint)]">{t}</span>
                                    ))}
                                    {doc.tags.length > 12 && <span className="text-[10px] text-[color:var(--ink-faint)]">+{doc.tags.length - 12} more</span>}
                                  </div>
                                )}
                                {doc.sentiment_label && (
                                  <p className="text-[10px] text-[color:var(--ink-faint)]">
                                    Sentiment: <span className={doc.sentiment_label === "positive" ? "text-[#41d39d]" : doc.sentiment_label === "negative" ? "text-[color:var(--danger)]" : "text-[color:var(--ink)]"}>{doc.sentiment_label}</span>
                                    {" "}({doc.sentiment_score.toFixed(2)})
                                  </p>
                                )}
                                <p className="text-[10px] text-[color:var(--ink-faint)]">Review: {doc.review_decision}</p>
                              </div>
                            </td>
                          </tr>
                        )}
                      </>
                    );
                  })}
                  {items.length === 0 && !loading && (
                    <tr><td colSpan={7} className="px-3 py-8 text-center text-xs text-[color:var(--ink-faint)]">No documents match.</td></tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="mt-3 flex items-center justify-between text-xs text-[color:var(--ink-faint)]">
                <button type="button" disabled={page === 1} onClick={() => handlePage(-1)} className="rounded-lg border border-[color:var(--line)] px-3 py-1 hover:border-[color:var(--accent)] disabled:opacity-40">← Prev</button>
                <span>Page {page} of {totalPages}</span>
                <button type="button" disabled={page >= totalPages} onClick={() => handlePage(1)} className="rounded-lg border border-[color:var(--line)] px-3 py-1 hover:border-[color:var(--accent)] disabled:opacity-40">Next →</button>
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}

/* ─── Manual Document Upload ───────────────────────────────────────── */
const SOURCE_KIND_OPTIONS = [
  "custom_document", "sec_speech", "sec_tm_faq", "sec_enforcement_litigation",
  "finra_regulatory_notice", "finra_comment_letter", "finra_awc",
  "doj_usao_press_release", "federal_reserve_speech_testimony", "cftc_press_release",
  "cftc_public_statement_remark", "sec_press_release_rss", "sec_administrative_proceeding",
  "sec_trading_suspension", "sec_federal_register", "sec_pcaob_rulemaking", "pcaob_update",
  "msrb_press_release", "treasury_featured_story", "treasury_press_release", "treasury_statement_remark",
  "sifma_news_item", "ici_news_item", "isda_news_item", "mfa_news_item", "fia_news_item",
  "aba_news_item", "bpi_news_item", "icba_news_item", "lsta_news_item", "congress_crs_product",
  "bloomberg_public_article", "substack_public_article", "jdsupra_article", "investmentnews_article",
  "citywire_article", "therecord_media_article", "wired_article", "tripwire_article",
  "akamai_blog_article", "ritholtz_article", "ft_portfolios_market_commentary",
  "liberty_street_economics_article", "wealth_of_common_sense_article", "wsj_dow_jones",
  "reddit_post", "hedge_fund_letter", "newsapi_article",
];

function ManualDocumentUploadSection() {
  const empty = { title: "", organization: "", source_kind: "custom_document", doc_type: "Document", speaker: "", date: "", url: "", content: "" };
  const [form, setForm] = useState(empty);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<{ ok: true; document_id: string } | { ok: false; error: string } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  function field(key: keyof typeof empty, val: string) {
    setForm((p) => ({ ...p, [key]: val }));
    setResult(null);
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      field("content", text);
      if (!form.title) field("title", file.name.replace(/\.[^.]+$/, "").replace(/[-_]/g, " "));
    };
    reader.readAsText(file);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!form.title.trim() || !form.content.trim()) return;
    setSubmitting(true);
    setResult(null);
    try {
      const res = await fetch("/api/admin/documents", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const d = await res.json() as { ok: boolean; document_id?: string; error?: string };
      if (d.ok) {
        setResult({ ok: true, document_id: d.document_id! });
        setForm(empty);
        if (fileRef.current) fileRef.current.value = "";
      } else {
        setResult({ ok: false, error: d.error ?? "Unknown error" });
      }
    } catch {
      setResult({ ok: false, error: "Network error" });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="mb-8">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Add Document</h2>
      <p className="mb-3 text-xs text-[color:var(--ink-faint)]">Manually add a document to the corpus. Paste text or load a .txt / .md file. Saved to <code className="rounded bg-[color:rgba(255,255,255,0.06)] px-1">custom_documents.json</code> in GCS.</p>

      <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.88)] px-4 py-4">
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          {/* Metadata */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="flex flex-col gap-1 sm:col-span-2">
              <label className="text-xs font-semibold text-[color:var(--ink-faint)]">Title <span className="text-[color:var(--danger)]">*</span></label>
              <input type="text" value={form.title} onChange={(e) => field("title", e.target.value)} placeholder="Document title" required className="form-control px-2 py-1.5 text-sm" />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-[color:var(--ink-faint)]">Organization</label>
              <input type="text" value={form.organization} onChange={(e) => field("organization", e.target.value)} placeholder="e.g. SEC, FINRA, Custom" className="form-control px-2 py-1.5 text-sm" />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-[color:var(--ink-faint)]">Source Kind</label>
              <select value={form.source_kind} onChange={(e) => field("source_kind", e.target.value)} className="form-control px-2 py-1.5 text-sm">
                {SOURCE_KIND_OPTIONS.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-[color:var(--ink-faint)]">Doc Type</label>
              <input type="text" value={form.doc_type} onChange={(e) => field("doc_type", e.target.value)} placeholder="e.g. Speech, Notice, Report" className="form-control px-2 py-1.5 text-sm" />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-[color:var(--ink-faint)]">Speaker / Author</label>
              <input type="text" value={form.speaker} onChange={(e) => field("speaker", e.target.value)} placeholder="Optional" className="form-control px-2 py-1.5 text-sm" />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-[color:var(--ink-faint)]">Date</label>
              <input type="date" value={form.date} onChange={(e) => field("date", e.target.value)} className="form-control px-2 py-1.5 text-sm" />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-[color:var(--ink-faint)]">URL</label>
              <input type="url" value={form.url} onChange={(e) => field("url", e.target.value)} placeholder="https://…" className="form-control px-2 py-1.5 text-sm" />
            </div>
          </div>

          {/* Content */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-[color:var(--ink-faint)]">Content <span className="text-[color:var(--danger)]">*</span></label>
              <label className="flex cursor-pointer items-center gap-1.5 text-xs text-[color:var(--accent)] hover:underline">
                <span>Load file</span>
                <input ref={fileRef} type="file" accept=".txt,.md,.text" onChange={handleFileChange} className="hidden" />
              </label>
            </div>
            <textarea
              value={form.content}
              onChange={(e) => field("content", e.target.value)}
              rows={10}
              placeholder="Paste the full document text here, or use Load file to read a .txt / .md file…"
              required
              className="form-control resize-y px-2 py-1.5 text-sm"
            />
            {form.content && (
              <p className="text-[10px] text-[color:var(--ink-faint)]">
                {form.content.trim().split(/\s+/).filter(Boolean).length.toLocaleString()} words · {form.content.split(/\n{2,}/).filter((p) => p.trim()).length} paragraphs
              </p>
            )}
          </div>

          {/* Actions */}
          <div className="flex flex-wrap items-center gap-4 border-t border-[color:var(--line)] pt-4">
            <button
              type="submit"
              disabled={submitting || !form.title.trim() || !form.content.trim()}
              className="btn-solid rounded-xl px-5 py-2 text-sm font-semibold disabled:opacity-40"
            >
              {submitting ? "Saving…" : "Add to Corpus"}
            </button>
            {result?.ok && (
              <span className="text-sm text-[color:var(--ok)]">
                Saved · ID: <code className="rounded bg-[color:rgba(255,255,255,0.06)] px-1 font-mono text-xs">{result.document_id}</code>
              </span>
            )}
            {result && !result.ok && (
              <span className="text-sm text-[color:var(--danger)]">{result.error}</span>
            )}
          </div>
        </form>
      </div>
    </section>
  );
}

/* ─── Divider ──────────────────────────────────────────────────────── */
function SectionDivider({ label }: { label: string }) {
  return (
    <div className="mb-8 flex items-center gap-3">
      <div className="h-px flex-1 bg-[color:var(--line)]" />
      <span className="text-xs font-bold uppercase tracking-[0.12em] text-[color:var(--ink-faint)]">{label}</span>
      <div className="h-px flex-1 bg-[color:var(--line)]" />
    </div>
  );
}

/* ─── Main page ────────────────────────────────────────────────────── */
export default function AdminPage() {
  const [tickers, setTickers] = useState<TickerEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<"idle" | "ok" | "error">("idle");
  const [saveError, setSaveError] = useState<string | null>(null);

  const [input, setInput] = useState("");
  const [validating, setValidating] = useState(false);
  const [preview, setPreview] = useState<ValidationResult | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch("/api/admin/ticker")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setTickers(data);
      })
      .finally(() => setLoading(false));
  }, []);

  async function handleConfirm() {
    const sym = input.trim().toUpperCase();
    if (!sym) return;
    if (tickers.length >= MAX) return;
    if (tickers.some((t) => t.symbol === sym)) {
      setPreview({ valid: false, error: `${sym} is already in the list` });
      return;
    }
    setValidating(true);
    setPreview(null);
    try {
      const res = await fetch("/api/admin/ticker/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: sym }),
      });
      const data: ValidationResult = await res.json();
      setPreview(data);
      if (data.valid) {
        setTickers((prev) => [...prev, { symbol: data.symbol, name: data.name }]);
        setInput("");
      }
    } finally {
      setValidating(false);
    }
  }

  function handleRename(symbol: string, name: string) {
    setTickers((prev) => prev.map((t) => (t.symbol === symbol ? { ...t, name } : t)));
    setSaveStatus("idle");
  }

  function handleRemove(symbol: string) {
    setTickers((prev) => prev.filter((t) => t.symbol !== symbol));
    setSaveStatus("idle");
  }

  async function handleSave() {
    setSaving(true);
    setSaveStatus("idle");
    setSaveError(null);
    try {
      const res = await fetch("/api/admin/ticker", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(tickers),
      });
      if (res.ok) {
        setSaveStatus("ok");
      } else {
        const body = await res.json().catch(() => ({}));
        setSaveError(body?.error ?? `HTTP ${res.status}`);
        setSaveStatus("error");
      }
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Network error");
      setSaveStatus("error");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-12">
      <p className="mb-1 text-xs font-bold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">Admin</p>
      <h1 className="mb-10 text-2xl font-bold text-[color:var(--ink)]">Pipeline Controls</h1>

      {/* ── Ticker bar ─────────────────────────────────────────────── */}
      <SectionDivider label="Ticker Bar" />

      <section className="mb-8">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
            Active Tickers
          </h2>
          <span className="text-xs text-[color:var(--ink-faint)]">
            {tickers.length} / {MAX}
          </span>
        </div>

        {loading ? (
          <p className="text-sm text-[color:var(--ink-faint)]">Loading…</p>
        ) : tickers.length === 0 ? (
          <p className="text-sm text-[color:var(--ink-faint)]">No tickers configured.</p>
        ) : (
          <ul className="space-y-2">
            {tickers.map((t) => (
              <li
                key={t.symbol}
                className="flex items-center gap-3 rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.88)] px-4 py-3"
              >
                <span className="w-14 flex-shrink-0 font-mono text-sm font-bold text-[color:var(--accent)]">
                  {t.symbol}
                </span>
                <input
                  type="text"
                  value={t.name}
                  onChange={(e) => handleRename(t.symbol, e.target.value)}
                  placeholder="Display name"
                  className="form-control min-w-0 flex-1 px-2 py-1 text-sm"
                />
                <button
                  type="button"
                  onClick={() => handleRemove(t.symbol)}
                  className="flex-shrink-0 rounded-lg border border-[color:rgba(255,107,127,0.4)] bg-[color:rgba(255,107,127,0.1)] px-3 py-1 text-xs font-semibold text-[color:var(--danger)] transition hover:bg-[color:rgba(255,107,127,0.2)]"
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="mb-8">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
          Add Ticker
        </h2>
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => {
              setInput(e.target.value.toUpperCase());
              setPreview(null);
            }}
            onKeyDown={(e) => e.key === "Enter" && handleConfirm()}
            placeholder="e.g. AAPL, SPY, ^VIX"
            disabled={tickers.length >= MAX}
            className="form-control flex-1 px-3 py-2 text-sm"
          />
          <button
            type="button"
            onClick={handleConfirm}
            disabled={!input.trim() || validating || tickers.length >= MAX}
            className="btn-solid min-w-[90px] rounded-xl px-4 py-2 text-sm disabled:opacity-40"
          >
            {validating ? "Checking…" : "Confirm"}
          </button>
        </div>

        {tickers.length >= MAX && (
          <p className="mt-2 text-xs text-[color:var(--warn)]">Maximum of {MAX} tickers reached.</p>
        )}

        {preview && (
          <div
            className={`mt-3 rounded-xl border px-4 py-3 text-sm ${
              preview.valid
                ? "border-[color:rgba(65,211,157,0.48)] bg-[color:rgba(65,211,157,0.08)] text-[color:var(--ok)]"
                : "border-[color:rgba(255,107,127,0.48)] bg-[color:rgba(255,107,127,0.08)] text-[color:var(--danger)]"
            }`}
          >
            {preview.valid ? (
              <span>
                <strong>{preview.symbol}</strong> — {preview.name}&nbsp;
                <span className="font-mono">
                  ${preview.price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
                &nbsp;
                <span className={preview.up ? "text-[color:var(--ok)]" : "text-[color:var(--danger)]"}>
                  {preview.up ? "▲" : "▼"} {Math.abs(preview.pct).toFixed(2)}%
                </span>
                &nbsp;— Added
              </span>
            ) : (
              preview.error
            )}
          </div>
        )}
      </section>

      <div className="mb-12 flex items-center gap-4">
        <button
          type="button"
          onClick={handleSave}
          disabled={saving || loading}
          className="btn-solid rounded-xl px-6 py-2.5 text-sm font-semibold disabled:opacity-40"
        >
          {saving ? "Saving…" : "Save Changes"}
        </button>
        {saveStatus === "ok" && (
          <span className="text-sm text-[color:var(--ok)]">Saved — ticker bar will update within 60 s</span>
        )}
        {saveStatus === "error" && (
          <span className="text-sm text-[color:var(--danger)]">
            Save failed{saveError ? `: ${saveError}` : " — try again"}
          </span>
        )}
      </div>

      {/* ── Knowledge Index ────────────────────────────────────────── */}
      <SectionDivider label="Knowledge Index" />
      <KnowledgeIndexSection />

      {/* ── Intel Feed ─────────────────────────────────────────────── */}
      <SectionDivider label="Connector Audit" />
      <ConnectorAuditSection />

      <SectionDivider label="Source Health" />
      <SourceHealthSection />

      <SectionDivider label="Intel Feed" />
      <XAccountManagerSection />
      <FeedManagerSection />
      <TopicRulesSection />

      {/* ── Enrichment Pipeline ────────────────────────────────────── */}
      <SectionDivider label="Enrichment Pipeline" />
      <EnrichmentPipelineSection />

      {/* ── Document Library ───────────────────────────────────────── */}
      <SectionDivider label="Document Library" />
      <DocumentLibrarySection />
      <ManualDocumentUploadSection />

      {/* ── Workflows ──────────────────────────────────────────────── */}
      <SectionDivider label="GitHub Actions" />

      <BloombergOnDemandSection />

      <SectionDivider label="YouTube Workflows" />

      <WorkflowPanel
        title="YouTube Ad Hoc Video"
        description="Paste one specific YouTube video URL. This runs once and enriches that video's transcript."
        workflowFile="youtube-video-ingest.yml"
        fields={YOUTUBE_VIDEO_FIELDS}
      />

      <YouTubeChannelManagerSection />

      <WorkflowPanel
        title="Rule and Comment Ingest"
        description="Ingest and enrich SEC or FINRA rule/comment URLs, then monitor submitted comment sources daily for 95 days."
        workflowFile="rule-comment-ingest.yml"
        fields={RULE_COMMENT_FIELDS}
      />

      <WorkflowPanel
        title="Policy Extraction"
        description="Crawl a regulatory source and extract new or updated documents into GCS."
        workflowFile="policy-extraction.yml"
        fields={POLICY_EXTRACTION_FIELDS}
      />

      <WorkflowPanel
        title="Financial News Ingest"
        description="Fetch new financial news articles from NewsAPI and save them to GCS."
        workflowFile="financial-news-ingest.yml"
        fields={NEWS_INGEST_FIELDS}
      />

      <WorkflowPanel
        title="Trends Aggregation"
        description="Recompute the daily trends report from all enriched documents and save to GCS."
        workflowFile="trends-daily.yml"
        fields={TRENDS_FIELDS}
      />
    </div>
  );
}
