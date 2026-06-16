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
type RssFeed = { id: number; label: string; feed_url: string; active: boolean };
type TopicRule = { id: number; topic_key: string; label: string; keywords: string; active: boolean; sort_order: number };

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
      { value: "sec_enforcement_litigation", label: "SEC Enforcement Litigation" },
      { value: "finra_regulatory_notice", label: "FINRA Regulatory Notice" },
      { value: "finra_key_topic", label: "FINRA Key Topic" },
      { value: "doj_usao_press_release", label: "DOJ USAO Press Release" },
      { value: "federal_reserve_speech_testimony", label: "Federal Reserve Speech / Testimony" },
      { value: "cftc_press_release", label: "CFTC Press Release" },
      { value: "cftc_public_statement_remark", label: "CFTC Public Statement / Remark" },
      { value: "congress_crs_product", label: "Congress CRS Product" },
      { value: "finra_comment_letter", label: "FINRA Rule Comment Letter" },
      { value: "finra_awc", label: "FINRA AWC Disciplinary Actions" },
      { value: "treasury_featured_story", label: "Treasury Featured Story" },
      { value: "treasury_press_release", label: "Treasury Press Release" },
      { value: "treasury_statement_remark", label: "Treasury Statement / Remark" },
      { value: "sifma_news_item", label: "SIFMA News" },
      { value: "jdsupra_article", label: "Trade Media: JD Supra" },
      { value: "investmentnews_article", label: "Trade Media: InvestmentNews" },
      { value: "citywire_article", label: "Trade Media: Citywire" },
      { value: "wsj_dow_jones", label: "WSJ / Dow Jones RSS" },
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


const TRENDS_FIELDS: FieldDef[] = [
  { name: "min_mentions", label: "Min tag mentions", type: "number", default: "5" },
  { name: "dry_run", label: "Dry run (skip OpenAI calls)", type: "boolean", default: "false" },
];

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
      <p className="mb-3 text-xs text-[color:var(--ink-faint)]">Manage Intel Feed RSS sources. Changes apply on the next 10-minute cron refresh.</p>
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

/* ─── Topic Rules Manager ──────────────────────────────────────────── */
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

  const [enrich, setEnrich] = useState({ source_kind: "newsapi_article", enrich_limit: "25", mode: "only_missing_or_failed", heuristic_only: "false", model: "" });
  const [dispatching, setDispatching] = useState(false);
  const [dispatchStatus, setDispatchStatus] = useState<"idle" | "ok" | "error">("idle");
  const [dispatchError, setDispatchError] = useState<string | null>(null);

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

        {/* Dispatch controls */}
        <div className={`${data ? "border-t border-[color:var(--line)] pt-4" : ""}`}>
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
              <label className="text-xs text-[color:var(--ink-faint)]">Model override</label>
              <input
                type="text"
                value={enrich.model}
                onChange={(e) => setEnrich((p) => ({ ...p, model: e.target.value }))}
                placeholder="e.g. gpt-4o (leave blank for default)"
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
              <span className="text-xs text-[color:var(--ink-faint)]">Skip OpenAI (heuristic only)</span>
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
  "finra_regulatory_notice", "finra_key_topic", "finra_comment_letter", "finra_awc",
  "doj_usao_press_release", "federal_reserve_speech_testimony", "cftc_press_release",
  "cftc_public_statement_remark", "treasury_press_release", "treasury_statement_remark",
  "sifma_news_item", "congress_crs_product",
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
      <SectionDivider label="Intel Feed" />
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
