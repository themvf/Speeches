"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

type HubMode = "home" | "operations" | "analytics" | "chats";
type JobStatus = "queued" | "running" | "success" | "failed" | "unknown";

interface ApiEnvelope<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

interface MetricsData {
  totals: { documents: number; organizations: number; enriched: number; pending_review: number };
  recent_ingest: { last_run_at: string; processed_count: number; failed_count: number };
  by_source_kind: Array<{ source_kind: string; count: number }>;
  runtime?: { data_source_mode: string; gcs_configured: boolean; github_actions_enabled: boolean };
}

interface DocumentItem {
  document_id: string;
  title: string;
  organization: string;
  source_kind: string;
  doc_type: string;
  url: string;
  date: string;
  published_at: string;
  word_count: number;
  topics: string[];
  enrichment_status: string;
  review_decision: string;
}

interface DocumentsFacets {
  sources: string[];
  organizations: string[];
  topics: string[];
  keywords: string[];
  statuses: string[];
}

interface DocumentsData {
  items: DocumentItem[];
  page: number;
  page_size: number;
  total: number;
  facets: DocumentsFacets;
}

interface JobState {
  job_id: string;
  status: JobStatus;
  workflow?: string;
  html_url?: string;
  updated_at?: string;
  conclusion?: string;
}

interface JobStartPayload {
  job_id: string;
  status: "queued";
}

interface IngestFormState {
  limit: number;
  lookback_days: number;
  selection: "new_or_updated" | "all";
}

interface EnrichFormState {
  limit: number;
  mode: "only_missing_or_failed" | "all";
  source_kind: string;
  heuristic_only: boolean;
  model: string;
}

interface NewsConnectorSettings {
  updated_at: string;
  query: string;
  lookback_days: number;
  max_pages: number;
  page_size: number;
  target_count: number;
  sort_by: string;
  organization_label: string;
  domains: string;
  exclude_domains: string;
  tags_csv: string;
}

interface NewsConnectorSaveResult {
  saved: boolean;
  local_saved: boolean;
  remote_saved: boolean;
  settings: NewsConnectorSettings;
}

interface ChatCitation {
  document_id: string;
  title: string;
  organization: string;
  source_kind: string;
  url: string;
  snippet: string;
}

interface ChatAnswerData {
  answer: string;
  citations: ChatCitation[];
}

interface ChatMessage {
  role: "assistant" | "user";
  content: string;
  citations?: ChatCitation[];
}

interface PolicyResearchHubProps {
  mode?: HubMode;
}

const EMPTY_FACETS: DocumentsFacets = { sources: [], organizations: [], topics: [], keywords: [], statuses: [] };
const DEFAULT_SETTINGS: NewsConnectorSettings = {
  updated_at: "",
  query: "federal reserve OR sec OR cftc OR finra OR doj antitrust OR enforcement OR regulation",
  lookback_days: 7,
  max_pages: 4,
  page_size: 50,
  target_count: 100,
  sort_by: "publishedAt",
  organization_label: "Financial News",
  domains: "",
  exclude_domains: "",
  tags_csv: "financial-regulation,policy,enforcement"
};

function fmt(value: number): string {
  return new Intl.NumberFormat("en-US").format(value || 0);
}

function fmtDate(value: string): string {
  if (!value) return "-";
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? value : d.toLocaleString("en-US", { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" });
}

function statusClass(value: string): string {
  const s = String(value || "").toLowerCase();
  if (["enriched", "reviewed", "success"].includes(s)) return "border-emerald-300 bg-emerald-50 text-emerald-800";
  if (["fallback_enriched", "queued", "running"].includes(s)) return "border-amber-300 bg-amber-50 text-amber-800";
  if (["failed", "rejected"].includes(s)) return "border-rose-300 bg-rose-50 text-rose-800";
  return "border-slate-300 bg-slate-50 text-slate-700";
}

function headerFor(mode: HubMode): { title: string; subtitle: string } {
  if (mode === "home") {
    return {
      title: "Regulatory Intelligence, Built For Decisions",
      subtitle: "Track policy and enforcement signals, triage incoming sources, and run ingestion and enrichment from one operational surface."
    };
  }
  if (mode === "operations") return { title: "Operations", subtitle: "Run extraction and enrichment workflows with connector settings and execution visibility." };
  if (mode === "analytics") return { title: "Analytics", subtitle: "Review source mix, corpus health, and pipeline outcomes." };
  return { title: "Agentic Chats", subtitle: "Ask policy questions and retrieve evidence-backed document context." };
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    cache: "no-store",
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers as Record<string, string> | undefined) }
  });
  const payload = (await res.json()) as ApiEnvelope<T>;
  if (!res.ok || !payload?.ok || !payload.data) throw new Error(payload?.error || `Request failed (${res.status})`);
  return payload.data;
}

export function PolicyResearchHub({ mode = "home" }: PolicyResearchHubProps) {
  const header = headerFor(mode);
  const needsDocs = mode === "home";
  const needsMetrics = mode === "analytics" || mode === "operations";
  const needsOps = mode === "operations";
  const needsChats = mode === "chats";

  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(needsMetrics);
  const [metricsError, setMetricsError] = useState("");

  const [items, setItems] = useState<DocumentItem[]>([]);
  const [facets, setFacets] = useState<DocumentsFacets>(EMPTY_FACETS);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [docsLoading, setDocsLoading] = useState(needsDocs);
  const [docsError, setDocsError] = useState("");
  const [q, setQ] = useState("");
  const [org, setOrg] = useState("");
  const [source, setSource] = useState("");
  const [topic, setTopic] = useState("");
  const [keyword, setKeyword] = useState("");
  const [status, setStatus] = useState("");
  const [sort, setSort] = useState("updated_desc");

  const [settings, setSettings] = useState<NewsConnectorSettings>(DEFAULT_SETTINGS);
  const [settingsLoading, setSettingsLoading] = useState(needsOps);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [settingsMsg, setSettingsMsg] = useState("");
  const [settingsErr, setSettingsErr] = useState("");

  const [ingest, setIngest] = useState<IngestFormState>({ limit: 10, lookback_days: 7, selection: "new_or_updated" });
  const [enrich, setEnrich] = useState<EnrichFormState>({
    limit: 25,
    mode: "only_missing_or_failed",
    source_kind: "newsapi_article",
    heuristic_only: false,
    model: ""
  });
  const [runAction, setRunAction] = useState<"ingest" | "enrich" | null>(null);
  const [activeJob, setActiveJob] = useState<JobState | null>(null);
  const [jobError, setJobError] = useState("");

  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "assistant", content: "Ask a policy question and I will retrieve relevant documents from your corpus." }
  ]);
  const [prompt, setPrompt] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");

  const sourceMix = useMemo(() => {
    const rows = metrics?.by_source_kind || [];
    const max = rows.reduce((best, row) => (row.count > best ? row.count : best), 1);
    return rows.map((row) => ({ ...row, width: Math.max(8, Math.round((row.count / max) * 100)) }));
  }, [metrics?.by_source_kind]);

  const totalPages = Math.max(1, Math.ceil(total / Math.max(1, pageSize)));

  const loadMetrics = useCallback(async () => {
    if (!needsMetrics) return;
    setMetricsLoading(true);
    setMetricsError("");
    try {
      setMetrics(await fetchJson<MetricsData>("/api/metrics"));
    } catch (err) {
      setMetricsError(err instanceof Error ? err.message : "Failed to load metrics.");
    } finally {
      setMetricsLoading(false);
    }
  }, [needsMetrics]);

  const loadDocs = useCallback(async () => {
    if (!needsDocs) return;
    setDocsLoading(true);
    setDocsError("");
    try {
      const params = new URLSearchParams({ page: String(page), page_size: String(pageSize), sort });
      if (q.trim()) params.set("q", q.trim());
      if (org.trim()) params.set("org", org.trim());
      if (source.trim()) params.set("source_kind", source.trim());
      if (topic.trim()) params.set("topic", topic.trim());
      if (keyword.trim()) params.set("keyword", keyword.trim());
      if (status.trim()) params.set("status", status.trim());
      const data = await fetchJson<DocumentsData>(`/api/documents?${params.toString()}`);
      setItems(data.items || []);
      setFacets(data.facets || EMPTY_FACETS);
      setTotal(data.total || 0);
      setPage(data.page || 1);
      setPageSize(data.page_size || 20);
    } catch (err) {
      setDocsError(err instanceof Error ? err.message : "Failed to load documents.");
    } finally {
      setDocsLoading(false);
    }
  }, [keyword, needsDocs, org, page, pageSize, q, sort, source, status, topic]);

  const loadSettings = useCallback(async () => {
    if (!needsOps) return;
    setSettingsLoading(true);
    setSettingsErr("");
    try {
      const data = await fetchJson<NewsConnectorSettings>("/api/settings/connectors/news");
      setSettings(data);
      setIngest((prev) => ({ ...prev, lookback_days: data.lookback_days }));
    } catch (err) {
      setSettingsErr(err instanceof Error ? err.message : "Failed to load settings.");
    } finally {
      setSettingsLoading(false);
    }
  }, [needsOps]);

  const saveSettings = useCallback(async () => {
    setSettingsSaving(true);
    setSettingsMsg("");
    setSettingsErr("");
    try {
      const result = await fetchJson<NewsConnectorSaveResult>("/api/settings/connectors/news", { method: "PUT", body: JSON.stringify(settings) });
      setSettings(result.settings);
      setSettingsMsg(result.saved ? `Settings saved (${result.remote_saved ? "remote" : "local"}).` : "Save failed.");
    } catch (err) {
      setSettingsErr(err instanceof Error ? err.message : "Failed to save settings.");
    } finally {
      setSettingsSaving(false);
    }
  }, [settings]);

  const launch = useCallback(
    async (kind: "ingest" | "enrich") => {
      setRunAction(kind);
      setJobError("");
      try {
        const payload = await fetchJson<JobStartPayload>(`/api/jobs/${kind}`, {
          method: "POST",
          body: JSON.stringify(kind === "ingest" ? ingest : enrich)
        });
        setActiveJob({ job_id: payload.job_id, status: payload.status });
      } catch (err) {
        setJobError(err instanceof Error ? err.message : `Failed to run ${kind}.`);
      } finally {
        setRunAction(null);
      }
    },
    [enrich, ingest]
  );

  const ask = useCallback(async () => {
    if (!needsChats || !prompt.trim()) return;
    const userPrompt = prompt.trim();
    setPrompt("");
    setChatError("");
    setChatLoading(true);
    setMessages((prev) => [...prev, { role: "user", content: userPrompt }]);
    try {
      const data = await fetchJson<ChatAnswerData>("/api/chats/ask", { method: "POST", body: JSON.stringify({ prompt: userPrompt, top_k: 5 }) });
      setMessages((prev) => [...prev, { role: "assistant", content: data.answer, citations: data.citations }]);
    } catch (err) {
      setChatError(err instanceof Error ? err.message : "Failed to run chat.");
    } finally {
      setChatLoading(false);
    }
  }, [needsChats, prompt]);

  useEffect(() => {
    if (needsMetrics) void loadMetrics();
  }, [loadMetrics, needsMetrics]);

  useEffect(() => {
    if (!needsDocs) return;
    const t = setTimeout(() => void loadDocs(), 200);
    return () => clearTimeout(t);
  }, [loadDocs, needsDocs]);

  useEffect(() => {
    if (needsOps) void loadSettings();
  }, [loadSettings, needsOps]);

  useEffect(() => {
    if (!needsOps || !activeJob?.job_id || !["queued", "running", "unknown"].includes(activeJob.status)) return;
    let canceled = false;
    const poll = async () => {
      try {
        const job = await fetchJson<JobState>(`/api/jobs/${activeJob.job_id}`);
        if (!canceled) {
          setActiveJob(job);
          if (["success", "failed"].includes(job.status)) {
            void loadMetrics();
            void loadDocs();
          }
        }
      } catch (err) {
        if (!canceled) setJobError(err instanceof Error ? err.message : "Failed to refresh job status.");
      }
    };
    void poll();
    const id = setInterval(() => void poll(), 4000);
    return () => {
      canceled = true;
      clearInterval(id);
    };
  }, [activeJob?.job_id, activeJob?.status, loadDocs, loadMetrics, needsOps]);

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-6 md:px-8 md:py-10">
      <header className="panel p-6 md:p-8">
        <span className="kicker">Policy Research Hub</span>
        <h1 className="mt-3 text-3xl font-bold leading-tight md:text-5xl">{header.title}</h1>
        <p className="mt-3 max-w-3xl text-base text-[color:rgba(16,36,59,0.76)] md:text-lg">{header.subtitle}</p>
      </header>

      {mode === "home" ? (
        <section className="panel p-5">
          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-7">
            <input className="rounded-xl border border-[color:var(--line)] px-3 py-2 text-sm xl:col-span-2" placeholder="Search text" value={q} onChange={(e) => { setPage(1); setQ(e.target.value); }} />
            <select className="rounded-xl border border-[color:var(--line)] px-3 py-2 text-sm" value={source} onChange={(e) => { setPage(1); setSource(e.target.value); }}>
              <option value="">All sources</option>{facets.sources.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
            <select className="rounded-xl border border-[color:var(--line)] px-3 py-2 text-sm" value={org} onChange={(e) => { setPage(1); setOrg(e.target.value); }}>
              <option value="">All orgs</option>{facets.organizations.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
            <select className="rounded-xl border border-[color:var(--line)] px-3 py-2 text-sm" value={topic} onChange={(e) => { setPage(1); setTopic(e.target.value); }}>
              <option value="">All topics</option>{facets.topics.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
            <input className="rounded-xl border border-[color:var(--line)] px-3 py-2 text-sm" placeholder="Keyword" value={keyword} onChange={(e) => { setPage(1); setKeyword(e.target.value); }} />
            <select className="rounded-xl border border-[color:var(--line)] px-3 py-2 text-sm" value={status} onChange={(e) => { setPage(1); setStatus(e.target.value); }}>
              <option value="">All statuses</option><option value="not_enriched">Not Enriched</option><option value="enriched">Enriched</option><option value="fallback_enriched">Fallback Enriched</option><option value="reviewed">Reviewed</option>
            </select>
          </div>
          <div className="mt-2 flex items-center justify-between">
            <p className="text-sm text-[color:rgba(16,36,59,0.7)]">{fmt(total)} matching documents</p>
            <select className="rounded-xl border border-[color:var(--line)] px-3 py-1.5 text-sm" value={sort} onChange={(e) => { setPage(1); setSort(e.target.value); }}>
              <option value="updated_desc">Recently Updated</option><option value="date_desc">Newest Published</option><option value="date_asc">Oldest Published</option>
            </select>
          </div>
          {docsError ? <p className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{docsError}</p> : null}
          <div className="mt-3 overflow-x-auto rounded-xl border border-[color:var(--line)] bg-white">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-[color:rgba(16,36,59,0.05)] text-xs uppercase tracking-[0.08em] text-[color:rgba(16,36,59,0.72)]"><tr><th className="px-3 py-2">Title</th><th className="px-3 py-2">Source</th><th className="px-3 py-2">Topics</th><th className="px-3 py-2">Status</th><th className="px-3 py-2">Published</th></tr></thead>
              <tbody>
                {docsLoading ? <tr><td colSpan={5} className="px-3 py-4 text-sm">Loading feed...</td></tr> : items.length === 0 ? <tr><td colSpan={5} className="px-3 py-4 text-sm">No documents match these filters.</td></tr> : items.map((d) => (
                  <tr key={d.document_id} className="border-t border-[color:var(--line)] align-top">
                    <td className="px-3 py-3"><p className="font-semibold">{d.title || "Untitled"}</p><p className="mt-1 text-xs text-[color:rgba(16,36,59,0.66)]">{d.organization} - {d.doc_type || "Document"}</p>{d.url ? <a href={d.url} target="_blank" rel="noreferrer" className="mt-1 inline-block text-xs text-[color:#2d5673] hover:underline">Open source</a> : null}</td>
                    <td className="px-3 py-3 text-xs"><span className="rounded-full border border-[color:var(--line)] bg-[color:rgba(16,36,59,0.04)] px-2 py-1">{d.source_kind || "unknown"}</span><p className="mt-2">{fmt(d.word_count)} words</p></td>
                    <td className="px-3 py-3 text-xs">{(d.topics || []).slice(0, 3).join(", ") || "-"}</td>
                    <td className="px-3 py-3 text-xs"><span className={`rounded-full border px-2 py-1 ${statusClass(d.enrichment_status)}`}>{d.enrichment_status || "not_enriched"}</span><p className="mt-2">Review: {d.review_decision || "pending"}</p></td>
                    <td className="px-3 py-3 text-xs">{fmtDate(d.published_at || d.date)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-3 flex items-center justify-between text-xs">
            <p>Page {page} of {totalPages}</p>
            <div className="flex gap-2">
              <button className="rounded-xl border border-[color:var(--line)] bg-white px-3 py-1.5 font-semibold disabled:opacity-50" disabled={page <= 1 || docsLoading} onClick={() => setPage(Math.max(1, page - 1))}>Previous</button>
              <button className="rounded-xl border border-[color:var(--line)] bg-white px-3 py-1.5 font-semibold disabled:opacity-50" disabled={page >= totalPages || docsLoading} onClick={() => setPage(Math.min(totalPages, page + 1))}>Next</button>
            </div>
          </div>
        </section>
      ) : null}

      {mode === "operations" ? (
        <section className="grid gap-4">
          <article className="panel p-5">
            <div className="flex items-center justify-between"><h2 className="text-xl font-semibold">News Connector Settings</h2><div className="flex gap-2"><button className="rounded-xl border border-[color:var(--line)] bg-white px-3 py-1.5 text-xs font-semibold" onClick={() => void loadSettings()}>Reload</button><button className="rounded-xl bg-[color:#2d5673] px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50" disabled={settingsSaving} onClick={() => void saveSettings()}>{settingsSaving ? "Saving..." : "Save Settings"}</button></div></div>
            {settingsLoading ? <p className="mt-2 text-sm">Loading settings...</p> : null}
            {settingsErr ? <p className="mt-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{settingsErr}</p> : null}
            {settingsMsg ? <p className="mt-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-700">{settingsMsg}</p> : null}
            <div className="mt-3 grid gap-2 md:grid-cols-2">
              <input className="rounded-xl border border-[color:var(--line)] px-3 py-2 text-sm md:col-span-2" value={settings.query} onChange={(e) => setSettings({ ...settings, query: e.target.value })} placeholder="Query" />
              <input className="rounded-xl border border-[color:var(--line)] px-3 py-2 text-sm" value={settings.organization_label} onChange={(e) => setSettings({ ...settings, organization_label: e.target.value })} placeholder="Organization label" />
              <select className="rounded-xl border border-[color:var(--line)] px-3 py-2 text-sm" value={settings.sort_by} onChange={(e) => setSettings({ ...settings, sort_by: e.target.value })}><option value="publishedAt">publishedAt</option><option value="relevancy">relevancy</option><option value="popularity">popularity</option></select>
              <input type="number" min={1} max={30} className="rounded-xl border border-[color:var(--line)] px-3 py-2 text-sm" value={settings.lookback_days} onChange={(e) => setSettings({ ...settings, lookback_days: Math.max(1, Math.min(30, Number.parseInt(e.target.value || "7", 10) || 7)) })} placeholder="Lookback days" />
              <input type="number" min={1} max={10} className="rounded-xl border border-[color:var(--line)] px-3 py-2 text-sm" value={settings.max_pages} onChange={(e) => setSettings({ ...settings, max_pages: Math.max(1, Math.min(10, Number.parseInt(e.target.value || "4", 10) || 4)) })} placeholder="Max pages" />
              <input type="number" min={10} max={100} className="rounded-xl border border-[color:var(--line)] px-3 py-2 text-sm" value={settings.page_size} onChange={(e) => setSettings({ ...settings, page_size: Math.max(10, Math.min(100, Number.parseInt(e.target.value || "50", 10) || 50)) })} placeholder="Page size" />
              <input type="number" min={10} max={500} className="rounded-xl border border-[color:var(--line)] px-3 py-2 text-sm" value={settings.target_count} onChange={(e) => setSettings({ ...settings, target_count: Math.max(10, Math.min(500, Number.parseInt(e.target.value || "100", 10) || 100)) })} placeholder="Target count" />
              <input className="rounded-xl border border-[color:var(--line)] px-3 py-2 text-sm md:col-span-2" value={settings.domains} onChange={(e) => setSettings({ ...settings, domains: e.target.value })} placeholder="Domains CSV" />
              <input className="rounded-xl border border-[color:var(--line)] px-3 py-2 text-sm md:col-span-2" value={settings.exclude_domains} onChange={(e) => setSettings({ ...settings, exclude_domains: e.target.value })} placeholder="Exclude domains CSV" />
              <input className="rounded-xl border border-[color:var(--line)] px-3 py-2 text-sm md:col-span-2" value={settings.tags_csv} onChange={(e) => setSettings({ ...settings, tags_csv: e.target.value })} placeholder="Tags CSV" />
            </div>
          </article>
          <section className="grid gap-4 md:grid-cols-2">
            <article className="panel p-5"><h2 className="text-xl font-semibold">Run Ingest</h2><div className="mt-3 grid grid-cols-2 gap-2 text-sm"><input type="number" min={1} className="rounded-lg border border-[color:var(--line)] px-2 py-1.5" value={ingest.limit} onChange={(e) => setIngest({ ...ingest, limit: Math.max(1, Number.parseInt(e.target.value || "10", 10) || 10) })} /><input type="number" min={1} className="rounded-lg border border-[color:var(--line)] px-2 py-1.5" value={ingest.lookback_days} onChange={(e) => setIngest({ ...ingest, lookback_days: Math.max(1, Number.parseInt(e.target.value || "7", 10) || 7) })} /><select className="col-span-2 rounded-lg border border-[color:var(--line)] px-2 py-1.5" value={ingest.selection} onChange={(e) => setIngest({ ...ingest, selection: e.target.value === "all" ? "all" : "new_or_updated" })}><option value="new_or_updated">new_or_updated</option><option value="all">all</option></select></div><button className="mt-3 w-full rounded-xl bg-[color:#2d5673] px-3 py-2 text-sm font-semibold text-white disabled:opacity-50" disabled={runAction !== null} onClick={() => void launch("ingest")}>{runAction === "ingest" ? "Launching..." : "Run Ingest"}</button></article>
            <article className="panel p-5"><h2 className="text-xl font-semibold">Run Enrichment</h2><div className="mt-3 grid grid-cols-2 gap-2 text-sm"><input type="number" min={1} className="rounded-lg border border-[color:var(--line)] px-2 py-1.5" value={enrich.limit} onChange={(e) => setEnrich({ ...enrich, limit: Math.max(1, Number.parseInt(e.target.value || "25", 10) || 25) })} /><input className="rounded-lg border border-[color:var(--line)] px-2 py-1.5" value={enrich.source_kind} onChange={(e) => setEnrich({ ...enrich, source_kind: e.target.value || "newsapi_article" })} /><select className="rounded-lg border border-[color:var(--line)] px-2 py-1.5" value={enrich.mode} onChange={(e) => setEnrich({ ...enrich, mode: e.target.value === "all" ? "all" : "only_missing_or_failed" })}><option value="only_missing_or_failed">only_missing_or_failed</option><option value="all">all</option></select><input className="rounded-lg border border-[color:var(--line)] px-2 py-1.5" placeholder="model (optional)" value={enrich.model} onChange={(e) => setEnrich({ ...enrich, model: e.target.value })} /><label className="col-span-2 flex items-center gap-2 text-xs"><input type="checkbox" checked={enrich.heuristic_only} onChange={(e) => setEnrich({ ...enrich, heuristic_only: e.target.checked })} />Heuristic-only</label></div><button className="mt-3 w-full rounded-xl bg-[color:#c77d28] px-3 py-2 text-sm font-semibold text-white disabled:opacity-50" disabled={runAction !== null} onClick={() => void launch("enrich")}>{runAction === "enrich" ? "Launching..." : "Run Enrichment"}</button></article>
          </section>
          <article className="panel p-5">
            <h2 className="text-xl font-semibold">Active Job</h2>
            {!activeJob ? <p className="mt-2 text-sm">No active run yet.</p> : <div className="mt-2 space-y-1 text-sm"><span className={`rounded-full border px-2 py-1 text-xs ${statusClass(activeJob.status)}`}>{activeJob.status}</span><p>Job: {activeJob.job_id}</p>{activeJob.workflow ? <p>Workflow: {activeJob.workflow}</p> : null}{activeJob.updated_at ? <p>Updated: {fmtDate(activeJob.updated_at)}</p> : null}{activeJob.conclusion ? <p>Conclusion: {activeJob.conclusion}</p> : null}{activeJob.html_url ? <a href={activeJob.html_url} target="_blank" rel="noreferrer" className="text-[color:#2d5673] hover:underline">Open GitHub run</a> : null}</div>}
            {jobError ? <p className="mt-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{jobError}</p> : null}
          </article>
        </section>
      ) : null}

      {mode === "analytics" ? (
        <section className="grid gap-4">
          <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <article className="panel p-4"><p className="text-xs uppercase tracking-[0.1em]">Documents</p><p className="mt-1 text-2xl font-semibold">{metricsLoading ? "..." : fmt(metrics?.totals.documents || 0)}</p></article>
            <article className="panel p-4"><p className="text-xs uppercase tracking-[0.1em]">Organizations</p><p className="mt-1 text-2xl font-semibold">{metricsLoading ? "..." : fmt(metrics?.totals.organizations || 0)}</p></article>
            <article className="panel p-4"><p className="text-xs uppercase tracking-[0.1em]">Enriched</p><p className="mt-1 text-2xl font-semibold">{metricsLoading ? "..." : fmt(metrics?.totals.enriched || 0)}</p></article>
            <article className="panel p-4"><p className="text-xs uppercase tracking-[0.1em]">Pending Review</p><p className="mt-1 text-2xl font-semibold">{metricsLoading ? "..." : fmt(metrics?.totals.pending_review || 0)}</p></article>
          </section>
          {metricsError ? <p className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{metricsError}</p> : null}
          <section className="grid gap-4 xl:grid-cols-[1.55fr_1fr]">
            <article className="panel p-5"><h2 className="text-xl font-semibold">Source Mix</h2><div className="mt-3 space-y-2">{sourceMix.length === 0 ? <p className="text-sm">No source telemetry yet.</p> : sourceMix.map((row) => <div key={row.source_kind}><div className="mb-1 flex justify-between text-xs"><span>{row.source_kind}</span><span>{fmt(row.count)}</span></div><div className="h-2 rounded-full bg-[color:rgba(16,36,59,0.1)]"><div className="h-2 rounded-full bg-[linear-gradient(90deg,#2d5673,#c77d28)]" style={{ width: `${row.width}%` }} /></div></div>)}</div></article>
            <article className="panel p-5"><h2 className="text-xl font-semibold">Pipeline Snapshot</h2><p className="mt-2 text-sm">Last run: <strong>{fmtDate(metrics?.recent_ingest.last_run_at || "")}</strong></p><div className="mt-3 grid grid-cols-2 gap-2 text-sm"><div className="rounded-xl border border-[color:var(--line)] bg-white px-3 py-2"><p className="text-xs">Processed</p><p className="text-lg font-semibold">{fmt(metrics?.recent_ingest.processed_count || 0)}</p></div><div className="rounded-xl border border-[color:var(--line)] bg-white px-3 py-2"><p className="text-xs">Failed</p><p className="text-lg font-semibold">{fmt(metrics?.recent_ingest.failed_count || 0)}</p></div></div></article>
          </section>
        </section>
      ) : null}

      {mode === "chats" ? (
        <section className="grid gap-4 xl:grid-cols-[1.35fr_1fr]">
          <article className="panel p-5">
            <div className="max-h-[480px] space-y-3 overflow-y-auto rounded-xl border border-[color:var(--line)] bg-white p-3">
              {messages.map((m, idx) => (
                <article key={`${m.role}_${idx}`} className={`rounded-xl border px-3 py-2 ${m.role === "assistant" ? "border-[color:rgba(45,86,115,0.22)] bg-[color:rgba(45,86,115,0.07)]" : "border-[color:rgba(199,125,40,0.28)] bg-[color:rgba(199,125,40,0.08)]"}`}>
                  <p className="text-xs font-semibold uppercase">{m.role}</p>
                  <p className="mt-1 whitespace-pre-wrap text-sm">{m.content}</p>
                  {m.citations?.length ? <div className="mt-2 space-y-2">{m.citations.map((c) => <div key={c.document_id} className="rounded-lg border border-[color:var(--line)] bg-white px-2 py-2 text-xs"><p className="font-semibold">{c.title}</p><p className="mt-1">{c.organization} - {c.source_kind}</p>{c.snippet ? <p className="mt-1">{c.snippet}</p> : null}{c.url ? <a href={c.url} target="_blank" rel="noreferrer" className="mt-1 inline-block text-[color:#2d5673] hover:underline">Open source</a> : null}</div>)}</div> : null}
                </article>
              ))}
            </div>
            {chatError ? <p className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{chatError}</p> : null}
            <div className="mt-3 flex gap-2"><input className="w-full rounded-xl border border-[color:var(--line)] px-3 py-2 text-sm" value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void ask(); } }} placeholder="Ask a policy question..." /><button className="rounded-xl bg-[color:#2d5673] px-4 py-2 text-sm font-semibold text-white disabled:opacity-50" onClick={() => void ask()} disabled={chatLoading}>{chatLoading ? "Thinking..." : "Ask"}</button></div>
          </article>
          <article className="panel p-5"><h2 className="text-xl font-semibold">Prompt Ideas</h2><div className="mt-3 space-y-2 text-sm"><p className="rounded-xl border border-[color:var(--line)] bg-white px-3 py-2">Which themes are rising in SEC speeches this month?</p><p className="rounded-xl border border-[color:var(--line)] bg-white px-3 py-2">Summarize enforcement trends across SEC and DOJ sources.</p><p className="rounded-xl border border-[color:var(--line)] bg-white px-3 py-2">Find documents mentioning capital requirements and liquidity risks.</p></div></article>
        </section>
      ) : null}
    </div>
  );
}
