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
  runtime?: {
    job_execution_mode?: "github_actions" | "local";
    data_source_mode: string;
    gcs_configured: boolean;
    github_actions_enabled: boolean;
    github_actions_enabled_flag?: boolean;
    github_actions_missing_required_env?: string[];
  };
}

interface DocumentItem {
  document_id: string;
  title: string;
  organization: string;
  source_kind: string;
  doc_type: string;
  speaker: string;
  url: string;
  date: string;
  published_at: string;
  word_count: number;
  keywords: string[];
  topics: string[];
  enrichment_status: string;
  review_decision: string;
}

interface DocumentsFacets {
  sources: string[];
  organizations: string[];
  topics: string[];
  key_topics: string[];
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
  status: JobStatus;
  workflow?: string;
  updated_at?: string;
  conclusion?: string;
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

interface ExtractFormState {
  connector:
    | "sec_speech"
    | "sec_tm_faq"
    | "sec_enforcement_litigation"
    | "finra_regulatory_notice"
    | "finra_comment_letter"
    | "finra_key_topic"
    | "doj_usao_press_release"
    | "federal_reserve_speech_testimony";
  selection: "new_or_updated" | "all";
  limit: number;
  max_pages: number;
  base_url: string;
  include_pdfs: boolean;
  include_rss: boolean;
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
  published_at?: string;
  url: string;
  snippet: string;
}

interface ChatAnswerData {
  answer: string;
  citations: ChatCitation[];
  retrieved_count?: number;
  model?: string;
}

interface ChatMessage {
  role: "assistant" | "user";
  content: string;
  citations?: ChatCitation[];
  model?: string;
}

interface PolicyResearchHubProps {
  mode?: HubMode;
}

const EMPTY_FACETS: DocumentsFacets = { sources: [], organizations: [], topics: [], key_topics: [], keywords: [], statuses: [] };
const DEFAULT_SETTINGS: NewsConnectorSettings = {
  updated_at: "",
  query: "federal reserve OR sec OR cftc OR finra OR doj antitrust OR enforcement OR regulation",
  lookback_days: 7,
  max_pages: 4,
  page_size: 50,
  target_count: 100,
  sort_by: "publishedAt",
  organization_label: "News",
  domains: "",
  exclude_domains: "",
  tags_csv: "financial-regulation,policy,enforcement"
};
const FINRA_COMMENT_NOTICE_EXAMPLE = "https://www.finra.org/rules-guidance/notices/26-06";

function fmt(value: number): string {
  return new Intl.NumberFormat("en-US").format(value || 0);
}

function fmtDate(value: string): string {
  if (!value) return "-";
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? value : d.toLocaleString("en-US", { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" });
}

function fmtDateOnly(value: string): string {
  if (!value) return "-";
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? value : d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function statusClass(value: string): string {
  const s = String(value || "").toLowerCase();
  if (["enriched", "reviewed", "success"].includes(s)) return "status-chip status-success";
  if (["fallback_enriched", "queued", "running"].includes(s)) return "status-chip status-warn";
  if (["failed", "rejected"].includes(s)) return "status-chip status-failure";
  return "status-chip status-neutral";
}

function displayOrganization(value: string): string {
  const raw = String(value || "").trim();
  if (!raw) {
    return "Unknown";
  }
  const lowered = raw.toLowerCase();
  if (lowered === "financial news" || lowered === "financials news") {
    return "News";
  }
  return raw;
}

function exactSpeakerName(item: DocumentItem): string {
  const raw = String(item.speaker || "").trim();
  return raw || "-";
}

const SOURCE_KIND_LABELS: Record<string, string> = {
  sec_speech: "SEC Speeches & Statements",
  sec_tm_faq: "SEC Trading & Markets FAQ",
  sec_enforcement_litigation: "SEC Enforcement Litigation",
  finra_regulatory_notice: "FINRA Regulatory Notices",
  finra_comment_letter: "FINRA Comment Letters",
  finra_key_topic: "FINRA Key Topics",
  doj_usao_press_release: "DOJ USAO Press Releases",
  federal_reserve_speech_testimony: "Federal Reserve Speeches/Testimony",
  newsapi_article: "News",
  uploaded: "Uploaded"
};

function displaySourceKind(value: string): string {
  const raw = String(value || "").trim();
  if (!raw) {
    return "Unknown";
  }
  const mapped = SOURCE_KIND_LABELS[raw];
  if (mapped) {
    return mapped;
  }
  return raw
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

const SOURCE_KIND_TYPE_LABELS: Record<string, string> = {
  sec_speech: "Speech",
  sec_tm_faq: "FAQ",
  sec_enforcement_litigation: "Litigation Release",
  finra_regulatory_notice: "Regulatory Notice",
  finra_comment_letter: "Comment Letter",
  finra_key_topic: "Key Topic",
  doj_usao_press_release: "Press Release",
  federal_reserve_speech_testimony: "Testimony",
  newsapi_article: "News Article",
  uploaded: "Uploaded Document"
};

function normalizeTypeLabel(value: string): string {
  const normalized = String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!normalized) {
    return "";
  }
  return normalized
    .split(" ")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ");
}

function displayType(docType: string, sourceKind: string): string {
  const normalized = normalizeTypeLabel(docType);
  if (normalized) {
    return normalized;
  }
  return SOURCE_KIND_TYPE_LABELS[sourceKind] || "Document";
}

function typeClass(typeLabel: string): string {
  const t = typeLabel.toLowerCase();
  if (t.includes("litigation")) return "type-chip type-litigation";
  if (t.includes("regulatory notice")) return "type-chip type-regulatory";
  if (t.includes("speech") || t.includes("statement")) return "type-chip type-speech";
  if (t.includes("testimony")) return "type-chip type-testimony";
  if (t.includes("news")) return "type-chip type-news";
  if (t.includes("press release")) return "type-chip type-press";
  if (t.includes("faq")) return "type-chip type-faq";
  if (t.includes("key topic")) return "type-chip type-key-topic";
  return "type-chip type-default";
}

function headerFor(mode: HubMode): { title: string; subtitle: string } {
  if (mode === "home") {
    return {
      title: "Regulatory Intelligence",
      subtitle: "Track policy and enforcement signals."
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
  const [sort, setSort] = useState("date_desc");

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
  const [extract, setExtract] = useState<ExtractFormState>({
    connector: "sec_speech",
    selection: "new_or_updated",
    limit: 20,
    max_pages: 5,
    base_url: "",
    include_pdfs: true,
    include_rss: true
  });
  const [runAction, setRunAction] = useState<"extract" | "ingest" | "enrich" | null>(null);
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
  const sourceOptions = useMemo(
    () => [...facets.sources].sort((a, b) => displaySourceKind(a).localeCompare(displaySourceKind(b))),
    [facets.sources]
  );
  const topicOptions = facets.key_topics.length > 0 ? facets.key_topics : facets.topics.slice(0, 10);
  const latestPublished = useMemo(() => {
    let latestDateMs = 0;

    for (const item of items) {
      const ms = Date.parse(String(item.published_at || item.date || ""));
      if (Number.isFinite(ms) && ms > latestDateMs) {
        latestDateMs = ms;
      }
    }

    return latestDateMs > 0 ? fmtDateOnly(new Date(latestDateMs).toISOString()) : "-";
  }, [items]);

  const totalPages = Math.max(1, Math.ceil(total / Math.max(1, pageSize)));
  const extractBaseUrlPlaceholder =
    extract.connector === "finra_comment_letter"
      ? `Required: FINRA notice URL (e.g., ${FINRA_COMMENT_NOTICE_EXAMPLE})`
      : "Optional index URL override";
  const jobExecutionMode = metrics?.runtime?.job_execution_mode || "github_actions";
  const githubActionsEnabled = metrics?.runtime?.github_actions_enabled ?? true;
  const githubActionsEnabledFlag = metrics?.runtime?.github_actions_enabled_flag ?? true;
  const githubActionsMissingEnv = metrics?.runtime?.github_actions_missing_required_env || [];

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
  }, [keyword, needsDocs, org, page, pageSize, q, sort, source, topic]);

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
    async (kind: "extract" | "ingest" | "enrich") => {
      setRunAction(kind);
      setJobError("");
      try {
        const requestBody =
          kind === "ingest"
            ? ingest
            : kind === "enrich"
              ? enrich
              : extract;
        const payload = await fetchJson<JobStartPayload>(`/api/jobs/${kind}`, {
          method: "POST",
          body: JSON.stringify(requestBody)
        });
        setActiveJob({
          job_id: payload.job_id,
          status: payload.status,
          workflow: payload.workflow,
          updated_at: payload.updated_at,
          conclusion: payload.conclusion
        });
      } catch (err) {
        setJobError(err instanceof Error ? err.message : `Failed to run ${kind}.`);
      } finally {
        setRunAction(null);
      }
    },
    [enrich, extract, ingest]
  );

  const ask = useCallback(async () => {
    if (!needsChats || !prompt.trim()) return;
    const userPrompt = prompt.trim();
    const history = messages
      .slice(-6)
      .map((message) => ({ role: message.role, content: message.content }))
      .filter((message) => message.content.trim().length > 0);
    setPrompt("");
    setChatError("");
    setChatLoading(true);
    setMessages((prev) => [...prev, { role: "user", content: userPrompt }]);
    try {
      const data = await fetchJson<ChatAnswerData>("/api/chats/ask", {
        method: "POST",
        body: JSON.stringify({ prompt: userPrompt, top_k: 8, history })
      });
      setMessages((prev) => [...prev, { role: "assistant", content: data.answer, citations: data.citations, model: data.model }]);
    } catch (err) {
      setChatError(err instanceof Error ? err.message : "Failed to run chat.");
    } finally {
      setChatLoading(false);
    }
  }, [messages, needsChats, prompt]);

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
      <header className="panel hero-panel reveal p-6 md:p-8">
        <span className="kicker">Policy Research Hub</span>
        <h1 className="mt-3 text-3xl font-bold leading-tight md:text-5xl">{header.title}</h1>
        <p className="mt-3 max-w-3xl text-base text-[color:var(--ink-soft)] md:text-lg">{header.subtitle}</p>
      </header>

      {mode === "home" ? (
        <section className="panel reveal reveal-delay-1 p-5 md:p-6">
          <div className="max-w-xs">
            <article className="insight-card">
              <p className="label">Latest Published</p>
              <p className="value">{latestPublished}</p>
            </article>
          </div>
          <div className="my-4 soft-divider" />
          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-6">
            <input
              className="form-control px-3 py-2 text-sm xl:col-span-2"
              placeholder="Search text"
              value={q}
              onChange={(e) => {
                setPage(1);
                setQ(e.target.value);
              }}
            />
            <select
              className="form-control px-3 py-2 text-sm"
              value={source}
              onChange={(e) => {
                setPage(1);
                setSource(e.target.value);
              }}
            >
              <option value="">All sources</option>
              {sourceOptions.map((v) => (
                <option key={v} value={v}>
                  {displaySourceKind(v)}
                </option>
              ))}
            </select>
            <select
              className="form-control px-3 py-2 text-sm"
              value={org}
              onChange={(e) => {
                setPage(1);
                setOrg(e.target.value);
              }}
            >
              <option value="">All orgs</option>
              {facets.organizations.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
            <select
              className="form-control px-3 py-2 text-sm"
              value={topic}
              onChange={(e) => {
                setPage(1);
                setTopic(e.target.value);
              }}
            >
              <option value="">Key topics (Top 10)</option>
              {topicOptions.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
            <input
              className="form-control px-3 py-2 text-sm"
              placeholder="Keyword Text"
              value={keyword}
              onChange={(e) => {
                setPage(1);
                setKeyword(e.target.value);
              }}
            />
          </div>
          <div className="mt-3 flex items-center justify-between">
            <p className="text-sm text-[color:var(--ink-soft)]">{fmt(total)} matching documents</p>
            <select
              className="form-control px-3 py-1.5 text-sm"
              value={sort}
              onChange={(e) => {
                setPage(1);
                setSort(e.target.value);
              }}
            >
              <option value="date_desc">Newest Published</option>
              <option value="updated_desc">Recently Updated</option>
              <option value="date_asc">Oldest Published</option>
            </select>
          </div>
          {docsError ? (
            <p className="callout callout-error mt-3">{docsError}</p>
          ) : null}
          <div className="feed-table-wrap mt-3">
            <table className="feed-table">
              <thead>
                <tr>
                  <th>Title</th>
                  <th>Organization</th>
                  <th>Type</th>
                  <th>Name</th>
                  <th>Keyword Text</th>
                  <th>Published</th>
                </tr>
              </thead>
              <tbody>
                {docsLoading ? (
                  <tr>
                    <td colSpan={6} className="text-sm">
                      Loading feed...
                    </td>
                  </tr>
                ) : items.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="text-sm">
                      No documents match these filters.
                    </td>
                  </tr>
                ) : (
                  items.map((d) => {
                    const typeLabel = displayType(d.doc_type, d.source_kind);
                    return (
                      <tr key={d.document_id}>
                        <td>
                          <p className="feed-title">{d.title || "Untitled"}</p>
                          {d.url ? (
                            <a
                              href={d.url}
                              target="_blank"
                              rel="noreferrer"
                              className="link-inline mt-1 inline-block text-xs"
                            >
                              Open source
                            </a>
                          ) : null}
                        </td>
                        <td className="text-xs">
                          <span className="tone-chip">{displayOrganization(d.organization)}</span>
                          <p className="feed-subtle mt-2">{fmt(d.word_count)} words</p>
                        </td>
                        <td className="text-xs">
                          <span className={typeClass(typeLabel)}>{typeLabel}</span>
                        </td>
                        <td className="text-xs">{exactSpeakerName(d)}</td>
                        <td className="text-xs">
                          {(d.keywords || []).slice(0, 4).join(", ") || (d.topics || []).slice(0, 4).join(", ") || "-"}
                        </td>
                        <td className="text-xs">{fmtDateOnly(d.published_at || d.date)}</td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
          <div className="mt-3 flex items-center justify-between text-xs">
            <p className="text-[color:var(--ink-faint)]">Page {page} of {totalPages}</p>
            <div className="flex gap-2">
              <button
                className="form-control px-3 py-1.5 font-semibold disabled:opacity-50"
                disabled={page <= 1 || docsLoading}
                onClick={() => setPage(Math.max(1, page - 1))}
              >
                Previous
              </button>
              <button
                className="form-control px-3 py-1.5 font-semibold disabled:opacity-50"
                disabled={page >= totalPages || docsLoading}
                onClick={() => setPage(Math.min(totalPages, page + 1))}
              >
                Next
              </button>
            </div>
          </div>
        </section>
      ) : null}

      {mode === "operations" ? (
        <section className="grid gap-4">
          {!metricsLoading && jobExecutionMode === "local" ? (
            <p className="callout callout-info">
              Extraction execution mode: local (direct Python pipeline with remote persistence required).
            </p>
          ) : null}
          {!metricsLoading && jobExecutionMode === "github_actions" && !githubActionsEnabled ? (
            <p className="callout callout-error">
              {githubActionsEnabledFlag
                ? `GitHub Actions dispatch is missing required Vercel env vars: ${githubActionsMissingEnv.join(", ")}`
                : "GitHub Actions dispatch is disabled by GITHUB_ACTIONS_ENABLED=false"}
            </p>
          ) : null}
          <article className="panel p-5">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">News Connector Settings</h2>
              <div className="flex gap-2">
                <button className="btn-muted px-3 py-1.5 text-xs" onClick={() => void loadSettings()}>
                  Reload
                </button>
                <button
                  className="btn-solid px-3 py-1.5 text-xs disabled:opacity-50"
                  disabled={settingsSaving}
                  onClick={() => void saveSettings()}
                >
                  {settingsSaving ? "Saving..." : "Save Settings"}
                </button>
              </div>
            </div>
            {settingsLoading ? <p className="mt-2 text-sm">Loading settings...</p> : null}
            {settingsErr ? <p className="callout callout-error mt-2">{settingsErr}</p> : null}
            {settingsMsg ? <p className="callout callout-success mt-2">{settingsMsg}</p> : null}
            <div className="mt-3 grid gap-2 md:grid-cols-2">
              <input
                className="form-control md:col-span-2 px-3 py-2 text-sm"
                value={settings.query}
                onChange={(e) => setSettings({ ...settings, query: e.target.value })}
                placeholder="Query"
              />
              <input
                className="form-control px-3 py-2 text-sm"
                value={settings.organization_label}
                onChange={(e) => setSettings({ ...settings, organization_label: e.target.value })}
                placeholder="Organization label"
              />
              <select
                className="form-control px-3 py-2 text-sm"
                value={settings.sort_by}
                onChange={(e) => setSettings({ ...settings, sort_by: e.target.value })}
              >
                <option value="publishedAt">publishedAt</option>
                <option value="relevancy">relevancy</option>
                <option value="popularity">popularity</option>
              </select>
              <input
                type="number"
                min={1}
                max={30}
                className="form-control px-3 py-2 text-sm"
                value={settings.lookback_days}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    lookback_days: Math.max(1, Math.min(30, Number.parseInt(e.target.value || "7", 10) || 7))
                  })
                }
                placeholder="Lookback days"
              />
              <input
                type="number"
                min={1}
                max={10}
                className="form-control px-3 py-2 text-sm"
                value={settings.max_pages}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    max_pages: Math.max(1, Math.min(10, Number.parseInt(e.target.value || "4", 10) || 4))
                  })
                }
                placeholder="Max pages"
              />
              <input
                type="number"
                min={10}
                max={100}
                className="form-control px-3 py-2 text-sm"
                value={settings.page_size}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    page_size: Math.max(10, Math.min(100, Number.parseInt(e.target.value || "50", 10) || 50))
                  })
                }
                placeholder="Page size"
              />
              <input
                type="number"
                min={10}
                max={500}
                className="form-control px-3 py-2 text-sm"
                value={settings.target_count}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    target_count: Math.max(10, Math.min(500, Number.parseInt(e.target.value || "100", 10) || 100))
                  })
                }
                placeholder="Target count"
              />
              <input
                className="form-control md:col-span-2 px-3 py-2 text-sm"
                value={settings.domains}
                onChange={(e) => setSettings({ ...settings, domains: e.target.value })}
                placeholder="Domains CSV"
              />
              <input
                className="form-control md:col-span-2 px-3 py-2 text-sm"
                value={settings.exclude_domains}
                onChange={(e) => setSettings({ ...settings, exclude_domains: e.target.value })}
                placeholder="Exclude domains CSV"
              />
              <input
                className="form-control md:col-span-2 px-3 py-2 text-sm"
                value={settings.tags_csv}
                onChange={(e) => setSettings({ ...settings, tags_csv: e.target.value })}
                placeholder="Tags CSV"
              />
            </div>
          </article>
          <section className="grid gap-4 md:grid-cols-3">
            <article className="panel p-5">
              <h2 className="text-xl font-semibold">Run Extraction</h2>
              <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                <select
                  className="form-control col-span-2 px-2 py-1.5"
                  value={extract.connector}
                  onChange={(e) =>
                    setExtract((prev) => {
                      const connector = e.target.value as ExtractFormState["connector"];
                      const nextBaseUrl =
                        connector === "finra_comment_letter" && !prev.base_url.trim()
                          ? FINRA_COMMENT_NOTICE_EXAMPLE
                          : prev.base_url;
                      return { ...prev, connector, base_url: nextBaseUrl };
                    })
                  }
                >
                  <option value="sec_speech">SEC Speeches &amp; Statements</option>
                  <option value="sec_enforcement_litigation">SEC Litigation Releases</option>
                  <option value="sec_tm_faq">SEC Trading & Markets FAQ</option>
                  <option value="finra_regulatory_notice">FINRA Regulatory Notices</option>
                  <option value="finra_comment_letter">FINRA Comment Letters (Rule URL)</option>
                  <option value="finra_key_topic">FINRA Key Topics</option>
                  <option value="doj_usao_press_release">DOJ USAO Press Releases</option>
                  <option value="federal_reserve_speech_testimony">Federal Reserve Speeches/Testimony</option>
                </select>
                <select
                  className="form-control px-2 py-1.5"
                  value={extract.selection}
                  onChange={(e) => setExtract({ ...extract, selection: e.target.value === "all" ? "all" : "new_or_updated" })}
                >
                  <option value="new_or_updated">new_or_updated</option>
                  <option value="all">all</option>
                </select>
                <input
                  type="number"
                  min={1}
                  className="form-control px-2 py-1.5"
                  value={extract.limit}
                  onChange={(e) => setExtract({ ...extract, limit: Math.max(1, Number.parseInt(e.target.value || "20", 10) || 20) })}
                />
                <input
                  type="number"
                  min={1}
                  className="form-control px-2 py-1.5"
                  value={extract.max_pages}
                  onChange={(e) => setExtract({ ...extract, max_pages: Math.max(1, Number.parseInt(e.target.value || "5", 10) || 5) })}
                  placeholder="Max pages"
                />
                <input
                  className="form-control col-span-2 px-2 py-1.5"
                  value={extract.base_url}
                  onChange={(e) => setExtract({ ...extract, base_url: e.target.value })}
                  placeholder={extractBaseUrlPlaceholder}
                />
                {extract.connector === "finra_comment_letter" ? (
                  <p className="col-span-2 text-xs text-[color:var(--ink-soft)]">
                    Use the FINRA notice URL. The scraper will discover linked comment letters from the comments section.
                  </p>
                ) : null}
                <label className="col-span-2 flex items-center gap-2 text-xs text-[color:var(--ink-soft)]">
                  <input
                    type="checkbox"
                    checked={extract.include_pdfs}
                    onChange={(e) => setExtract({ ...extract, include_pdfs: e.target.checked })}
                  />
                  Include PDFs (SEC TM FAQ, FINRA comment letters)
                </label>
                <label className="col-span-2 flex items-center gap-2 text-xs text-[color:var(--ink-soft)]">
                  <input
                    type="checkbox"
                    checked={extract.include_rss}
                    onChange={(e) => setExtract({ ...extract, include_rss: e.target.checked })}
                  />
                  Use RSS supplement (FINRA notices)
                </label>
              </div>
              <button
                className="btn-solid mt-3 w-full px-3 py-2 text-sm disabled:opacity-50"
                disabled={runAction !== null}
                onClick={() => void launch("extract")}
              >
                {runAction === "extract" ? "Launching..." : "Run Extraction"}
              </button>
            </article>
            <article className="panel p-5">
              <h2 className="text-xl font-semibold">Run Ingest</h2>
              <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                <input
                  type="number"
                  min={1}
                  className="form-control px-2 py-1.5"
                  value={ingest.limit}
                  onChange={(e) => setIngest({ ...ingest, limit: Math.max(1, Number.parseInt(e.target.value || "10", 10) || 10) })}
                />
                <input
                  type="number"
                  min={1}
                  className="form-control px-2 py-1.5"
                  value={ingest.lookback_days}
                  onChange={(e) =>
                    setIngest({
                      ...ingest,
                      lookback_days: Math.max(1, Number.parseInt(e.target.value || "7", 10) || 7)
                    })
                  }
                />
                <select
                  className="form-control col-span-2 px-2 py-1.5"
                  value={ingest.selection}
                  onChange={(e) => setIngest({ ...ingest, selection: e.target.value === "all" ? "all" : "new_or_updated" })}
                >
                  <option value="new_or_updated">new_or_updated</option>
                  <option value="all">all</option>
                </select>
              </div>
              <button
                className="btn-solid mt-3 w-full px-3 py-2 text-sm disabled:opacity-50"
                disabled={runAction !== null}
                onClick={() => void launch("ingest")}
              >
                {runAction === "ingest" ? "Launching..." : "Run Ingest"}
              </button>
            </article>
            <article className="panel p-5">
              <h2 className="text-xl font-semibold">Run Enrichment</h2>
              <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                <input
                  type="number"
                  min={1}
                  className="form-control px-2 py-1.5"
                  value={enrich.limit}
                  onChange={(e) => setEnrich({ ...enrich, limit: Math.max(1, Number.parseInt(e.target.value || "25", 10) || 25) })}
                />
                <input
                  className="form-control px-2 py-1.5"
                  value={enrich.source_kind}
                  onChange={(e) => setEnrich({ ...enrich, source_kind: e.target.value || "newsapi_article" })}
                />
                <select
                  className="form-control px-2 py-1.5"
                  value={enrich.mode}
                  onChange={(e) => setEnrich({ ...enrich, mode: e.target.value === "all" ? "all" : "only_missing_or_failed" })}
                >
                  <option value="only_missing_or_failed">only_missing_or_failed</option>
                  <option value="all">all</option>
                </select>
                <input
                  className="form-control px-2 py-1.5"
                  placeholder="model (optional)"
                  value={enrich.model}
                  onChange={(e) => setEnrich({ ...enrich, model: e.target.value })}
                />
                <label className="col-span-2 flex items-center gap-2 text-xs text-[color:var(--ink-soft)]">
                  <input
                    type="checkbox"
                    checked={enrich.heuristic_only}
                    onChange={(e) => setEnrich({ ...enrich, heuristic_only: e.target.checked })}
                  />
                  Heuristic-only
                </label>
              </div>
              <button
                className="btn-accent mt-3 w-full px-3 py-2 text-sm disabled:opacity-50"
                disabled={runAction !== null}
                onClick={() => void launch("enrich")}
              >
                {runAction === "enrich" ? "Launching..." : "Run Enrichment"}
              </button>
            </article>
          </section>
          <article className="panel p-5">
            <h2 className="text-xl font-semibold">Active Job</h2>
            {!activeJob ? (
              <p className="mt-2 text-sm">No active run yet.</p>
            ) : (
              <div className="mt-2 space-y-1 text-sm">
                <span className={statusClass(activeJob.status)}>{activeJob.status}</span>
                <p>Job: {activeJob.job_id}</p>
                {activeJob.workflow ? <p>Workflow: {activeJob.workflow}</p> : null}
                {activeJob.updated_at ? <p>Updated: {fmtDate(activeJob.updated_at)}</p> : null}
                {activeJob.conclusion ? <p>Conclusion: {activeJob.conclusion}</p> : null}
                {activeJob.html_url ? (
                  <a href={activeJob.html_url} target="_blank" rel="noreferrer" className="link-inline">
                    Open GitHub run
                  </a>
                ) : null}
              </div>
            )}
            {jobError ? <p className="callout callout-error mt-2">{jobError}</p> : null}
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
          {metricsError ? <p className="callout callout-error">{metricsError}</p> : null}
          <section className="grid gap-4 xl:grid-cols-[1.55fr_1fr]">
            <article className="panel p-5">
              <h2 className="text-xl font-semibold">Source Mix</h2>
              <div className="mt-3 space-y-2">
                {sourceMix.length === 0 ? (
                  <p className="text-sm">No source telemetry yet.</p>
                ) : (
                  sourceMix.map((row) => (
                    <div key={row.source_kind}>
                      <div className="mb-1 flex justify-between text-xs">
                        <span>{row.source_kind}</span>
                        <span>{fmt(row.count)}</span>
                      </div>
                      <div className="h-2 rounded-full bg-[color:rgba(79,213,255,0.12)]">
                        <div
                          className="h-2 rounded-full bg-[linear-gradient(90deg,#4fd5ff,#f09b3d)]"
                          style={{ width: `${row.width}%` }}
                        />
                      </div>
                    </div>
                  ))
                )}
              </div>
            </article>
            <article className="panel p-5">
              <h2 className="text-xl font-semibold">Pipeline Snapshot</h2>
              <p className="mt-2 text-sm">
                Last run: <strong>{fmtDate(metrics?.recent_ingest.last_run_at || "")}</strong>
              </p>
              <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(10,23,37,0.9)] px-3 py-2">
                  <p className="text-xs">Processed</p>
                  <p className="text-lg font-semibold">{fmt(metrics?.recent_ingest.processed_count || 0)}</p>
                </div>
                <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(10,23,37,0.9)] px-3 py-2">
                  <p className="text-xs">Failed</p>
                  <p className="text-lg font-semibold">{fmt(metrics?.recent_ingest.failed_count || 0)}</p>
                </div>
              </div>
            </article>
          </section>
        </section>
      ) : null}

      {mode === "chats" ? (
        <section className="grid gap-4 xl:grid-cols-[1.35fr_1fr]">
          <article className="panel p-5">
            <div className="max-h-[480px] space-y-3 overflow-y-auto rounded-xl border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.9)] p-3">
              {messages.map((m, idx) => (
                <article
                  key={`${m.role}_${idx}`}
                  className={`rounded-xl border px-3 py-2 ${
                    m.role === "assistant"
                      ? "border-[color:rgba(79,213,255,0.28)] bg-[color:rgba(79,213,255,0.09)]"
                      : "border-[color:rgba(240,155,61,0.35)] bg-[color:rgba(240,155,61,0.1)]"
                  }`}
                >
                  <p className="text-xs font-semibold uppercase">{m.role}</p>
                  <p className="mt-1 whitespace-pre-wrap text-sm">{m.content}</p>
                  {m.model ? <p className="mt-1 text-[11px] uppercase tracking-[0.18em] text-slate-400">Model: {m.model}</p> : null}
                  {m.citations?.length ? (
                    <div className="mt-2 space-y-2">
                      {m.citations.map((c) => (
                        <div
                          key={c.document_id}
                          className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.96)] px-2 py-2 text-xs"
                        >
                          <p className="font-semibold">{c.title}</p>
                          <p className="mt-1">
                            {c.organization} - {c.source_kind}
                          </p>
                          {c.published_at ? <p className="mt-1">{fmtDateOnly(c.published_at)}</p> : null}
                          {c.snippet ? <p className="mt-1">{c.snippet}</p> : null}
                          {c.url ? (
                            <a href={c.url} target="_blank" rel="noreferrer" className="link-inline mt-1 inline-block">
                              Open source
                            </a>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  ) : null}
                </article>
              ))}
            </div>
            {chatError ? <p className="callout callout-error mt-3">{chatError}</p> : null}
            <div className="mt-3 flex gap-2">
              <input
                className="form-control w-full px-3 py-2 text-sm"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void ask();
                  }
                }}
                placeholder="Ask a policy question..."
              />
              <button
                className="btn-solid px-4 py-2 text-sm disabled:opacity-50"
                onClick={() => void ask()}
                disabled={chatLoading}
              >
                {chatLoading ? "Thinking..." : "Ask"}
              </button>
            </div>
          </article>
          <article className="panel p-5">
            <h2 className="text-xl font-semibold">Prompt Ideas</h2>
            <div className="mt-3 space-y-2 text-sm">
              <p className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(10,23,37,0.92)] px-3 py-2">
                Which themes are rising in SEC speeches this month?
              </p>
              <p className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(10,23,37,0.92)] px-3 py-2">
                Summarize enforcement trends across SEC and DOJ sources.
              </p>
              <p className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(10,23,37,0.92)] px-3 py-2">
                Find documents mentioning capital requirements and liquidity risks.
              </p>
            </div>
          </article>
        </section>
      ) : null}
    </div>
  );
}
