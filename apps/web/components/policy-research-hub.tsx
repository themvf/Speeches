"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

type JobStatus = "queued" | "running" | "success" | "failed" | "unknown";

interface ApiEnvelope<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

interface SourceCount {
  source_kind: string;
  count: number;
}

interface MetricsData {
  totals: {
    documents: number;
    organizations: number;
    enriched: number;
    pending_review: number;
  };
  recent_ingest: {
    last_run_at: string;
    processed_count: number;
    failed_count: number;
  };
  by_source_kind: SourceCount[];
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
  enrichment_status: string;
  review_decision: string;
  updated_at: string;
}

interface DocumentsData {
  items: DocumentItem[];
  page: number;
  page_size: number;
  total: number;
}

interface JobState {
  job_id: string;
  provider: string;
  status: JobStatus;
  status_url?: string;
  github_run_id?: number;
  workflow?: string;
  html_url?: string;
  created_at?: string;
  started_at?: string;
  updated_at?: string;
  finished_at?: string;
  conclusion?: string;
  artifacts?: string[];
}

interface JobStartPayload {
  job_id: string;
  provider: "github_actions";
  status: "queued";
  status_url: string;
  github_run_id: number;
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

function metricFormatter(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatDate(value: string): string {
  if (!value) {
    return "-";
  }
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) {
    return value;
  }
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit"
  });
}

function statusPillClass(status: string): string {
  const normalized = status.toLowerCase();
  if (normalized === "enriched" || normalized === "reviewed" || normalized === "success") {
    return "border-emerald-300 bg-emerald-50 text-emerald-800";
  }
  if (normalized === "fallback_enriched" || normalized === "queued" || normalized === "running") {
    return "border-amber-300 bg-amber-50 text-amber-800";
  }
  if (normalized === "failed" || normalized === "rejected") {
    return "border-rose-300 bg-rose-50 text-rose-800";
  }
  return "border-slate-300 bg-slate-50 text-slate-700";
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    cache: "no-store",
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers as Record<string, string> | undefined)
    }
  });

  let parsed: ApiEnvelope<T> | null = null;
  try {
    parsed = (await response.json()) as ApiEnvelope<T>;
  } catch {
    parsed = null;
  }

  if (!response.ok || !parsed || parsed.ok === false || !parsed.data) {
    const message = parsed?.error || `Request failed with status ${response.status}`;
    throw new Error(message);
  }

  return parsed.data;
}

export function PolicyResearchHub() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(true);
  const [metricsError, setMetricsError] = useState("");

  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [documentsTotal, setDocumentsTotal] = useState(0);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [documentsError, setDocumentsError] = useState("");

  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("");
  const [sort, setSort] = useState("updated_desc");

  const [ingestForm, setIngestForm] = useState<IngestFormState>({
    limit: 10,
    lookback_days: 7,
    selection: "new_or_updated"
  });
  const [enrichForm, setEnrichForm] = useState<EnrichFormState>({
    limit: 25,
    mode: "only_missing_or_failed",
    source_kind: "newsapi_article",
    heuristic_only: false,
    model: ""
  });

  const [activeJob, setActiveJob] = useState<JobState | null>(null);
  const [jobError, setJobError] = useState("");
  const [runAction, setRunAction] = useState<"ingest" | "enrich" | null>(null);

  const fetchMetrics = useCallback(async () => {
    setMetricsLoading(true);
    setMetricsError("");
    try {
      const payload = await fetchJson<MetricsData>("/api/metrics");
      setMetrics(payload);
    } catch (error) {
      setMetricsError(error instanceof Error ? error.message : "Failed to load metrics.");
    } finally {
      setMetricsLoading(false);
    }
  }, []);

  const fetchDocuments = useCallback(async () => {
    setDocumentsLoading(true);
    setDocumentsError("");
    try {
      const params = new URLSearchParams();
      params.set("page", "1");
      params.set("page_size", "14");
      params.set("sort", sort);
      if (query.trim()) {
        params.set("q", query.trim());
      }
      if (status.trim()) {
        params.set("status", status.trim());
      }

      const payload = await fetchJson<DocumentsData>(`/api/documents?${params.toString()}`);
      setDocuments(payload.items || []);
      setDocumentsTotal(payload.total || 0);
    } catch (error) {
      setDocumentsError(error instanceof Error ? error.message : "Failed to load documents.");
    } finally {
      setDocumentsLoading(false);
    }
  }, [query, status, sort]);

  useEffect(() => {
    void fetchMetrics();
  }, [fetchMetrics]);

  useEffect(() => {
    const timer = setTimeout(() => {
      void fetchDocuments();
    }, 220);
    return () => clearTimeout(timer);
  }, [fetchDocuments]);

  useEffect(() => {
    if (!activeJob?.job_id || !["queued", "running", "unknown"].includes(activeJob.status)) {
      return;
    }

    let cancelled = false;

    const poll = async () => {
      try {
        const payload = await fetchJson<JobState>(`/api/jobs/${activeJob.job_id}`);
        if (!cancelled) {
          setActiveJob(payload);
          if (["success", "failed"].includes(payload.status)) {
            void fetchMetrics();
            void fetchDocuments();
          }
        }
      } catch (error) {
        if (!cancelled) {
          setJobError(error instanceof Error ? error.message : "Failed to refresh job status.");
        }
      }
    };

    void poll();
    const interval = setInterval(() => {
      void poll();
    }, 4000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [activeJob?.job_id, activeJob?.status, fetchDocuments, fetchMetrics]);

  const sourceMix = useMemo(() => {
    const rows = metrics?.by_source_kind || [];
    const max = rows.reduce((best, item) => (item.count > best ? item.count : best), 1);
    return rows.slice(0, 8).map((item) => ({
      ...item,
      width: Math.max(8, Math.round((item.count / max) * 100))
    }));
  }, [metrics?.by_source_kind]);

  const launchIngest = useCallback(async () => {
    setRunAction("ingest");
    setJobError("");
    try {
      const payload = await fetchJson<JobStartPayload>("/api/jobs/ingest", {
        method: "POST",
        body: JSON.stringify(ingestForm)
      });
      setActiveJob(payload);
    } catch (error) {
      setJobError(error instanceof Error ? error.message : "Failed to launch ingest job.");
    } finally {
      setRunAction(null);
    }
  }, [ingestForm]);

  const launchEnrich = useCallback(async () => {
    setRunAction("enrich");
    setJobError("");
    try {
      const payload = await fetchJson<JobStartPayload>("/api/jobs/enrich", {
        method: "POST",
        body: JSON.stringify(enrichForm)
      });
      setActiveJob(payload);
    } catch (error) {
      setJobError(error instanceof Error ? error.message : "Failed to launch enrichment job.");
    } finally {
      setRunAction(null);
    }
  }, [enrichForm]);

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-6 md:px-8 md:py-10">
      <header className="panel reveal p-6 md:p-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <span className="kicker">Policy Research Hub</span>
            <h1 className="mt-3 text-3xl font-bold leading-tight md:text-5xl">Regulatory Intelligence, Built For Decisions</h1>
            <p className="mt-3 max-w-3xl text-base text-[color:rgba(16,36,59,0.76)] md:text-lg">
              Track policy and enforcement signals, triage incoming sources, and run ingestion and enrichment from one
              operational surface.
            </p>
          </div>
          <nav className="flex flex-wrap gap-2 text-sm font-semibold">
            <a href="#signals" className="rounded-full border border-[color:var(--line)] bg-white/80 px-3 py-1.5">
              Signals
            </a>
            <a href="#feed" className="rounded-full border border-[color:var(--line)] bg-white/80 px-3 py-1.5">
              Research Feed
            </a>
            <a href="#operations" className="rounded-full border border-[color:var(--line)] bg-white/80 px-3 py-1.5">
              Operations
            </a>
          </nav>
        </div>

        <section id="signals" className="mt-6 grid gap-3 md:grid-cols-4">
          <article className="panel p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:rgba(45,86,115,0.8)]">Documents</p>
            <p className="mt-1 text-2xl font-semibold">{metricsLoading ? "..." : metricFormatter(metrics?.totals.documents || 0)}</p>
          </article>
          <article className="panel p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:rgba(45,86,115,0.8)]">Organizations</p>
            <p className="mt-1 text-2xl font-semibold">{metricsLoading ? "..." : metricFormatter(metrics?.totals.organizations || 0)}</p>
          </article>
          <article className="panel p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:rgba(45,86,115,0.8)]">Enriched</p>
            <p className="mt-1 text-2xl font-semibold">{metricsLoading ? "..." : metricFormatter(metrics?.totals.enriched || 0)}</p>
          </article>
          <article className="panel p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:rgba(45,86,115,0.8)]">Pending Review</p>
            <p className="mt-1 text-2xl font-semibold">{metricsLoading ? "..." : metricFormatter(metrics?.totals.pending_review || 0)}</p>
          </article>
        </section>

        {metricsError ? <p className="mt-4 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{metricsError}</p> : null}

        <section className="mt-5 grid gap-3 md:grid-cols-[1.35fr_1fr]">
          <article className="rounded-2xl border border-[color:var(--line)] bg-white/75 p-4">
            <p className="text-sm font-semibold">Source Mix</p>
            <div className="mt-3 space-y-2">
              {sourceMix.length === 0 ? (
                <p className="text-sm text-[color:rgba(16,36,59,0.66)]">No source telemetry yet.</p>
              ) : (
                sourceMix.map((row) => (
                  <div key={row.source_kind}>
                    <div className="mb-1 flex items-center justify-between text-xs">
                      <span>{row.source_kind}</span>
                      <span>{metricFormatter(row.count)}</span>
                    </div>
                    <div className="h-2 rounded-full bg-[color:rgba(16,36,59,0.1)]">
                      <div
                        className="h-2 rounded-full bg-[linear-gradient(90deg,#2d5673,#c77d28)]"
                        style={{ width: `${row.width}%` }}
                      />
                    </div>
                  </div>
                ))
              )}
            </div>
          </article>

          <article className="rounded-2xl border border-[color:var(--line)] bg-white/75 p-4">
            <p className="text-sm font-semibold">Latest Pipeline Snapshot</p>
            <p className="mt-2 text-sm text-[color:rgba(16,36,59,0.75)]">
              Last run: <strong>{formatDate(metrics?.recent_ingest.last_run_at || "")}</strong>
            </p>
            <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
              <div className="rounded-xl border border-[color:var(--line)] bg-white px-3 py-2">
                <p className="text-xs text-[color:rgba(16,36,59,0.65)]">Processed</p>
                <p className="text-lg font-semibold">{metricFormatter(metrics?.recent_ingest.processed_count || 0)}</p>
              </div>
              <div className="rounded-xl border border-[color:var(--line)] bg-white px-3 py-2">
                <p className="text-xs text-[color:rgba(16,36,59,0.65)]">Failed</p>
                <p className="text-lg font-semibold">{metricFormatter(metrics?.recent_ingest.failed_count || 0)}</p>
              </div>
            </div>
          </article>
        </section>
      </header>

      <section className="grid gap-5 lg:grid-cols-[1.6fr_1fr]">
        <article id="feed" className="panel reveal reveal-delay-1 p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-2xl font-semibold">Research Feed</h2>
            <p className="text-sm text-[color:rgba(16,36,59,0.7)]">{metricFormatter(documentsTotal)} matching documents</p>
          </div>

          <div className="mt-4 grid gap-2 sm:grid-cols-3">
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search title, source, text"
              className="rounded-xl border border-[color:var(--line)] bg-white px-3 py-2 text-sm outline-none focus:border-[color:#2d5673]"
            />
            <select
              value={status}
              onChange={(event) => setStatus(event.target.value)}
              className="rounded-xl border border-[color:var(--line)] bg-white px-3 py-2 text-sm outline-none focus:border-[color:#2d5673]"
            >
              <option value="">All statuses</option>
              <option value="not_enriched">Not Enriched</option>
              <option value="enriched">Enriched</option>
              <option value="fallback_enriched">Fallback Enriched</option>
              <option value="reviewed">Reviewed</option>
            </select>
            <select
              value={sort}
              onChange={(event) => setSort(event.target.value)}
              className="rounded-xl border border-[color:var(--line)] bg-white px-3 py-2 text-sm outline-none focus:border-[color:#2d5673]"
            >
              <option value="updated_desc">Recently Updated</option>
              <option value="date_desc">Newest Published</option>
              <option value="date_asc">Oldest Published</option>
            </select>
          </div>

          {documentsError ? (
            <p className="mt-4 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{documentsError}</p>
          ) : null}

          <div className="mt-4 overflow-x-auto rounded-xl border border-[color:var(--line)] bg-white">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-[color:rgba(16,36,59,0.05)] text-xs uppercase tracking-[0.08em] text-[color:rgba(16,36,59,0.72)]">
                <tr>
                  <th className="px-3 py-2">Title</th>
                  <th className="px-3 py-2">Source</th>
                  <th className="px-3 py-2">Status</th>
                  <th className="px-3 py-2">Published</th>
                </tr>
              </thead>
              <tbody>
                {documentsLoading ? (
                  <tr>
                    <td colSpan={4} className="px-3 py-4 text-sm text-[color:rgba(16,36,59,0.65)]">
                      Loading feed...
                    </td>
                  </tr>
                ) : documents.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="px-3 py-4 text-sm text-[color:rgba(16,36,59,0.65)]">
                      No documents match these filters.
                    </td>
                  </tr>
                ) : (
                  documents.map((doc) => (
                    <tr key={doc.document_id} className="border-t border-[color:var(--line)] align-top">
                      <td className="px-3 py-3">
                        <p className="font-semibold text-[color:#10243b]">{doc.title || "Untitled"}</p>
                        <p className="mt-1 text-xs text-[color:rgba(16,36,59,0.66)]">{doc.organization} - {doc.doc_type || "Document"}</p>
                        {doc.url ? (
                          <a href={doc.url} target="_blank" rel="noreferrer" className="mt-1 inline-block text-xs text-[color:#2d5673] underline-offset-2 hover:underline">
                            Open source
                          </a>
                        ) : null}
                      </td>
                      <td className="px-3 py-3 text-xs">
                        <span className="rounded-full border border-[color:var(--line)] bg-[color:rgba(16,36,59,0.04)] px-2 py-1">
                          {doc.source_kind || "unknown"}
                        </span>
                        <p className="mt-2 text-[color:rgba(16,36,59,0.65)]">{metricFormatter(doc.word_count || 0)} words</p>
                      </td>
                      <td className="px-3 py-3 text-xs">
                        <span className={`rounded-full border px-2 py-1 ${statusPillClass(doc.enrichment_status)}`}>
                          {doc.enrichment_status || "not_enriched"}
                        </span>
                        <p className="mt-2 text-[color:rgba(16,36,59,0.65)]">Review: {doc.review_decision || "pending"}</p>
                      </td>
                      <td className="px-3 py-3 text-xs text-[color:rgba(16,36,59,0.7)]">{formatDate(doc.published_at || doc.date)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </article>

        <article id="operations" className="panel reveal reveal-delay-2 p-5">
          <h2 className="text-2xl font-semibold">Operations Console</h2>
          <p className="mt-2 text-sm text-[color:rgba(16,36,59,0.72)]">Launch collection or enrichment jobs and monitor execution in real time.</p>

          <div className="mt-4 space-y-4">
            <section className="rounded-xl border border-[color:var(--line)] bg-white p-3">
              <h3 className="text-sm font-semibold">Launch Ingest</h3>
              <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                <label className="flex flex-col gap-1">
                  <span className="text-xs">Limit</span>
                  <input
                    type="number"
                    min={1}
                    value={ingestForm.limit}
                    onChange={(event) => setIngestForm((prev) => ({ ...prev, limit: Math.max(1, Number.parseInt(event.target.value, 10) || 1) }))}
                    className="rounded-lg border border-[color:var(--line)] px-2 py-1.5"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs">Lookback Days</span>
                  <input
                    type="number"
                    min={1}
                    value={ingestForm.lookback_days}
                    onChange={(event) =>
                      setIngestForm((prev) => ({ ...prev, lookback_days: Math.max(1, Number.parseInt(event.target.value, 10) || 1) }))
                    }
                    className="rounded-lg border border-[color:var(--line)] px-2 py-1.5"
                  />
                </label>
                <label className="col-span-2 flex flex-col gap-1">
                  <span className="text-xs">Selection</span>
                  <select
                    value={ingestForm.selection}
                    onChange={(event) =>
                      setIngestForm((prev) => ({
                        ...prev,
                        selection: event.target.value === "all" ? "all" : "new_or_updated"
                      }))
                    }
                    className="rounded-lg border border-[color:var(--line)] px-2 py-1.5"
                  >
                    <option value="new_or_updated">new_or_updated</option>
                    <option value="all">all</option>
                  </select>
                </label>
              </div>
              <button
                onClick={() => void launchIngest()}
                disabled={runAction !== null}
                className="mt-3 w-full rounded-xl bg-[color:#2d5673] px-3 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-55"
              >
                {runAction === "ingest" ? "Launching..." : "Run Ingest"}
              </button>
            </section>

            <section className="rounded-xl border border-[color:var(--line)] bg-white p-3">
              <h3 className="text-sm font-semibold">Launch Enrichment</h3>
              <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                <label className="flex flex-col gap-1">
                  <span className="text-xs">Limit</span>
                  <input
                    type="number"
                    min={1}
                    value={enrichForm.limit}
                    onChange={(event) => setEnrichForm((prev) => ({ ...prev, limit: Math.max(1, Number.parseInt(event.target.value, 10) || 1) }))}
                    className="rounded-lg border border-[color:var(--line)] px-2 py-1.5"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs">Source Kind</span>
                  <input
                    value={enrichForm.source_kind}
                    onChange={(event) => setEnrichForm((prev) => ({ ...prev, source_kind: event.target.value || "newsapi_article" }))}
                    className="rounded-lg border border-[color:var(--line)] px-2 py-1.5"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs">Mode</span>
                  <select
                    value={enrichForm.mode}
                    onChange={(event) =>
                      setEnrichForm((prev) => ({
                        ...prev,
                        mode: event.target.value === "all" ? "all" : "only_missing_or_failed"
                      }))
                    }
                    className="rounded-lg border border-[color:var(--line)] px-2 py-1.5"
                  >
                    <option value="only_missing_or_failed">only_missing_or_failed</option>
                    <option value="all">all</option>
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs">Model (optional)</span>
                  <input
                    value={enrichForm.model}
                    onChange={(event) => setEnrichForm((prev) => ({ ...prev, model: event.target.value }))}
                    placeholder="gpt-4o-mini"
                    className="rounded-lg border border-[color:var(--line)] px-2 py-1.5"
                  />
                </label>
                <label className="col-span-2 flex items-center gap-2 text-xs">
                  <input
                    type="checkbox"
                    checked={enrichForm.heuristic_only}
                    onChange={(event) => setEnrichForm((prev) => ({ ...prev, heuristic_only: event.target.checked }))}
                  />
                  Heuristic-only (skip OpenAI)
                </label>
              </div>
              <button
                onClick={() => void launchEnrich()}
                disabled={runAction !== null}
                className="mt-3 w-full rounded-xl bg-[color:#c77d28] px-3 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-55"
              >
                {runAction === "enrich" ? "Launching..." : "Run Enrichment"}
              </button>
            </section>

            <section className="rounded-xl border border-[color:var(--line)] bg-white p-3 text-sm">
              <h3 className="font-semibold">Active Job</h3>
              {!activeJob ? (
                <p className="mt-2 text-[color:rgba(16,36,59,0.66)]">No active run yet.</p>
              ) : (
                <div className="mt-2 space-y-2">
                  <p>
                    <span className={`rounded-full border px-2 py-1 text-xs ${statusPillClass(activeJob.status)}`}>{activeJob.status}</span>
                  </p>
                  <p className="text-xs">Job ID: {activeJob.job_id}</p>
                  {activeJob.workflow ? <p className="text-xs">Workflow: {activeJob.workflow}</p> : null}
                  {activeJob.updated_at ? <p className="text-xs">Updated: {formatDate(activeJob.updated_at)}</p> : null}
                  {activeJob.html_url ? (
                    <a href={activeJob.html_url} target="_blank" rel="noreferrer" className="inline-block text-xs text-[color:#2d5673] underline-offset-2 hover:underline">
                      Open GitHub run
                    </a>
                  ) : null}
                </div>
              )}
              {jobError ? <p className="mt-2 rounded-md border border-rose-200 bg-rose-50 px-2 py-1 text-xs text-rose-700">{jobError}</p> : null}
            </section>
          </div>
        </article>
      </section>
    </main>
  );
}
