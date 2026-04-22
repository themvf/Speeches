"use client";

import { useEffect, useRef, useState } from "react";

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

const NEWS_ENRICH_FIELDS: FieldDef[] = [
  { name: "enrich_limit", label: "Max articles to enrich", type: "number", default: "25" },
  {
    name: "mode",
    label: "Enrichment mode",
    type: "select",
    default: "only_missing_or_failed",
    options: [
      { value: "only_missing_or_failed", label: "Missing / Failed only" },
      { value: "all", label: "All (re-enrich)" },
    ],
  },
  { name: "source_kind", label: "Source kind", type: "text", default: "newsapi_article" },
  { name: "model", label: "Model override", type: "text", placeholder: "e.g. gpt-4o (leave blank for default)" },
  { name: "heuristic_only", label: "Skip OpenAI (heuristic only)", type: "boolean", default: "false" },
];

const TRENDS_FIELDS: FieldDef[] = [
  { name: "min_mentions", label: "Min tag mentions", type: "number", default: "5" },
  { name: "dry_run", label: "Dry run (skip OpenAI calls)", type: "boolean", default: "false" },
];

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
              Dispatched — check GitHub Actions for progress
            </span>
          )}
          {status === "error" && (
            <span className="text-sm text-[color:var(--danger)]">
              Failed{error ? `: ${error}` : " — try again"}
            </span>
          )}
        </div>
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
        title="Financial News Enrich"
        description="Run OpenAI enrichment on ingested news articles (tags, summary, stance)."
        workflowFile="financial-news-enrich.yml"
        fields={NEWS_ENRICH_FIELDS}
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
