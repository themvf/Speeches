"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { expandQuery } from "@/lib/synonyms";
import type {
  EnforcementBetaAction,
  EnforcementBetaAgencyPayload,
  EnforcementBetaCitation,
  EnforcementBetaCitationActivity,
  EnforcementBetaPayload
} from "@/app/api/enforcement/beta/route";

type ApiEnvelope<T> = { ok: boolean; data?: T; error?: string };
type ViewMode = "combined" | "sec" | "finra";

const AGENCY_STYLE = {
  SEC: {
    color: "#ff6b7f",
    muted: "rgba(255,107,127,0.12)",
    line: "rgba(255,107,127,0.28)",
    glow: "rgba(255,107,127,0.32)"
  },
  FINRA: {
    color: "var(--accent)",
    muted: "rgba(79,213,255,0.12)",
    line: "rgba(79,213,255,0.28)",
    glow: "rgba(79,213,255,0.32)"
  }
} as const;

function agencyKey(agency: "SEC" | "FINRA"): "sec" | "finra" {
  return agency === "SEC" ? "sec" : "finra";
}

function formatNumber(value: number): string {
  return value.toLocaleString("en-US");
}

function formatDate(value: string): string {
  if (!value) {
    return "Unavailable";
  }
  const parsed = new Date(value.length <= 7 ? `${value}-01T00:00:00Z` : value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC"
  });
}

function formatMonth(value: string): string {
  const parsed = new Date(`${value}-01T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleDateString("en-US", {
    month: "short",
    year: "2-digit",
    timeZone: "UTC"
  });
}

function dateMs(value: string): number {
  const parsed = new Date(value.length <= 7 ? `${value}-01T00:00:00Z` : value);
  const ms = parsed.getTime();
  return Number.isNaN(ms) ? 0 : ms;
}

function monthDateRange(month: string): { from: string; to: string } {
  const [yearStr, monStr] = month.split("-");
  const lastDay = new Date(Date.UTC(Number(yearStr), Number(monStr), 0)).getUTCDate();
  return { from: `${month}-01`, to: `${month}-${String(lastDay).padStart(2, "0")}` };
}

function citationKey(citation: EnforcementBetaCitation): string {
  return `${citation.agency}:${citation.citation}`;
}

function selectClass(): string {
  return "form-control min-h-11 w-full px-3 text-sm";
}

function inputClass(): string {
  return "form-control min-h-11 w-full px-3 text-sm";
}

function cellBackground(count: number, max: number, agency: "SEC" | "FINRA"): string {
  if (count <= 0 || max <= 0) {
    return "rgba(9,20,36,0.62)";
  }
  const base = agency === "SEC" ? "255,107,127" : "79,213,255";
  const opacity = Math.max(0.12, (count / max) * 0.84);
  return `rgba(${base},${opacity.toFixed(2)})`;
}

function MetricCard({
  label,
  value,
  detail,
  accent
}: {
  label: string;
  value: string;
  detail?: string;
  accent?: string;
}) {
  return (
    <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.64)] p-4">
      <p className="text-xs font-semibold uppercase text-[color:var(--ink-faint)]">{label}</p>
      <p className="mt-2 text-2xl font-semibold tabular-nums text-[color:var(--ink)]" style={{ color: accent, letterSpacing: 0 }}>
        {value}
      </p>
      {detail ? <p className="mt-1 text-xs text-[color:var(--ink-faint)]">{detail}</p> : null}
    </div>
  );
}

function AgencyBadge({ agency }: { agency: "SEC" | "FINRA" }) {
  const style = AGENCY_STYLE[agency];
  return (
    <span
      className="inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase"
      style={{ color: style.color, background: style.muted, border: `1px solid ${style.line}` }}
    >
      {agency}
    </span>
  );
}

function ModeButton({
  mode,
  active,
  onClick,
  children
}: {
  mode: ViewMode;
  active: boolean;
  onClick: (mode: ViewMode) => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={() => onClick(mode)}
      className="min-h-11 rounded-xl border px-4 text-sm font-semibold transition-colors"
      style={{
        background: active ? "rgba(79,213,255,0.14)" : "rgba(9,20,36,0.55)",
        borderColor: active ? "rgba(79,213,255,0.42)" : "rgba(255,255,255,0.08)",
        color: active ? "var(--ink)" : "var(--ink-faint)"
      }}
    >
      {children}
    </button>
  );
}

function QualityBar({
  label,
  value,
  total,
  color
}: {
  label: string;
  value: number;
  total: number;
  color: string;
}) {
  const pct = total > 0 ? Math.round((value / total) * 100) : 0;
  return (
    <div>
      <div className="mb-1 flex items-center justify-between gap-3">
        <span className="text-xs text-[color:var(--ink)]">{label}</span>
        <span className="text-xs tabular-nums text-[color:var(--ink-faint)]">{pct}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-[color:rgba(9,20,36,0.72)]">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

function HeatmapPanel({
  agency,
  data,
  months
}: {
  agency: "SEC" | "FINRA";
  data: EnforcementBetaAgencyPayload;
  months: string[];
}) {
  const style = AGENCY_STYLE[agency];
  const rows = data.top_citations;

  return (
    <section
      className="rounded-xl border bg-[color:rgba(9,21,34,0.52)]"
      style={{ borderColor: style.line }}
    >
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[color:var(--line-soft)] px-4 py-3">
        <div>
          <div className="flex items-center gap-2">
            <AgencyBadge agency={agency} />
            <h2 className="text-sm font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
              {agency === "SEC" ? "Statute and Rule Activity" : "Rule Violation Activity"}
            </h2>
          </div>
          <p className="mt-1 text-xs text-[color:var(--ink-faint)]">
            {formatNumber(data.total_actions)} actions, {formatNumber(data.cited_actions)} with extracted citations
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[11px] text-[color:var(--ink-faint)]">Less</span>
          {[0, 0.25, 0.5, 0.75, 1].map((step) => (
            <span
              key={step}
              className="h-4 w-4 rounded-sm border"
              style={{
                background: step === 0 ? "rgba(9,20,36,0.62)" : cellBackground(step, 1, agency),
                borderColor: style.line
              }}
            />
          ))}
          <span className="text-[11px] text-[color:var(--ink-faint)]">More</span>
        </div>
      </div>

      <div className="p-4">
        {rows.length === 0 ? (
          <div className="rounded-xl border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.58)] p-6 text-center">
            <p className="text-sm font-semibold text-[color:var(--ink)]">No citation activity found</p>
            <p className="mt-1 text-xs text-[color:var(--ink-faint)]">
              {agency === "FINRA"
                ? "FINRA AWC documents will appear here after ingestion and PDF extraction."
                : "SEC litigation releases are loaded, but citation extraction did not find rule or statute references in the current window."}
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <div style={{ minWidth: 760 }}>
              <div className="mb-2 grid items-end gap-1" style={{ gridTemplateColumns: `220px repeat(${months.length}, 26px) 56px` }}>
                <div />
                {months.map((month) => (
                  <div
                    key={month}
                    className="h-12 text-center text-[9px] font-medium leading-none text-[color:var(--ink-faint)]"
                    style={{ writingMode: "vertical-rl" }}
                  >
                    {formatMonth(month)}
                  </div>
                ))}
                <div className="text-right text-[10px] uppercase text-[color:var(--ink-faint)]">Total</div>
              </div>

              <div className="space-y-1">
                {rows.map((row) => (
                  <div
                    key={row.citation}
                    className="grid items-center gap-1"
                    style={{ gridTemplateColumns: `220px repeat(${months.length}, 26px) 56px` }}
                  >
                    <div className="min-w-0 pr-2">
                      <p className="truncate text-xs font-semibold text-[color:var(--ink)]" title={row.citation}>
                        {row.citation}
                      </p>
                      <p className="truncate text-[10px] text-[color:var(--ink-faint)]" title={row.label}>
                        {row.label}
                      </p>
                    </div>
                    {row.by_month.map((count, index) => (
                      <div
                        key={`${row.citation}-${months[index]}`}
                        className="h-6 w-6 rounded-sm border"
                        title={`${row.citation} - ${formatMonth(months[index])}: ${count}`}
                        style={{
                          background: cellBackground(count, data.max_cell_value, agency),
                          borderColor: count > 0 ? style.line : "rgba(255,255,255,0.06)",
                          boxShadow: count > data.max_cell_value * 0.65 ? `0 0 10px ${style.glow}` : "none"
                        }}
                      />
                    ))}
                    <div className="text-right text-xs font-semibold tabular-nums" style={{ color: style.color }}>
                      {row.total}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

function TopCitationList({
  title,
  rows,
  agency,
  selected,
  onSelect,
}: {
  title: string;
  rows: EnforcementBetaCitationActivity[];
  agency: "SEC" | "FINRA";
  selected: string;
  onSelect: (value: string) => void;
}) {
  const style = AGENCY_STYLE[agency];
  const max = rows.length ? Math.max(...rows.map((row) => row.total), 1) : 1;

  return (
    <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.52)] p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase text-[color:var(--ink-faint)]">{title}</h3>
        <AgencyBadge agency={agency} />
      </div>
      {rows.length === 0 ? (
        <p className="text-xs text-[color:var(--ink-faint)]">No extracted citations.</p>
      ) : (
        <div className="space-y-3">
          {rows.slice(0, 8).map((row) => {
            const key = `${agency}:${row.citation}`;
            const isActive = selected === key;
            return (
              <div
                key={row.citation}
                onClick={() => onSelect(isActive ? "" : key)}
                className="cursor-pointer rounded-md px-1.5 py-1 -mx-1.5 transition-colors hover:bg-[color:rgba(255,255,255,0.04)]"
                style={isActive ? { background: "rgba(255,255,255,0.07)" } : undefined}
              >
                <div className="mb-1 flex items-center justify-between gap-3">
                  <span className="truncate text-xs font-medium" style={{ color: isActive ? style.color : "var(--ink)" }} title={row.label}>
                    {row.citation}
                  </span>
                  <span className="text-xs tabular-nums text-[color:var(--ink-faint)]">{row.total}</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-[color:rgba(9,20,36,0.75)]">
                  <div className="h-full rounded-full" style={{ width: `${(row.total / max) * 100}%`, background: style.color, opacity: isActive ? 1 : 0.7 }} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ActionRow({ action, snippet }: { action: EnforcementBetaAction; snippet?: string }) {
  const style = AGENCY_STYLE[action.agency];
  const citationPreview = action.citations.slice(0, 3);
  const row = (
    <div className="rounded-xl border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.46)] p-4 transition-colors hover:border-[color:var(--line-strong)]">
      <div className="flex flex-wrap items-center gap-2">
        <AgencyBadge agency={action.agency} />
        <span className="rounded-full border border-[color:var(--line-soft)] px-2 py-0.5 text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">
          {action.doc_type}
        </span>
        {action.release_no ? (
          <span className="rounded-full border border-[color:var(--line-soft)] px-2 py-0.5 text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">
            {action.release_no}
          </span>
        ) : null}
        <span className="ml-auto text-xs text-[color:var(--ink-faint)]">{formatDate(action.date)}</span>
      </div>
      <h3 className="mt-3 text-sm font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
        {action.title}
      </h3>
      <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-[color:var(--ink-faint)]">
        <span>{action.action_type}</span>
        <span style={{ color: "rgba(216,231,246,0.34)" }}>|</span>
        <span>{action.outcome_status}</span>
        <span style={{ color: "rgba(216,231,246,0.34)" }}>|</span>
        <span>{action.forum}</span>
      </div>
      {action.summary ? <p className="mt-2 line-clamp-2 text-xs leading-relaxed text-[color:var(--ink-faint)]">{action.summary}</p> : null}
      {snippet ? <p className="mt-1.5 line-clamp-2 text-[11px] italic leading-relaxed text-[color:rgba(79,213,255,0.55)]">{snippet}</p> : null}
      <div className="mt-3 flex flex-wrap gap-1.5">
        {citationPreview.length > 0 ? (
          citationPreview.map((citation) => (
            <span
              key={citation.citation}
              className="rounded-full px-2 py-1 text-[10px] font-semibold"
              style={{ color: style.color, background: style.muted, border: `1px solid ${style.line}` }}
            >
              {citation.citation}
            </span>
          ))
        ) : (
          <span className="rounded-full border border-[color:var(--line-soft)] px-2 py-1 text-[10px] text-[color:var(--ink-faint)]">
            No extracted citations
          </span>
        )}
        {action.sanctions.slice(0, 2).map((sanction) => (
          <span key={sanction} className="rounded-full border border-amber-400/25 bg-amber-400/10 px-2 py-1 text-[10px] font-semibold text-amber-300">
            {sanction}
          </span>
        ))}
      </div>
    </div>
  );

  if (!action.url) {
    return row;
  }

  return (
    <a href={action.url} target="_blank" rel="noopener noreferrer" className="block">
      {row}
    </a>
  );
}

function ActionsPanel({ actions, snippets }: { actions: EnforcementBetaAction[]; snippets?: Record<string, string> }) {
  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.52)]">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[color:var(--line-soft)] px-4 py-3">
        <h2 className="text-sm font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
          Filtered Enforcement Actions
        </h2>
        <span className="text-xs tabular-nums text-[color:var(--ink-faint)]">{formatNumber(actions.length)} shown</span>
      </div>
      <div className="max-h-[680px] space-y-3 overflow-y-auto p-3">
        {actions.length === 0 ? (
          <div className="rounded-xl border border-[color:var(--line-soft)] p-6 text-center text-sm text-[color:var(--ink-faint)]">
            No actions match the active filters.
          </div>
        ) : (
          actions.slice(0, 80).map((action) => (
            <ActionRow
              key={`${action.agency}-${action.document_id || action.url || action.title}`}
              action={action}
              snippet={snippets?.[action.document_id]}
            />
          ))
        )}
      </div>
    </section>
  );
}

function Filters({
  payload,
  viewMode,
  setViewMode,
  query,
  setQuery,
  docType,
  setDocType,
  actionType,
  setActionType,
  outcome,
  setOutcome,
  citation,
  setCitation,
  dateFrom,
  setDateFrom,
  dateTo,
  setDateTo,
  semanticMode,
  setSemanticMode,
  semanticLoading,
}: {
  payload: EnforcementBetaPayload;
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  query: string;
  setQuery: (value: string) => void;
  docType: string;
  setDocType: (value: string) => void;
  actionType: string;
  setActionType: (value: string) => void;
  outcome: string;
  setOutcome: (value: string) => void;
  citation: string;
  setCitation: (value: string) => void;
  dateFrom: string;
  setDateFrom: (value: string) => void;
  dateTo: string;
  setDateTo: (value: string) => void;
  semanticMode: boolean;
  setSemanticMode: (v: boolean) => void;
  semanticLoading: boolean;
}) {
  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.56)] p-4">
      <div className="flex flex-wrap gap-2">
        <ModeButton mode="combined" active={viewMode === "combined"} onClick={setViewMode}>
          Combined
        </ModeButton>
        <ModeButton mode="sec" active={viewMode === "sec"} onClick={setViewMode}>
          SEC
        </ModeButton>
        <ModeButton mode="finra" active={viewMode === "finra"} onClick={setViewMode}>
          FINRA
        </ModeButton>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <label className="block">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-xs font-semibold uppercase text-[color:var(--ink-faint)]">Search</span>
            <button
              type="button"
              onClick={() => setSemanticMode(!semanticMode)}
              className={`flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-medium transition-colors ${
                semanticMode
                  ? "border border-[color:rgba(79,213,255,0.5)] bg-[color:rgba(79,213,255,0.12)] text-[color:var(--accent)]"
                  : "border border-[color:var(--line)] text-[color:var(--ink-faint)] hover:text-[color:var(--ink)]"
              }`}
              title="Use OpenAI vector search for concept matching (blockchain → crypto, digital assets)"
            >
              <span className={`h-1.5 w-1.5 rounded-full ${semanticMode ? "bg-[color:var(--accent)]" : "bg-[color:var(--ink-faint)]"}`} />
              Semantic{semanticLoading ? "…" : ""}
            </button>
          </div>
          <input className={inputClass()} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Title, entity, release, citation" />
        </label>

        <label className="block">
          <span className="mb-1 block text-xs font-semibold uppercase text-[color:var(--ink-faint)]">Document Type</span>
          <select className={selectClass()} value={docType} onChange={(event) => setDocType(event.target.value)}>
            <option value="">All document types</option>
            {payload.filters.doc_types.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className="mb-1 block text-xs font-semibold uppercase text-[color:var(--ink-faint)]">Action Type</span>
          <select className={selectClass()} value={actionType} onChange={(event) => setActionType(event.target.value)}>
            <option value="">All action types</option>
            {payload.filters.action_types.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className="mb-1 block text-xs font-semibold uppercase text-[color:var(--ink-faint)]">Outcome</span>
          <select className={selectClass()} value={outcome} onChange={(event) => setOutcome(event.target.value)}>
            <option value="">All outcomes</option>
            {payload.filters.outcomes.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>

        <label className="block xl:col-span-2">
          <span className="mb-1 block text-xs font-semibold uppercase text-[color:var(--ink-faint)]">Citation</span>
          <select className={selectClass()} value={citation} onChange={(event) => setCitation(event.target.value)}>
            <option value="">All citations</option>
            {payload.filters.citations.map((value) => (
              <option key={citationKey(value)} value={citationKey(value)}>
                {value.agency} - {value.citation}
              </option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className="mb-1 block text-xs font-semibold uppercase text-[color:var(--ink-faint)]">From</span>
          <input className={inputClass()} type="date" value={dateFrom} onChange={(event) => setDateFrom(event.target.value)} />
        </label>

        <label className="block">
          <span className="mb-1 block text-xs font-semibold uppercase text-[color:var(--ink-faint)]">To</span>
          <input className={inputClass()} type="date" value={dateTo} onChange={(event) => setDateTo(event.target.value)} />
        </label>
      </div>
    </section>
  );
}

export function EnforcementBetaDashboard() {
  const [payload, setPayload] = useState<EnforcementBetaPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("combined");
  const [query, setQuery] = useState("");
  const [docType, setDocType] = useState("");
  const [actionType, setActionType] = useState("");
  const [outcome, setOutcome] = useState("");
  const [citation, setCitation] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [semanticMode, setSemanticMode] = useState(false);
  const [semanticDocIds, setSemanticDocIds] = useState<Set<string> | null>(null);
  const [semanticLoading, setSemanticLoading] = useState(false);
  const [semanticSnippets, setSemanticSnippets] = useState<Record<string, string>>({});
  const semanticAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    fetch("/api/enforcement/beta")
      .then((response) => response.json() as Promise<ApiEnvelope<EnforcementBetaPayload>>)
      .then((envelope) => {
        if (!envelope.ok || !envelope.data) {
          throw new Error(envelope.error || "Failed to load enforcement beta data");
        }
        setPayload(envelope.data);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    const q = query.trim();
    if (!semanticMode || !q) {
      setSemanticDocIds(null);
      setSemanticSnippets({});
      return;
    }
    if (semanticAbortRef.current) semanticAbortRef.current.abort();
    const controller = new AbortController();
    semanticAbortRef.current = controller;
    setSemanticLoading(true);
    const expanded = expandQuery(q);
    fetch(`/api/search?q=${encodeURIComponent(expanded)}&topK=40`, {
      signal: controller.signal,
      cache: "no-store",
    })
      .then((res) => res.json() as Promise<{ ok: boolean; data?: { document_ids: string[]; snippets: Record<string, string> }; error?: string }>)
      .then((env) => {
        if (env.ok && env.data?.document_ids) {
          setSemanticDocIds(new Set(env.data.document_ids));
          setSemanticSnippets(env.data.snippets || {});
        } else {
          setSemanticDocIds(new Set());
          setSemanticSnippets({});
        }
      })
      .catch(() => {})
      .finally(() => setSemanticLoading(false));
  }, [query, semanticMode]);

  const filteredActions = useMemo(() => {
    if (!payload) {
      return [];
    }
    const q = query.trim().toLowerCase();
    const fromMs = dateFrom ? dateMs(`${dateFrom}T00:00:00Z`) : 0;
    const toMs = dateTo ? dateMs(`${dateTo}T23:59:59Z`) : 0;

    return payload.combined_actions.filter((action) => {
      if (viewMode !== "combined" && agencyKey(action.agency) !== viewMode) {
        return false;
      }
      if (docType && action.doc_type !== docType) {
        return false;
      }
      if (actionType && action.action_type !== actionType) {
        return false;
      }
      if (outcome && action.outcome_status !== outcome) {
        return false;
      }
      if (citation && !action.citations.some((item) => citationKey(item) === citation)) {
        return false;
      }
      const actionMs = dateMs(action.date);
      if (fromMs && (!actionMs || actionMs < fromMs)) {
        return false;
      }
      if (toMs && (!actionMs || actionMs > toMs)) {
        return false;
      }
      if (!q) {
        return true;
      }
      if (semanticMode && semanticDocIds !== null) {
        return semanticDocIds.has(action.document_id);
      }
      const haystack = [
        action.title,
        action.release_no,
        action.doc_type,
        action.action_type,
        action.outcome_status,
        action.forum,
        action.summary,
        ...action.entities,
        ...action.citations.map((item) => `${item.citation} ${item.label}`)
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(q);
    });
  }, [actionType, citation, dateFrom, dateTo, docType, outcome, payload, query, semanticDocIds, semanticMode, viewMode]);

  const visibleAgencies = useMemo(() => {
    if (!payload) {
      return [];
    }
    if (viewMode === "sec") {
      return [payload.agencies.sec];
    }
    if (viewMode === "finra") {
      return [payload.agencies.finra];
    }
    return [payload.agencies.sec, payload.agencies.finra];
  }, [payload, viewMode]);

  if (loading) {
    return (
      <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.55)] p-8 text-sm text-[color:var(--ink-faint)]">
        Loading Enforcement...
      </div>
    );
  }

  if (error || !payload) {
    return (
      <div className="rounded-xl border border-red-400/25 bg-red-500/10 p-4 text-sm text-red-200">
        {error || "Enforcement data is unavailable."}
      </div>
    );
  }

  const filteredSecCount = filteredActions.filter((action) => action.agency === "SEC").length;
  const filteredFinraCount = filteredActions.filter((action) => action.agency === "FINRA").length;
  const filteredCitedCount = filteredActions.filter((action) => action.data_quality.has_citations).length;
  const filteredCoverage = filteredActions.length ? Math.round((filteredCitedCount / filteredActions.length) * 100) : 0;

  return (
    <div className="space-y-6">
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <MetricCard label="Showing" value={formatNumber(filteredActions.length)} detail={`${formatNumber(payload.totals.combined_actions)} total actions`} />
        <MetricCard label="SEC" value={formatNumber(filteredSecCount)} detail={`${formatNumber(payload.totals.sec_actions)} total`} accent={AGENCY_STYLE.SEC.color} />
        <MetricCard label="FINRA" value={formatNumber(filteredFinraCount)} detail={`${formatNumber(payload.totals.finra_actions)} total`} accent={AGENCY_STYLE.FINRA.color} />
        <MetricCard label="Citation Coverage" value={`${filteredCoverage}%`} detail={`${formatNumber(filteredCitedCount)} filtered cited actions`} accent="var(--ok)" />
        <MetricCard label="Latest Action" value={formatDate(payload.totals.latest_action_date)} detail="Across SEC and FINRA" />
      </div>

      <Filters
        payload={payload}
        viewMode={viewMode}
        setViewMode={setViewMode}
        query={query}
        setQuery={setQuery}
        docType={docType}
        setDocType={setDocType}
        actionType={actionType}
        setActionType={setActionType}
        outcome={outcome}
        setOutcome={setOutcome}
        citation={citation}
        setCitation={setCitation}
        dateFrom={dateFrom}
        setDateFrom={setDateFrom}
        dateTo={dateTo}
        setDateTo={setDateTo}
        semanticMode={semanticMode}
        setSemanticMode={setSemanticMode}
        semanticLoading={semanticLoading}
      />

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="space-y-4 xl:col-span-1">
          <TopCitationList title="SEC Top Citations" rows={payload.agencies.sec.top_citations} agency="SEC" selected={citation} onSelect={setCitation} />
          <TopCitationList title="FINRA Top Citations" rows={payload.agencies.finra.top_citations} agency="FINRA" selected={citation} onSelect={setCitation} />
          <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.52)] p-4">
            <h3 className="text-xs font-semibold uppercase text-[color:var(--ink-faint)]">Data Quality</h3>
            <div className="mt-4 space-y-4">
              {(["SEC", "FINRA"] as const).map((agency) => {
                const data = payload.agencies[agencyKey(agency)];
                return (
                  <div key={agency} className="space-y-3">
                    <div className="flex items-center justify-between">
                      <AgencyBadge agency={agency} />
                      <span className="text-xs text-[color:var(--ink-faint)]">{formatNumber(data.total_actions)} actions</span>
                    </div>
                    <QualityBar label="Dated" value={data.dated_actions} total={data.total_actions} color={AGENCY_STYLE[agency].color} />
                    <QualityBar label="Full Text" value={data.full_text_actions} total={data.total_actions} color="var(--ok)" />
                    <QualityBar label="Citations" value={data.cited_actions} total={data.total_actions} color="var(--warn)" />
                  </div>
                );
              })}
            </div>
          </div>
        </div>
        <div className="xl:col-span-2">
          <ActionsPanel actions={filteredActions} snippets={semanticMode ? semanticSnippets : undefined} />
        </div>
      </div>
    </div>
  );
}
