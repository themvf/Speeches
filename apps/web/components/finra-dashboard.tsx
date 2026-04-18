"use client";

import { useEffect, useRef, useState } from "react";
import type {
  FINRAHeatmapPayload,
  FINRAHeatmapRule,
  FINRARecentCase,
} from "@/app/api/finra/heatmap/route";

/* ─── Helpers ────────────────────────────────────────────────────────────── */
function fmtMonth(ym: string): string {
  const [y, mo] = ym.split("-");
  const names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  return `${names[(Number(mo) || 1) - 1]} '${y.slice(2)}`;
}

function fmtDate(value: string): string {
  if (!value) return "—";
  const d = new Date(value.length <= 7 ? `${value}-01` : value);
  return Number.isNaN(d.getTime())
    ? value
    : d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
}

function cellBg(count: number, max: number): string {
  if (count === 0 || max === 0) return "rgba(9,20,36,0.55)";
  const r = count / max;
  if (r < 0.15) return "rgba(79,213,255,0.14)";
  if (r < 0.35) return "rgba(79,213,255,0.30)";
  if (r < 0.55) return "rgba(79,213,255,0.50)";
  if (r < 0.75) return "rgba(79,213,255,0.70)";
  return "rgba(79,213,255,0.92)";
}

function cellGlow(count: number, max: number): string {
  if (count === 0 || max === 0) return "none";
  const r = count / max;
  if (r < 0.45) return "none";
  if (r < 0.70) return "0 0 8px rgba(79,213,255,0.28)";
  return "0 0 14px rgba(79,213,255,0.55), 0 0 3px rgba(79,213,255,0.8)";
}

function cellBorder(count: number, max: number): string {
  if (count === 0 || max === 0) return "1px solid rgba(79,213,255,0.06)";
  const r = count / max;
  if (r < 0.35) return "1px solid rgba(79,213,255,0.15)";
  return "1px solid rgba(79,213,255,0.35)";
}

function docTypeClass(dt: string): string {
  const u = (dt || "").toUpperCase();
  if (u.includes("OHO")) return "bg-amber-500/10 text-amber-400 border border-amber-500/25";
  if (u === "NAC") return "bg-violet-500/10 text-violet-400 border border-violet-500/25";
  return "bg-[color:rgba(79,213,255,0.1)] text-[color:var(--accent)] border border-[color:rgba(79,213,255,0.22)]";
}

/* ─── Stat card ──────────────────────────────────────────────────────────── */
function StatCard({
  label,
  value,
  accent = false,
}: {
  label: string;
  value: string | number;
  accent?: boolean;
}) {
  return (
    <div className="relative overflow-hidden rounded-2xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.65)] px-5 py-4">
      <div
        className="absolute inset-x-0 top-0 h-px"
        style={{
          background: accent
            ? "linear-gradient(90deg, transparent, rgba(79,213,255,0.7) 50%, transparent)"
            : "linear-gradient(90deg, transparent, rgba(79,213,255,0.3) 50%, transparent)",
        }}
      />
      <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
        {label}
      </p>
      <p
        className="mt-1.5 text-3xl font-bold tabular-nums"
        style={{
          color: accent ? "var(--accent)" : "var(--ink)",
          textShadow: accent ? "0 0 20px rgba(79,213,255,0.35)" : "none",
        }}
      >
        {value}
      </p>
    </div>
  );
}

/* ─── Legend ─────────────────────────────────────────────────────────────── */
function Legend() {
  const steps = [0, 0.15, 0.35, 0.60, 1.0];
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[11px] text-[color:var(--ink-faint)]">Less</span>
      {steps.map((r, i) => (
        <div
          key={i}
          className="h-4 w-4 rounded-sm"
          style={{
            background: r === 0 ? "rgba(9,20,36,0.55)" : `rgba(79,213,255,${0.14 + r * 0.78})`,
            border: "1px solid rgba(79,213,255,0.15)",
            boxShadow: r > 0.6 ? "0 0 6px rgba(79,213,255,0.4)" : "none",
          }}
        />
      ))}
      <span className="text-[11px] text-[color:var(--ink-faint)]">More</span>
    </div>
  );
}

/* ─── Tooltip ────────────────────────────────────────────────────────────── */
interface TooltipState {
  rule: string;
  label: string;
  month: string;
  count: number;
  x: number;
  y: number;
}

function HeatmapTooltip({ tip }: { tip: TooltipState }) {
  return (
    <div
      className="pointer-events-none fixed z-50"
      style={{ left: tip.x, top: tip.y, transform: "translateX(-50%) translateY(-100%)" }}
    >
      <div
        className="rounded-xl px-3 py-2 text-xs"
        style={{
          background: "rgba(6,14,24,0.96)",
          border: "1px solid rgba(79,213,255,0.25)",
          boxShadow: "0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px rgba(79,213,255,0.08)",
          backdropFilter: "blur(12px)",
          minWidth: 170,
        }}
      >
        <p className="font-semibold text-[color:var(--ink)]">
          FINRA Rule {tip.rule}
          {tip.label && (
            <span className="ml-1.5 font-normal text-[color:var(--ink-faint)]">
              · {tip.label}
            </span>
          )}
        </p>
        <p className="mt-0.5 text-[color:var(--ink-faint)]">
          {fmtMonth(tip.month)} &nbsp;·&nbsp;{" "}
          <span
            className="font-semibold tabular-nums"
            style={{ color: tip.count > 0 ? "var(--accent)" : "var(--ink-faint)" }}
          >
            {tip.count} {tip.count === 1 ? "document" : "documents"}
          </span>
        </p>
      </div>
    </div>
  );
}

/* ─── Heatmap grid ───────────────────────────────────────────────────────── */
function RuleHeatmap({
  rules,
  months,
  maxVal,
}: {
  rules: FINRAHeatmapRule[];
  months: string[];
  maxVal: number;
}) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  if (rules.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <div
          className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl"
          style={{ background: "rgba(79,213,255,0.06)", border: "1px solid rgba(79,213,255,0.12)" }}
        >
          <span className="text-3xl">⬜</span>
        </div>
        <p className="text-sm font-medium text-[color:var(--ink)]">No rule citation data yet</p>
        <p className="mt-1.5 max-w-sm text-xs text-[color:var(--ink-faint)]">
          Discover and extract FINRA AWC documents in Streamlit to populate the heatmap. Rule citations are extracted automatically from the full PDF text.
        </p>
      </div>
    );
  }

  return (
    <>
      {tooltip && <HeatmapTooltip tip={tooltip} />}
      <div className="overflow-x-auto">
        <div style={{ minWidth: `${220 + months.length * 32}px` }}>
          {/* Month header row */}
          <div className="mb-1 flex">
            <div style={{ width: 220, flexShrink: 0 }} />
            <div className="flex gap-0.5">
              {months.map((m) => (
                <div
                  key={m}
                  className="w-7 text-center text-[9px] font-medium leading-none text-[color:var(--ink-faint)]"
                  style={{ writingMode: "vertical-rl", paddingBottom: 4, height: 38 }}
                >
                  {fmtMonth(m)}
                </div>
              ))}
            </div>
          </div>

          {/* Rule rows */}
          <div className="space-y-0.5">
            {rules.map((r) => (
              <div key={r.rule} className="flex items-center">
                {/* Rule label */}
                <div
                  className="flex shrink-0 flex-col pr-3"
                  style={{ width: 220 }}
                >
                  <span className="text-xs font-semibold text-[color:var(--ink)]">
                    Rule {r.rule}
                  </span>
                  {r.label && (
                    <span
                      className="mt-0.5 truncate text-[10px] text-[color:var(--ink-faint)]"
                      title={r.label}
                    >
                      {r.label}
                    </span>
                  )}
                </div>

                {/* Cells */}
                <div className="flex gap-0.5">
                  {r.by_month.map((count, mIdx) => (
                    <div
                      key={mIdx}
                      className="h-7 w-7 cursor-pointer rounded-sm transition-all duration-100 hover:scale-110 hover:z-10"
                      style={{
                        background: cellBg(count, maxVal),
                        boxShadow: cellGlow(count, maxVal),
                        border: cellBorder(count, maxVal),
                      }}
                      onMouseEnter={(e) => {
                        const rect = e.currentTarget.getBoundingClientRect();
                        setTooltip({
                          rule: r.rule,
                          label: r.label,
                          month: months[mIdx],
                          count,
                          x: rect.left + rect.width / 2,
                          y: rect.top - 10,
                        });
                      }}
                      onMouseLeave={() => setTooltip(null)}
                    />
                  ))}

                  {/* Total badge */}
                  <div className="ml-2 flex items-center">
                    <span
                      className="rounded-full px-1.5 py-0.5 text-[10px] font-semibold tabular-nums"
                      style={{
                        background: "rgba(79,213,255,0.08)",
                        color: "var(--ink-faint)",
                        border: "1px solid rgba(79,213,255,0.12)",
                      }}
                    >
                      {r.total}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

/* ─── Type distribution bars ─────────────────────────────────────────────── */
function TypeDistribution({
  counts,
  total,
}: {
  counts: Record<string, number>;
  total: number;
}) {
  const types = Object.entries(counts).sort((a, b) => b[1] - a[1]);

  const barColor: Record<string, string> = {
    AWC: "var(--accent)",
    "OHO Decision": "#f09b3d",
    NAC: "#a78bfa",
  };

  return (
    <div className="space-y-3">
      {types.map(([type, count]) => {
        const pct = total > 0 ? (count / total) * 100 : 0;
        const color = Object.entries(barColor).find(([k]) =>
          type.toUpperCase().includes(k.toUpperCase())
        )?.[1] ?? "var(--ink-faint)";
        return (
          <div key={type}>
            <div className="mb-1 flex items-center justify-between">
              <span className="text-xs font-medium text-[color:var(--ink)]">{type}</span>
              <span className="text-xs tabular-nums text-[color:var(--ink-faint)]">
                {count} &nbsp;·&nbsp; {pct.toFixed(0)}%
              </span>
            </div>
            <div
              className="h-2 w-full overflow-hidden rounded-full"
              style={{ background: "rgba(9,20,36,0.7)", border: "1px solid rgba(79,213,255,0.08)" }}
            >
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{
                  width: `${pct}%`,
                  background: color,
                  boxShadow: `0 0 8px ${color}55`,
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ─── Recent case row ────────────────────────────────────────────────────── */
function CaseRow({ c }: { c: FINRARecentCase }) {
  const inner = (
    <div
      className="flex items-start gap-3 rounded-xl px-3 py-2.5 transition-colors"
      style={{ background: "rgba(9,20,36,0.4)" }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLDivElement).style.background = "rgba(79,213,255,0.04)";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLDivElement).style.background = "rgba(9,20,36,0.4)";
      }}
    >
      <div className="min-w-0 flex-1">
        <p className="truncate text-xs font-medium text-[color:var(--ink)]">{c.title || c.document_id}</p>
        <p className="mt-0.5 text-[10px] text-[color:var(--ink-faint)]">{fmtDate(c.date)}</p>
      </div>
      <span
        className={`mt-0.5 shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold ${docTypeClass(c.doc_type)}`}
      >
        {c.doc_type}
      </span>
    </div>
  );

  return c.url ? (
    <a href={c.url} target="_blank" rel="noopener noreferrer" className="block">
      {inner}
    </a>
  ) : (
    inner
  );
}

/* ─── Main dashboard ─────────────────────────────────────────────────────── */
interface ApiEnvelope<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

export function FINRADashboard() {
  const [payload, setPayload] = useState<FINRAHeatmapPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/finra/heatmap")
      .then((r) => r.json() as Promise<ApiEnvelope<FINRAHeatmapPayload>>)
      .then((env) => {
        if (!env.ok || !env.data) throw new Error(env.error || "Failed to load");
        setPayload(env.data);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const total = payload?.total_cases ?? 0;
  const typeCounts = payload?.doc_type_counts ?? {};

  return (
    <div className="space-y-6">
      {/* ── Stats row ───────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard label="Total Actions" value={loading ? "—" : total.toLocaleString()} accent />
        <StatCard label="AWC" value={loading ? "—" : (typeCounts["AWC"] ?? 0).toLocaleString()} />
        <StatCard label="OHO Decisions" value={loading ? "—" : (typeCounts["OHO Decision"] ?? 0).toLocaleString()} />
        <StatCard label="NAC" value={loading ? "—" : (typeCounts["NAC"] ?? 0).toLocaleString()} />
      </div>

      {/* ── Rule heatmap ────────────────────────────────────────────────── */}
      <div
        className="rounded-2xl"
        style={{
          border: "1px solid rgba(79,213,255,0.16)",
          background: "rgba(9,21,34,0.55)",
          backdropFilter: "blur(8px)",
          boxShadow: "0 0 0 1px rgba(79,213,255,0.04), 0 24px 56px rgba(0,0,0,0.35)",
        }}
      >
        {/* Panel header */}
        <div
          className="flex flex-wrap items-center justify-between gap-3 border-b px-5 py-4"
          style={{ borderColor: "rgba(79,213,255,0.1)" }}
        >
          <div>
            <h2 className="text-sm font-semibold text-[color:var(--ink)]">
              Rule Violation Activity
            </h2>
            <p className="mt-0.5 text-xs text-[color:var(--ink-faint)]">
              FINRA rules cited in enforcement documents · last 18 months
            </p>
          </div>
          <Legend />
        </div>

        {/* Heatmap body */}
        <div className="px-5 py-5">
          {loading && (
            <div className="flex items-center justify-center py-16 text-sm text-[color:var(--ink-faint)]">
              Loading heatmap…
            </div>
          )}
          {!loading && error && (
            <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-xs text-red-400">
              {error}
            </div>
          )}
          {!loading && !error && payload && (
            <RuleHeatmap
              rules={payload.rules}
              months={payload.months}
              maxVal={payload.max_cell_value}
            />
          )}
        </div>
      </div>

      {/* ── Bottom row ──────────────────────────────────────────────────── */}
      {!loading && !error && payload && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
          {/* Recent cases */}
          <div
            className="rounded-2xl border lg:col-span-3"
            style={{
              borderColor: "rgba(79,213,255,0.12)",
              background: "rgba(9,21,34,0.5)",
            }}
          >
            <div
              className="border-b px-4 py-3"
              style={{ borderColor: "rgba(79,213,255,0.08)" }}
            >
              <h3 className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
                Recent Actions
              </h3>
            </div>
            <div className="space-y-px p-2 max-h-[420px] overflow-y-auto">
              {payload.recent_cases.length === 0 ? (
                <p className="py-6 text-center text-xs text-[color:var(--ink-faint)]">
                  No documents ingested yet.
                </p>
              ) : (
                payload.recent_cases.map((c) => <CaseRow key={c.document_id || c.url} c={c} />)
              )}
            </div>
          </div>

          {/* Type distribution */}
          <div
            className="rounded-2xl border lg:col-span-2"
            style={{
              borderColor: "rgba(79,213,255,0.12)",
              background: "rgba(9,21,34,0.5)",
            }}
          >
            <div
              className="border-b px-4 py-3"
              style={{ borderColor: "rgba(79,213,255,0.08)" }}
            >
              <h3 className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
                Action Type Distribution
              </h3>
            </div>
            <div className="p-4">
              {Object.keys(typeCounts).length === 0 ? (
                <p className="py-4 text-center text-xs text-[color:var(--ink-faint)]">
                  No data yet.
                </p>
              ) : (
                <TypeDistribution counts={typeCounts} total={total} />
              )}

              {/* Divider + footnote */}
              <div
                className="mt-6 border-t pt-4"
                style={{ borderColor: "rgba(79,213,255,0.08)" }}
              >
                <p className="text-[11px] leading-relaxed text-[color:var(--ink-faint)]">
                  <strong className="text-[color:var(--ink)]">AWC</strong> — Acceptance, Waiver &amp; Consent
                  <br />
                  <strong className="text-[color:var(--ink)]">OHO</strong> — Office of Hearing Officers
                  <br />
                  <strong className="text-[color:var(--ink)]">NAC</strong> — National Adjudicatory Council
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
