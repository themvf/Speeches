"use client";

import { useEffect, useState } from "react";
import type {
  AgencyHeatmap,
  EnforcementHeatmapPayload,
  HeatmapRule,
  RecentCase,
} from "@/app/api/enforcement/heatmap/route";

/* ─── Color tokens ───────────────────────────────────────────────────────── */
const COLORS = {
  finra: {
    base:    [79, 213, 255] as const,   // cyan
    accent:  "rgba(79,213,255,",
    text:    "#4fd5ff",
    border:  "rgba(79,213,255,",
    glow:    "rgba(79,213,255,",
    badge:   "bg-[color:rgba(79,213,255,0.10)] text-[color:#4fd5ff] border border-[color:rgba(79,213,255,0.22)]",
    tab:     "rgba(79,213,255,0.12)",
    tabText: "#4fd5ff",
    gradient:"linear-gradient(90deg, transparent, rgba(79,213,255,0.7) 50%, transparent)",
  },
  sec: {
    base:    [255, 80, 80] as const,    // red
    accent:  "rgba(255,80,80,",
    text:    "#ff5050",
    border:  "rgba(255,80,80,",
    glow:    "rgba(255,80,80,",
    badge:   "bg-[color:rgba(255,80,80,0.10)] text-[color:#ff5050] border border-[color:rgba(255,80,80,0.22)]",
    tab:     "rgba(255,80,80,0.12)",
    tabText: "#ff5050",
    gradient:"linear-gradient(90deg, transparent, rgba(255,80,80,0.7) 50%, transparent)",
  },
} as const;

type Agency = "finra" | "sec";

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

function cellBg(count: number, max: number, c: typeof COLORS[Agency]): string {
  if (count === 0 || max === 0) return "rgba(9,20,36,0.55)";
  const r = count / max;
  const a = c.accent;
  if (r < 0.15) return `${a}0.14)`;
  if (r < 0.35) return `${a}0.30)`;
  if (r < 0.55) return `${a}0.50)`;
  if (r < 0.75) return `${a}0.70)`;
  return `${a}0.92)`;
}

function cellGlow(count: number, max: number, c: typeof COLORS[Agency]): string {
  if (count === 0 || max === 0) return "none";
  const r = count / max;
  const g = c.glow;
  if (r < 0.45) return "none";
  if (r < 0.70) return `0 0 8px ${g}0.28)`;
  return `0 0 14px ${g}0.55), 0 0 3px ${g}0.8)`;
}

function cellBorder(count: number, max: number, c: typeof COLORS[Agency]): string {
  const b = c.border;
  if (count === 0 || max === 0) return `1px solid ${b}0.06)`;
  const r = count / max;
  if (r < 0.35) return `1px solid ${b}0.15)`;
  return `1px solid ${b}0.35)`;
}

/* ─── Tooltip ────────────────────────────────────────────────────────────── */
interface TooltipState {
  rule: string; label: string; month: string; count: number;
  x: number; y: number; agency: Agency;
}

function HeatmapTooltip({ tip }: { tip: TooltipState }) {
  const c = COLORS[tip.agency];
  const prefix = tip.agency === "finra" ? "FINRA Rule" : "SEC §";
  return (
    <div
      className="pointer-events-none fixed z-50"
      style={{ left: tip.x, top: tip.y, transform: "translateX(-50%) translateY(-100%)" }}
    >
      <div
        className="rounded-xl px-3 py-2 text-xs"
        style={{
          background: "rgba(6,14,24,0.96)",
          border: `1px solid ${c.border}0.25)`,
          boxShadow: `0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px ${c.border}0.08)`,
          backdropFilter: "blur(12px)",
          minWidth: 170,
        }}
      >
        <p className="font-semibold text-[color:var(--ink)]">
          {prefix} {tip.rule}
          {tip.label && (
            <span className="ml-1.5 font-normal text-[color:var(--ink-faint)]">
              · {tip.label}
            </span>
          )}
        </p>
        <p className="mt-0.5 text-[color:var(--ink-faint)]">
          {fmtMonth(tip.month)} &nbsp;·&nbsp;{" "}
          <span className="font-semibold tabular-nums" style={{ color: tip.count > 0 ? c.text : undefined }}>
            {tip.count} {tip.count === 1 ? "document" : "documents"}
          </span>
        </p>
      </div>
    </div>
  );
}

/* ─── Legend ─────────────────────────────────────────────────────────────── */
function Legend({ agency }: { agency: Agency }) {
  const c = COLORS[agency];
  const steps = [0, 0.15, 0.35, 0.60, 1.0];
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[11px] text-[color:var(--ink-faint)]">Less</span>
      {steps.map((r, i) => (
        <div
          key={i}
          className="h-4 w-4 rounded-sm"
          style={{
            background: r === 0 ? "rgba(9,20,36,0.55)" : `${c.accent}${0.14 + r * 0.78})`,
            border: `1px solid ${c.border}0.15)`,
            boxShadow: r > 0.6 ? `0 0 6px ${c.glow}0.4)` : "none",
          }}
        />
      ))}
      <span className="text-[11px] text-[color:var(--ink-faint)]">More</span>
    </div>
  );
}

/* ─── Stat card ──────────────────────────────────────────────────────────── */
function StatCard({
  label, value, color, gradient, dimmed = false,
}: {
  label: string; value: string | number;
  color: string; gradient: string; dimmed?: boolean;
}) {
  return (
    <div
      className="relative overflow-hidden rounded-2xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.65)] px-5 py-4"
      style={dimmed ? { opacity: 0.55 } : {}}
    >
      <div className="absolute inset-x-0 top-0 h-px" style={{ background: gradient }} />
      <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
        {label}
      </p>
      <p className="mt-1.5 text-3xl font-bold tabular-nums" style={{ color, textShadow: `0 0 20px ${color}55` }}>
        {value}
      </p>
    </div>
  );
}

/* ─── Heatmap grid ───────────────────────────────────────────────────────── */
function RuleHeatmap({
  agency, rules, months, maxVal,
}: {
  agency: Agency; rules: HeatmapRule[]; months: string[]; maxVal: number;
}) {
  const c = COLORS[agency];
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  if (rules.length === 0) {
    const hint = agency === "finra"
      ? "Discover and extract FINRA AWC documents in Streamlit — rule citations are pulled from the full PDF text."
      : "SEC enforcement actions are loaded. Try enriching documents to extract statute citations.";
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <div
          className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl"
          style={{ background: `${c.accent}0.06)`, border: `1px solid ${c.border}0.12)` }}
        >
          <span className="text-3xl">⬜</span>
        </div>
        <p className="text-sm font-medium text-[color:var(--ink)]">No citation data yet</p>
        <p className="mt-1.5 max-w-sm text-xs text-[color:var(--ink-faint)]">{hint}</p>
      </div>
    );
  }

  const rulePrefix = agency === "finra" ? "Rule" : "§";

  return (
    <>
      {tooltip && <HeatmapTooltip tip={tooltip} />}
      <div className="overflow-x-auto">
        <div style={{ minWidth: `${220 + months.length * 32}px` }}>
          {/* Month header */}
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
                <div className="flex shrink-0 flex-col pr-3" style={{ width: 220 }}>
                  <span className="text-xs font-semibold text-[color:var(--ink)]">
                    {rulePrefix} {r.rule}
                  </span>
                  {r.label && (
                    <span className="mt-0.5 truncate text-[10px] text-[color:var(--ink-faint)]" title={r.label}>
                      {r.label}
                    </span>
                  )}
                </div>
                <div className="flex gap-0.5">
                  {r.by_month.map((count, mIdx) => (
                    <div
                      key={mIdx}
                      className="h-7 w-7 cursor-pointer rounded-sm transition-all duration-100 hover:scale-110 hover:z-10"
                      style={{
                        background: cellBg(count, maxVal, c),
                        boxShadow: cellGlow(count, maxVal, c),
                        border: cellBorder(count, maxVal, c),
                      }}
                      onMouseEnter={(e) => {
                        const rect = e.currentTarget.getBoundingClientRect();
                        setTooltip({ rule: r.rule, label: r.label, month: months[mIdx], count, x: rect.left + rect.width / 2, y: rect.top - 10, agency });
                      }}
                      onMouseLeave={() => setTooltip(null)}
                    />
                  ))}
                  <div className="ml-2 flex items-center">
                    <span
                      className="rounded-full px-1.5 py-0.5 text-[10px] font-semibold tabular-nums"
                      style={{ background: `${c.accent}0.08)`, color: "var(--ink-faint)", border: `1px solid ${c.border}0.12)` }}
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

/* ─── Recent case row ────────────────────────────────────────────────────── */
function CaseRow({ c: cas }: { c: RecentCase }) {
  const col = COLORS[cas.agency === "FINRA" ? "finra" : "sec"];
  const inner = (
    <div
      className="flex items-start gap-3 rounded-xl px-3 py-2.5 transition-colors"
      style={{ background: "rgba(9,20,36,0.4)" }}
      onMouseEnter={(e) => { (e.currentTarget as HTMLDivElement).style.background = `${col.accent}0.04)`; }}
      onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.background = "rgba(9,20,36,0.4)"; }}
    >
      <div
        className="mt-1 h-2 w-2 shrink-0 rounded-full"
        style={{ background: col.text, boxShadow: `0 0 6px ${col.text}88`, flexShrink: 0 }}
      />
      <div className="min-w-0 flex-1">
        <p className="truncate text-xs font-medium text-[color:var(--ink)]">{cas.title || cas.document_id}</p>
        <p className="mt-0.5 text-[10px] text-[color:var(--ink-faint)]">
          {cas.agency} &nbsp;·&nbsp; {fmtDate(cas.date)}
        </p>
      </div>
      <span
        className="mt-0.5 shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold"
        style={{
          background: `${col.accent}0.10)`,
          color: col.text,
          border: `1px solid ${col.border}0.22)`,
        }}
      >
        {cas.doc_type}
      </span>
    </div>
  );

  return cas.url ? (
    <a href={cas.url} target="_blank" rel="noopener noreferrer" className="block">{inner}</a>
  ) : inner;
}

/* ─── Main dashboard ─────────────────────────────────────────────────────── */
interface ApiEnvelope<T> { ok: boolean; data?: T; error?: string; }

export function EnforcementDashboard() {
  const [payload, setPayload] = useState<EnforcementHeatmapPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Agency>("sec");

  useEffect(() => {
    fetch("/api/enforcement/heatmap")
      .then((r) => r.json() as Promise<ApiEnvelope<EnforcementHeatmapPayload>>)
      .then((env) => {
        if (!env.ok || !env.data) throw new Error(env.error || "Failed to load");
        setPayload(env.data);
        // default to whichever agency has more data
        if ((env.data.finra.total_cases ?? 0) > (env.data.sec.total_cases ?? 0)) {
          setActiveTab("finra");
        }
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const finraTotal = payload?.finra.total_cases ?? 0;
  const secTotal = payload?.sec.total_cases ?? 0;
  const combined = finraTotal + secTotal;
  const months = payload?.months ?? [];

  const active: AgencyHeatmap | null = payload ? payload[activeTab] : null;
  const ac = COLORS[activeTab];

  /* ── Merged recent cases (interleaved, newest first) */
  const mergedRecent: RecentCase[] = payload
    ? [...payload.finra.recent_cases, ...payload.sec.recent_cases]
        .sort((a, b) => {
          const ta = a.date ? new Date(a.date).getTime() : 0;
          const tb = b.date ? new Date(b.date).getTime() : 0;
          return tb - ta;
        })
        .slice(0, 30)
    : [];

  return (
    <div className="space-y-6">
      {/* ── Combined stat cards ────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard
          label="Total Actions"
          value={loading ? "—" : combined.toLocaleString()}
          color="var(--ink)"
          gradient="linear-gradient(90deg, transparent, rgba(255,255,255,0.2) 50%, transparent)"
        />
        <StatCard
          label="FINRA Actions"
          value={loading ? "—" : finraTotal.toLocaleString()}
          color={COLORS.finra.text}
          gradient={COLORS.finra.gradient}
          dimmed={activeTab === "sec"}
        />
        <StatCard
          label="SEC Actions"
          value={loading ? "—" : secTotal.toLocaleString()}
          color={COLORS.sec.text}
          gradient={COLORS.sec.gradient}
          dimmed={activeTab === "finra"}
        />
        <StatCard
          label={activeTab === "finra" ? "AWC" : "Litigation Releases"}
          value={loading ? "—" : (
            activeTab === "finra"
              ? (payload?.finra.doc_type_counts["AWC"] ?? 0).toLocaleString()
              : (payload?.sec.doc_type_counts["Litigation Release"] ?? secTotal).toLocaleString()
          )}
          color={ac.text}
          gradient={ac.gradient}
        />
      </div>

      {/* ── Agency tabs ────────────────────────────────────────────────── */}
      <div className="flex gap-2">
        {(["finra", "sec"] as Agency[]).map((agency) => {
          const c = COLORS[agency];
          const isActive = activeTab === agency;
          return (
            <button
              key={agency}
              type="button"
              onClick={() => setActiveTab(agency)}
              className="rounded-xl px-4 py-2 text-sm font-semibold transition-all duration-150"
              style={{
                background: isActive ? c.tab : "rgba(9,20,36,0.4)",
                color: isActive ? c.tabText : "var(--ink-faint)",
                border: `1px solid ${isActive ? c.border + "0.35)" : "rgba(255,255,255,0.07)"}`,
                boxShadow: isActive ? `0 0 12px ${c.glow}0.15)` : "none",
              }}
            >
              {agency.toUpperCase()} Enforcement
            </button>
          );
        })}
      </div>

      {/* ── Rule / citation heatmap ──────────────────────────────────────── */}
      <div
        className="rounded-2xl"
        style={{
          border: `1px solid ${ac.border}0.16)`,
          background: "rgba(9,21,34,0.55)",
          backdropFilter: "blur(8px)",
          boxShadow: `0 0 0 1px ${ac.border}0.04), 0 24px 56px rgba(0,0,0,0.35)`,
          transition: "border-color 0.3s, box-shadow 0.3s",
        }}
      >
        <div
          className="flex flex-wrap items-center justify-between gap-3 border-b px-5 py-4"
          style={{ borderColor: `${ac.border}0.10)` }}
        >
          <div>
            <h2 className="text-sm font-semibold text-[color:var(--ink)]">
              {activeTab === "finra" ? "Rule Violation Activity" : "Statute / Rule Citation Activity"}
            </h2>
            <p className="mt-0.5 text-xs text-[color:var(--ink-faint)]">
              {activeTab === "finra"
                ? "FINRA rules cited in AWC, OHO & NAC enforcement documents · last 18 months"
                : "Securities law sections cited in SEC litigation releases · last 18 months"}
            </p>
          </div>
          <Legend agency={activeTab} />
        </div>

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
          {!loading && !error && active && (
            <RuleHeatmap
              agency={activeTab}
              rules={active.rules}
              months={months}
              maxVal={active.max_cell_value}
            />
          )}
        </div>
      </div>

      {/* ── Bottom row ──────────────────────────────────────────────────── */}
      {!loading && !error && payload && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
          {/* Recent actions — combined feed */}
          <div
            className="rounded-2xl border lg:col-span-3"
            style={{ borderColor: "rgba(255,255,255,0.08)", background: "rgba(9,21,34,0.5)" }}
          >
            <div className="flex items-center gap-3 border-b px-4 py-3" style={{ borderColor: "rgba(255,255,255,0.06)" }}>
              <h3 className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
                Recent Enforcement Actions
              </h3>
              <div className="flex gap-1.5">
                <span className="h-2 w-2 rounded-full" style={{ background: COLORS.finra.text }} title="FINRA" />
                <span className="h-2 w-2 rounded-full" style={{ background: COLORS.sec.text }} title="SEC" />
              </div>
            </div>
            <div className="max-h-[420px] space-y-px overflow-y-auto p-2">
              {mergedRecent.length === 0 ? (
                <p className="py-6 text-center text-xs text-[color:var(--ink-faint)]">No documents ingested yet.</p>
              ) : (
                mergedRecent.map((c) => <CaseRow key={`${c.agency}-${c.document_id || c.url}`} c={c} />)
              )}
            </div>
          </div>

          {/* Doc type legend */}
          <div
            className="rounded-2xl border lg:col-span-2"
            style={{ borderColor: "rgba(255,255,255,0.08)", background: "rgba(9,21,34,0.5)" }}
          >
            <div className="border-b px-4 py-3" style={{ borderColor: "rgba(255,255,255,0.06)" }}>
              <h3 className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
                Action Types
              </h3>
            </div>
            <div className="p-4 space-y-5">
              {/* FINRA types */}
              <div>
                <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest" style={{ color: COLORS.finra.text }}>
                  FINRA
                </p>
                {Object.entries(payload.finra.doc_type_counts).sort((a, b) => b[1] - a[1]).map(([type, count]) => {
                  const pct = finraTotal > 0 ? (count / finraTotal) * 100 : 0;
                  return (
                    <div key={type} className="mb-2">
                      <div className="mb-1 flex items-center justify-between">
                        <span className="text-xs font-medium text-[color:var(--ink)]">{type}</span>
                        <span className="text-xs tabular-nums text-[color:var(--ink-faint)]">{count}</span>
                      </div>
                      <div className="h-1.5 w-full overflow-hidden rounded-full" style={{ background: "rgba(9,20,36,0.7)" }}>
                        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: COLORS.finra.text, boxShadow: `0 0 6px ${COLORS.finra.text}55` }} />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* SEC types */}
              <div>
                <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest" style={{ color: COLORS.sec.text }}>
                  SEC
                </p>
                {Object.entries(payload.sec.doc_type_counts).sort((a, b) => b[1] - a[1]).map(([type, count]) => {
                  const pct = secTotal > 0 ? (count / secTotal) * 100 : 0;
                  return (
                    <div key={type} className="mb-2">
                      <div className="mb-1 flex items-center justify-between">
                        <span className="text-xs font-medium text-[color:var(--ink)]">{type}</span>
                        <span className="text-xs tabular-nums text-[color:var(--ink-faint)]">{count}</span>
                      </div>
                      <div className="h-1.5 w-full overflow-hidden rounded-full" style={{ background: "rgba(9,20,36,0.7)" }}>
                        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: COLORS.sec.text, boxShadow: `0 0 6px ${COLORS.sec.text}55` }} />
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="border-t pt-4" style={{ borderColor: "rgba(255,255,255,0.06)" }}>
                <p className="text-[11px] leading-relaxed text-[color:var(--ink-faint)]">
                  <strong className="text-[color:var(--ink)]">AWC</strong> — Acceptance, Waiver &amp; Consent<br />
                  <strong className="text-[color:var(--ink)]">OHO</strong> — Office of Hearing Officers<br />
                  <strong className="text-[color:var(--ink)]">NAC</strong> — National Adjudicatory Council<br />
                  <strong className="text-[color:var(--ink)]">LR</strong> — SEC Litigation Release
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
