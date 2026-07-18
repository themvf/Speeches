"use client";

import { useState } from "react";
import type { CompanyKpi, CompanyKpis, MarketFundamentalsData, MarketKpiData } from "@/lib/server/types";

interface Props {
  data: MarketKpiData | null;
  loading: boolean;
  error: string | null;
}

// Two company-identity hues, distinct from this page's existing semantic
// colors (--accent cyan, --ok green, --danger red/rose, --warn amber all
// already carry meaning elsewhere on /market) and from each other -
// verified colorblind-separable (CVD ΔE 38+, all three checks bar the
// generic mid-tone lightness band, which this app's whole existing
// palette already sits outside of by design - see --accent/--ok/--danger
// in globals.css, all bright-on-near-black).
const COMPANY_COLOR: Record<string, string> = {
  AAPL: "#7c93ff",
  GOOGL: "#f0609e",
};
const FALLBACK_COLORS = ["#7c93ff", "#f0609e", "#41d39d", "#f2ab43"];

function colorFor(ticker: string, index: number): string {
  return COMPANY_COLOR[ticker] ?? FALLBACK_COLORS[index % FALLBACK_COLORS.length]!;
}

function fmt(value: number, unit: CompanyKpi["unit"]): string {
  if (unit === "usd_per_share") return `$${value.toFixed(2)}`;
  if (unit === "percent") return `${value.toFixed(1)}%`;
  if (unit === "count") return value.toLocaleString();
  const abs = Math.abs(value);
  if (abs >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
  return `$${value.toLocaleString()}`;
}

function pctChange(current: number, prior: number | undefined): number | null {
  if (prior == null || prior === 0) return null;
  return ((current - prior) / Math.abs(prior)) * 100;
}

function quarterLabel(periodEnd: string): string {
  const d = new Date(`${periodEnd}T00:00:00Z`);
  return d.toLocaleDateString("en-US", { month: "short", year: "numeric", timeZone: "UTC" });
}

function DeltaTag({ label, value }: { label: string; value: number | null }) {
  if (value == null) {
    return <span className="text-[10px] text-[color:var(--ink-faint)]">{label} —</span>;
  }
  const color = value >= 0 ? "var(--ok)" : "var(--danger)";
  const sign = value >= 0 ? "+" : "−";
  return (
    <span className="text-[10px] tabular-nums">
      <span className="text-[color:var(--ink-faint)]">{label}</span>{" "}
      <span style={{ color }} className="font-semibold">{sign}{Math.abs(value).toFixed(1)}%</span>
    </span>
  );
}

const CHART_W = 300;
const CHART_H = 84;
const PAD_X = 6;
const PAD_Y = 10;

function KpiChart({ kpi, color, strike }: { kpi: CompanyKpi; color: string; strike: number | null }) {
  const values = kpi.series.map((p) => p.value);
  if (strike != null && Number.isFinite(strike)) values.push(strike);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || Math.abs(hi) || 1;
  const y = (v: number) => CHART_H - PAD_Y - ((v - lo) / span) * (CHART_H - 2 * PAD_Y);
  const x = (i: number) => PAD_X + i * ((CHART_W - 2 * PAD_X) / Math.max(1, kpi.series.length - 1));

  const path = kpi.series.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} className="block w-full" role="img" aria-label={`${kpi.label} quarterly trend`}>
      {[1, 2, 3].map((i) => (
        <line
          key={i}
          x1={PAD_X} x2={CHART_W - PAD_X}
          y1={PAD_Y + (i * (CHART_H - 2 * PAD_Y)) / 4}
          y2={PAD_Y + (i * (CHART_H - 2 * PAD_Y)) / 4}
          stroke="var(--line)" strokeWidth={1}
        />
      ))}
      <path d={path} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" />
      {kpi.series.map((p, i) => (
        <circle
          key={p.periodEnd}
          cx={x(i)} cy={y(p.value)}
          r={i === kpi.series.length - 1 ? 4.5 : 3.5}
          fill={p.derived ? "var(--bg-elev-strong)" : color}
          stroke={color}
          strokeWidth={2}
        />
      ))}
      {strike != null && Number.isFinite(strike) && (
        <>
          <line
            x1={PAD_X} x2={CHART_W - PAD_X} y1={y(strike)} y2={y(strike)}
            stroke="var(--ink-faint)" strokeWidth={1.5} strokeDasharray="5 4"
          />
          <text x={CHART_W - PAD_X} y={Math.max(10, y(strike) - 5)} textAnchor="end" fontSize={10} fill="var(--ink-faint)">
            strike {fmt(strike, kpi.unit)}
          </text>
        </>
      )}
    </svg>
  );
}

function KpiCard({ kpi, color }: { kpi: CompanyKpi; color: string }) {
  const [strikeInput, setStrikeInput] = useState("");
  const strike = strikeInput.trim() === "" ? null : Number(strikeInput);
  const series = kpi.series;
  const latest = series[series.length - 1];
  const prior = series.length > 1 ? series[series.length - 2] : undefined;
  const yearAgo = series.length > 4 ? series[series.length - 5] : undefined;

  if (!latest) return null;

  return (
    <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)] p-4">
      <div className="flex items-baseline justify-between gap-2">
        <p className="text-[10px] font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
          {kpi.label}{latest.derived && " · derived"}
        </p>
      </div>
      <div className="mt-1 flex items-baseline gap-2">
        <span className="text-xl font-semibold tabular-nums text-[color:var(--ink)]">{fmt(latest.value, kpi.unit)}</span>
        <span className="text-[10px] text-[color:var(--ink-faint)]">{quarterLabel(latest.periodEnd)}</span>
      </div>
      <div className="mt-1 flex gap-3">
        <DeltaTag label="QoQ" value={pctChange(latest.value, prior?.value)} />
        <DeltaTag label="YoY" value={pctChange(latest.value, yearAgo?.value)} />
      </div>
      <div className="mt-3">
        <KpiChart kpi={kpi} color={color} strike={strike} />
      </div>
      <label className="mt-2 flex items-center gap-2 text-[10px] text-[color:var(--ink-faint)]">
        CBOE strike
        <input
          type="number"
          step="any"
          value={strikeInput}
          onChange={(e) => setStrikeInput(e.target.value)}
          placeholder={kpi.unit === "usd" ? `e.g. ${(latest.value / 1e9).toFixed(1)}e9` : `e.g. ${latest.value.toFixed(2)}`}
          aria-label={`CBOE strike overlay for ${kpi.label}`}
          className="w-24 rounded-md border border-[color:var(--line)] bg-transparent px-1.5 py-0.5 text-[color:var(--ink)] tabular-nums"
        />
      </label>
      <details className="mt-2 text-[11px] text-[color:var(--ink-faint)]">
        <summary className="cursor-pointer">Data ({series.length} quarters)</summary>
        <table className="mt-1.5 w-full tabular-nums">
          <tbody>
            {[...series].reverse().map((p) => (
              <tr key={p.periodEnd} className="border-b border-[color:var(--line-soft)] last:border-0">
                <td className="py-0.5 text-left">{p.periodEnd}</td>
                <td className="py-0.5 text-right">{fmt(p.value, kpi.unit)}</td>
                <td className="py-0.5 pl-2 text-right text-[color:var(--ink-faint)]">{p.derived ? "derived Q4" : "reported"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}

// SEC-13 Tier C: counts arrive in base units (DAP 3.43e9), so compact large
// magnitudes instead of fmt()'s full toLocaleString.
function fmtOperational(value: number, unit: CompanyKpi["unit"]): string {
  if (unit === "count") {
    const abs = Math.abs(value);
    if (abs >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
    if (abs >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
    return value.toLocaleString();
  }
  return fmt(value, unit);
}

function CompanySection({ company, color }: { company: CompanyKpis; color: string }) {
  const latestEnd = company.kpis[0]?.series.at(-1)?.periodEnd;
  return (
    <section>
      <div className="flex items-baseline gap-3 border-b-2 pb-2" style={{ borderColor: color }}>
        <span className="text-xs font-bold tracking-[0.08em]" style={{ color }}>{company.ticker}</span>
        <h3 className="text-lg font-semibold text-[color:var(--ink)]">{company.name}</h3>
        {latestEnd && (
          <span className="ml-auto text-[10px] text-[color:var(--ink-faint)]">latest quarter ends {latestEnd}</span>
        )}
      </div>
      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {company.kpis.map((kpi) => (
          <KpiCard key={kpi.kpiKey} kpi={kpi} color={color} />
        ))}
      </div>
      {company.operational && company.operational.length > 0 && (
        <div className="mt-3 rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)] px-4 py-3">
          <p className="text-[10px] font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
            Operational (from the earnings release)
          </p>
          <div className="mt-2 flex flex-wrap gap-x-6 gap-y-2">
            {company.operational.map((kpi) => (
              <div key={kpi.kpiKey} className="text-xs" title={`Evidence: “${kpi.evidence}”`}>
                <span className="text-[color:var(--ink-faint)]">{kpi.label} </span>
                <span className="font-semibold tabular-nums text-[color:var(--ink)]">
                  {fmtOperational(kpi.value, kpi.unit)}
                </span>
                {kpi.period && <span className="ml-1 text-[10px] text-[color:var(--ink-faint)]">({kpi.period})</span>}
              </div>
            ))}
          </div>
          <p className="mt-1.5 text-[10px] text-[color:var(--ink-faint)]">
            LLM-extracted from the 8-K earnings release (not XBRL-tagged); human-reviewed before display. Hover a value for its verbatim source quote.
          </p>
        </div>
      )}
    </section>
  );
}

// SEC-54: on-demand fundamentals lookup for any ticker in the industry
// universe, rendered with the same KPI cards as the curated companies.
function FundamentalsLookup() {
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<MarketFundamentalsData | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function lookup() {
    const ticker = query.trim().toUpperCase();
    if (!ticker) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`/api/market/fundamentals?ticker=${encodeURIComponent(ticker)}`);
      const env = (await res.json()) as { ok: boolean; data?: MarketFundamentalsData; error?: string };
      if (env.ok && env.data) setResult(env.data);
      else { setResult(null); setError(env.error ?? "Lookup failed"); }
    } catch {
      setError("Network error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)] p-4">
      <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
        Look up any company — quarterly fundamentals on demand (SEC XBRL)
      </p>
      <div className="flex gap-2">
        <input
          value={query}
          onChange={(e) => { setQuery(e.target.value); setError(null); }}
          onKeyDown={(e) => { if (e.key === "Enter") lookup(); }}
          placeholder="Ticker, e.g. CAT or LLY…"
          className="w-44 rounded-lg border border-[color:var(--line)] bg-transparent px-3 py-1.5 text-sm text-[color:var(--ink)]"
        />
        <button
          type="button"
          onClick={lookup}
          disabled={busy || !query.trim()}
          className="rounded-lg border border-[rgba(79,213,255,0.35)] bg-[rgba(79,213,255,0.1)] px-3 py-1.5 text-sm font-medium text-[color:var(--accent)] hover:bg-[rgba(79,213,255,0.16)] disabled:opacity-50"
        >
          {busy ? "Fetching…" : "Fetch"}
        </button>
        {result && (
          <button type="button" onClick={() => setResult(null)} className="text-[10px] text-[color:var(--ink-faint)] underline hover:text-[color:var(--accent)]">
            clear
          </button>
        )}
      </div>
      {error && <p className="mt-2 text-xs text-red-400">{error}</p>}
      {result && (
        <div className="mt-4 space-y-3">
          <CompanySection company={result.company} color="#f2ab43" />
          <p className="text-[10px] text-[color:var(--ink-faint)]">{result.note} · {result.source}</p>
        </div>
      )}
    </div>
  );
}

export function CboeTab({ data, loading, error }: Props) {
  // null = show all companies; a ticker filters to that company. Declared
  // before the early returns so hook order stays stable across renders.
  const [selected, setSelected] = useState<string | null>(null);

  if (loading && !data) {
    return <div className="flex items-center justify-center py-16 text-sm text-[color:var(--ink-faint)]">Loading KPI data…</div>;
  }
  if (error && !data) {
    return <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-400">{error}</div>;
  }
  if (!data) return null;

  // Colors are keyed to each company's position in the full list, so they
  // stay stable when the view is filtered to one ticker.
  const colorByTicker = new Map(data.companies.map((company, i) => [company.ticker, colorFor(company.ticker, i)]));
  const shown = selected ? data.companies.filter((company) => company.ticker === selected) : data.companies;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
            CBOE KPI Options — Company Metrics
          </p>
          <p className="mt-0.5 text-[10px] text-[color:var(--ink-faint)]">
            Research context only — not investment advice.
          </p>
        </div>
        <span className="text-[10px] text-[color:var(--ink-faint)]">Source: {data.source}</span>
      </div>

      {data.warning && (
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-300">
          {data.warning} · snapshot date {data.snapshotDate}
        </div>
      )}

      <p className="max-w-3xl text-xs text-[color:var(--ink-faint)]">
        Each card shows a CBOE-listed KPI&apos;s latest reported value, its change vs. the prior quarter (QoQ)
        and the same quarter a year ago (YoY), and the full quarterly trend — hollow points are derived
        fiscal Q4s (full year minus nine months). Enter a CBOE strike level in any card to draw it against
        the trend and see where the listing sits relative to the recent trajectory.
      </p>

      <FundamentalsLookup />

      {/* Company filter - one row of ticker chips over 22 companies. */}
      <div className="flex flex-wrap gap-1.5">
        <button
          type="button"
          onClick={() => setSelected(null)}
          className={`rounded-lg px-2.5 py-1 text-[11px] font-medium ${selected === null ? "bg-[rgba(79,213,255,0.14)] text-[color:var(--ink)]" : "text-[color:var(--ink-faint)] hover:text-[color:var(--ink-soft)]"}`}
        >
          All {data.companies.length}
        </button>
        {data.companies.map((company) => (
          <button
            key={company.ticker}
            type="button"
            onClick={() => setSelected(company.ticker)}
            className={`rounded-lg px-2.5 py-1 text-[11px] font-semibold ${selected === company.ticker ? "text-[color:var(--ink)]" : "text-[color:var(--ink-faint)] hover:text-[color:var(--ink-soft)]"}`}
            style={selected === company.ticker ? { backgroundColor: `color-mix(in srgb, ${colorByTicker.get(company.ticker)} 18%, transparent)`, color: colorByTicker.get(company.ticker) } : undefined}
          >
            {company.ticker}
          </button>
        ))}
      </div>

      <div className="space-y-8">
        {shown.map((company) => (
          <CompanySection key={company.ticker} company={company} color={colorByTicker.get(company.ticker)!} />
        ))}
      </div>
    </div>
  );
}
