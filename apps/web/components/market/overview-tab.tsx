"use client";

import type { MarketIndexQuote, MarketOverviewData, MarketStatus, VixQuote } from "@/lib/server/types";

interface Props {
  data: MarketOverviewData | null;
  loading: boolean;
  error: string | null;
}

function fmtPrice(n: number): string {
  if (n >= 1000) return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtChange(n: number): string {
  const sign = n >= 0 ? "+" : "";
  return `${sign}${n.toFixed(2)}`;
}

function fmtPct(n: number): string {
  const sign = n >= 0 ? "+" : "";
  return `${sign}${n.toFixed(2)}%`;
}

function fmtLarge(n: number): string {
  if (n >= 1e12) return `$${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9)  return `$${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6)  return `$${(n / 1e6).toFixed(1)}M`;
  return `$${n.toLocaleString()}`;
}

const INDEX_NAMES: Record<string, string> = {
  SPY: "S&P 500",
  DIA: "Dow Jones",
  QQQ: "NASDAQ",
  IWM: "Russell 2000",
};

/* ── StatusPill ────────────────────────────────────────────────────────── */

function StatusPill({ status }: { status: MarketStatus }) {
  const cfg: Record<MarketStatus, { label: string; cls: string }> = {
    OPEN:   { label: "OPEN",   cls: "border-[color:rgba(65,211,157,0.3)] bg-[color:rgba(65,211,157,0.12)] text-[#41d39d]" },
    CLOSED: { label: "CLOSED", cls: "border-[color:rgba(107,114,128,0.3)] bg-[color:rgba(107,114,128,0.1)] text-[color:var(--ink-faint)]" },
    PRE:    { label: "PRE",    cls: "border-[color:rgba(242,171,67,0.3)] bg-[color:rgba(242,171,67,0.1)] text-[#f2ab43]" },
    AFTER:  { label: "AFTER",  cls: "border-[color:rgba(242,171,67,0.3)] bg-[color:rgba(242,171,67,0.1)] text-[#f2ab43]" },
  };
  const { label, cls } = cfg[status] ?? cfg.CLOSED;
  return (
    <span className={`rounded border px-1.5 py-0.5 text-[9px] font-bold tracking-[0.08em] ${cls}`}>
      {label}
    </span>
  );
}

/* ── IndexCard ─────────────────────────────────────────────────────────── */

function IndexCard({ q }: { q: MarketIndexQuote }) {
  const color = q.up ? "#41d39d" : "#f87171";
  const name = INDEX_NAMES[q.symbol] ?? q.name;

  return (
    <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.5)] p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">{name}</span>
        <StatusPill status={q.status} />
      </div>

      <div>
        <div className="text-2xl font-bold tabular-nums text-[color:var(--ink)] leading-none">
          {fmtPrice(q.price)}
        </div>
        <div className="mt-1 flex items-center gap-2 text-xs tabular-nums" style={{ color }}>
          <span>{fmtChange(q.change)}</span>
          <span className="font-semibold">{fmtPct(q.pct)}</span>
        </div>
      </div>

      {/* Trend bar placeholder */}
      <div className="h-1 w-full overflow-hidden rounded-full bg-[color:rgba(255,255,255,0.05)]">
        <div
          className="h-full rounded-full"
          style={{
            width: `${Math.min(100, Math.abs(q.pct) * 8 + 40)}%`,
            backgroundColor: color,
            opacity: 0.6,
          }}
        />
      </div>
    </div>
  );
}

/* ── VixMeter ──────────────────────────────────────────────────────────── */

function VixMeter({ vix }: { vix: VixQuote }) {
  const labelColor: Record<string, string> = {
    GREED:   "#41d39d",
    CALM:    "#4fd5ff",
    CONCERN: "#f2ab43",
    PANIC:   "#f87171",
  };

  return (
    <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.5)] p-5">
      <div className="flex items-center justify-between mb-4">
        <span className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
          VIX — Fear &amp; Greed
        </span>
        <span className="text-xs font-bold tracking-[0.1em]" style={{ color: labelColor[vix.label] }}>
          {vix.label}
        </span>
      </div>

      <div className="flex items-baseline gap-3 mb-5">
        <span className="text-3xl font-bold tabular-nums text-[color:var(--ink)]">{vix.value.toFixed(2)}</span>
        <span className="text-sm tabular-nums" style={{ color: vix.pct >= 0 ? "#f87171" : "#41d39d" }}>
          {fmtPct(vix.pct)}
        </span>
      </div>

      {/* Gradient bar */}
      <div className="relative h-3 w-full rounded-full overflow-hidden"
        style={{ background: "linear-gradient(to right, #41d39d 0%, #4fd5ff 30%, #f2ab43 65%, #f87171 100%)" }}
      >
        <div
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-4 w-1.5 rounded-full bg-white shadow-md"
          style={{ left: `${vix.gradientPct}%` }}
        />
      </div>
      <div className="flex justify-between mt-1 text-[10px] text-[color:var(--ink-faint)]">
        <span>GREED</span>
        <span>CALM</span>
        <span>PANIC</span>
      </div>
    </div>
  );
}

/* ── GlobalIndexTable ──────────────────────────────────────────────────── */

function GlobalIndexTable({ indices }: { indices: MarketIndexQuote[] }) {
  const maxAbs = Math.max(...indices.map((q) => Math.abs(q.pct)), 1);

  return (
    <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)] overflow-hidden">
      <div className="border-b border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)] px-4 py-2">
        <span className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
          Global Indexes
        </span>
      </div>
      <table className="w-full text-xs">
        <tbody>
          {indices.map((q) => {
            const color = q.up ? "#41d39d" : "#f87171";
            const barW = Math.round((Math.abs(q.pct) / maxAbs) * 60);
            return (
              <tr key={q.symbol} className="border-b border-[color:var(--line)] last:border-0 hover:bg-[color:rgba(79,213,255,0.03)]">
                <td className="px-4 py-2.5 font-semibold text-[color:var(--ink)] w-24">{q.name}</td>
                <td className="px-2 py-2.5 tabular-nums text-right text-[color:var(--ink)]">{fmtPrice(q.price)}</td>
                <td className="px-2 py-2.5 tabular-nums text-right" style={{ color }}>{fmtChange(q.change)}</td>
                <td className="px-2 py-2.5 tabular-nums text-right font-semibold" style={{ color }}>{fmtPct(q.pct)}</td>
                <td className="px-4 py-2.5 w-20">
                  <div className="flex justify-end">
                    <div className="h-3 rounded-sm" style={{ width: barW, backgroundColor: color, opacity: 0.7 }} />
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ── OverviewTab ───────────────────────────────────────────────────────── */

export function OverviewTab({ data, loading, error }: Props) {
  if (loading && !data) {
    return (
      <div className="flex items-center justify-center py-16 text-sm text-[color:var(--ink-faint)]">
        Loading market data…
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-400">
        {error}
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="space-y-6">
      {/* US Index grid */}
      <div>
        <p className="mb-3 text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
          Market Overview
        </p>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {data.indices.map((q) => <IndexCard key={q.symbol} q={q} />)}
        </div>
      </div>

      {/* VIX meter */}
      {data.vix && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <VixMeter vix={data.vix} />
          <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.5)] p-5 flex flex-col justify-center gap-1">
            <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Volatility Index</p>
            <p className="text-xs text-[color:var(--ink-faint)] leading-relaxed mt-2">
              The VIX measures market expected volatility over the next 30 days.
              Below 15 signals low fear (greed), above 30 indicates elevated stress.
            </p>
          </div>
        </div>
      )}

      {/* Global indices */}
      {data.globalIndices.length > 0 && (
        <GlobalIndexTable indices={data.globalIndices} />
      )}

      {/* Timestamp */}
      <p className="text-right text-[11px] text-[color:var(--ink-faint)]">
        Updated {new Date(data.generatedAt).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })}
      </p>
    </div>
  );
}
