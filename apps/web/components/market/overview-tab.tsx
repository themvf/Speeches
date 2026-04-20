"use client";

import { useState } from "react";
import type {
  CommodityQuote,
  MarketBondsData,
  MarketCommoditiesData,
  MarketIndexQuote,
  MarketOverviewData,
  MarketStatus,
  TreasuryYield,
  VixQuote,
} from "@/lib/server/types";

type IndexRange = "d1" | "w1" | "m1" | "ytd";

const INDEX_RANGES: { id: IndexRange; label: string }[] = [
  { id: "d1",  label: "1D" },
  { id: "w1",  label: "1W" },
  { id: "m1",  label: "1M" },
  { id: "ytd", label: "YTD" },
];

interface TabStateSlice<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

interface Props extends TabStateSlice<MarketOverviewData> {
  commodities: TabStateSlice<MarketCommoditiesData>;
  bonds: TabStateSlice<MarketBondsData>;
}

/* ── Formatters ────────────────────────────────────────────────────────── */

function fmtPrice(n: number): string {
  return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtPct(n: number, showSign = true): string {
  const sign = showSign && n >= 0 ? "+" : "";
  return `${sign}${n.toFixed(2)}%`;
}

function fmtChange(n: number): string {
  return `${n >= 0 ? "+" : ""}${n.toFixed(2)}`;
}

/* ── Shared atoms ──────────────────────────────────────────────────────── */

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

function RangeSelector({ value, onChange }: { value: IndexRange; onChange: (r: IndexRange) => void }) {
  return (
    <div className="flex items-center gap-0.5 rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)] p-1">
      {INDEX_RANGES.map(({ id, label }) => (
        <button
          key={id}
          type="button"
          onClick={() => onChange(id)}
          className={`rounded-lg px-3 py-1 text-xs font-medium transition-colors ${
            value === id
              ? "bg-[color:rgba(79,213,255,0.18)] text-[color:var(--ink)]"
              : "text-[color:var(--ink-faint)] hover:text-[color:var(--ink)]"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function SectionHeader({ title, count }: { title: string; count?: number }) {
  return (
    <p className="mb-3 text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
      {title}
      {count !== undefined && <span className="ml-2 font-normal">{count}</span>}
    </p>
  );
}

/* ── MiniSparkline ─────────────────────────────────────────────────────── */

function MiniSparkline({ prices, up }: { prices: number[]; up: boolean }) {
  if (prices.length < 2) return <div className="h-10 w-20 opacity-30 bg-[color:var(--line)] rounded" />;

  const W = 80; const H = 40;
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const range = max - min || 1;
  const xs = prices.map((_, i) => (i / (prices.length - 1)) * W);
  const ys = prices.map((p) => H - ((p - min) / range) * (H - 4) - 2);
  const color = up ? "#41d39d" : "#f87171";

  return (
    <svg width={W} height={H} className="shrink-0 overflow-visible" aria-hidden>
      <defs>
        <linearGradient id={`sg-${up}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      {/* Fill area */}
      <polygon
        points={[
          ...xs.map((x, i) => `${x.toFixed(1)},${ys[i].toFixed(1)}`),
          `${W},${H}`, `0,${H}`,
        ].join(" ")}
        fill={`url(#sg-${up})`}
      />
      {/* Line */}
      <polyline
        points={xs.map((x, i) => `${x.toFixed(1)},${ys[i].toFixed(1)}`).join(" ")}
        fill="none" stroke={color} strokeWidth="1.5"
        strokeLinejoin="round" strokeLinecap="round" opacity="0.85"
      />
      <circle cx={xs[xs.length - 1]} cy={ys[ys.length - 1]} r="2" fill={color} />
    </svg>
  );
}

/* ── IndexCard ─────────────────────────────────────────────────────────── */

function IndexCard({ q, range }: { q: MarketIndexQuote; range: IndexRange }) {
  const displayPct = q.pcts[range] ?? q.pct;
  const color = displayPct >= 0 ? "#41d39d" : "#f87171";

  return (
    <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.5)] p-4 flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">{q.name}</span>
        <StatusPill status={q.status} />
      </div>
      <div className="flex items-end justify-between gap-2">
        <div>
          <div className="text-2xl font-bold tabular-nums text-[color:var(--ink)] leading-none">
            {fmtPrice(q.price)}
          </div>
          <div className="mt-1 flex items-center gap-2 text-xs tabular-nums" style={{ color }}>
            {range === "d1" && <span>{fmtChange(q.change)}</span>}
            <span className="font-semibold">{fmtPct(displayPct)}</span>
          </div>
        </div>
        {q.sparkline.length >= 2 && (
          <MiniSparkline prices={q.sparkline} up={displayPct >= 0} />
        )}
      </div>
    </div>
  );
}

/* ── VixMeter ──────────────────────────────────────────────────────────── */

function VixMeter({ vix }: { vix: VixQuote }) {
  const labelColor: Record<string, string> = {
    GREED: "#41d39d", CALM: "#4fd5ff", CONCERN: "#f2ab43", PANIC: "#f87171",
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
      <div className="relative h-3 w-full rounded-full overflow-hidden"
        style={{ background: "linear-gradient(to right, #41d39d 0%, #4fd5ff 30%, #f2ab43 65%, #f87171 100%)" }}>
        <div
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-4 w-1.5 rounded-full bg-white shadow-md"
          style={{ left: `${vix.gradientPct}%` }}
        />
      </div>
      <div className="flex justify-between mt-1 text-[10px] text-[color:var(--ink-faint)]">
        <span>GREED</span><span>CALM</span><span>PANIC</span>
      </div>
    </div>
  );
}

/* ── GlobalIndexTable ──────────────────────────────────────────────────── */

function GlobalIndexTable({ indices }: { indices: MarketIndexQuote[] }) {
  const maxAbs = Math.max(...indices.map((q) => Math.abs(q.pct)), 1);
  return (
    <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
      <div className="border-b border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)] px-4 py-2">
        <span className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
          Global Indexes <span className="font-normal">{indices.length}</span>
        </span>
      </div>
      <table className="w-full text-xs">
        <tbody>
          {indices.map((q) => {
            const color = q.up ? "#41d39d" : "#f87171";
            const barW = Math.round((Math.abs(q.pct) / maxAbs) * 60);
            return (
              <tr key={q.symbol} className="border-b border-[color:var(--line)] last:border-0 hover:bg-[color:rgba(79,213,255,0.03)]">
                <td className="px-4 py-2.5 font-semibold text-[color:var(--ink)] w-28">{q.name}</td>
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

/* ── CommoditiesTable ──────────────────────────────────────────────────── */

function CommoditiesTable({ commodities }: { commodities: CommodityQuote[] }) {
  const maxAbs = Math.max(...commodities.map((c) => Math.abs(c.pct)), 1);
  const groups: { label: string; key: string }[] = [
    { label: "Metals",      key: "metals"      },
    { label: "Energy",      key: "energy"      },
    { label: "Agriculture", key: "agriculture" },
  ];

  return (
    <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
      {groups.map(({ label, key }) => {
        const items = commodities.filter((c) => c.category === key);
        if (items.length === 0) return null;
        return (
          <div key={key}>
            <div className="border-b border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)] px-4 py-2">
              <span className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
                {label} <span className="font-normal">{items.length}</span>
              </span>
            </div>
            <table className="w-full text-xs">
              <tbody>
                {items.map((c) => {
                  const color = c.up ? "#41d39d" : "#f87171";
                  const barW = Math.round((Math.abs(c.pct) / maxAbs) * 60);
                  return (
                    <tr key={c.symbol} className="border-b border-[color:var(--line)] last:border-0 hover:bg-[color:rgba(79,213,255,0.03)]">
                      <td className="px-4 py-2.5 font-semibold text-[color:var(--ink)] w-32">{c.name}</td>
                      <td className="px-2 py-2.5 tabular-nums text-right text-[color:var(--ink-faint)] text-[10px]">{c.symbol}</td>
                      <td className="px-2 py-2.5 tabular-nums text-right text-[color:var(--ink)]">{fmtPrice(c.price)}</td>
                      <td className="px-2 py-2.5 tabular-nums text-right" style={{ color }}>{fmtChange(c.change)}</td>
                      <td className="px-2 py-2.5 tabular-nums text-right font-semibold" style={{ color }}>{fmtPct(c.pct)}</td>
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
      })}
    </div>
  );
}

/* ── BondsTable ────────────────────────────────────────────────────────── */

function BondsTable({ yields, dxy }: { yields: TreasuryYield[]; dxy: MarketBondsData["dxy"] }) {
  const allPcts = [...yields.map((y) => Math.abs(y.pct)), Math.abs(dxy?.pct ?? 0)].filter(Boolean);
  const maxAbs = Math.max(...allPcts, 1);

  const rows = [
    ...yields.map((y) => ({
      name: y.label,
      value: y.rate.toFixed(2),
      change: fmtChange(y.change),
      pct: fmtPct(y.pct),
      up: y.up,
      absPct: Math.abs(y.pct),
    })),
    ...(dxy ? [{
      name: "USD Index (UUP)",
      value: fmtPrice(dxy.price),
      change: fmtChange(dxy.change),
      pct: fmtPct(dxy.pct),
      up: dxy.up,
      absPct: Math.abs(dxy.pct),
    }] : []),
  ];

  return (
    <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
      <div className="border-b border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)] px-4 py-2">
        <span className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
          Bonds &amp; Rates <span className="font-normal">{rows.length}</span>
        </span>
      </div>
      <table className="w-full text-xs">
        <tbody>
          {rows.map((row) => {
            const color = row.up ? "#41d39d" : "#f87171";
            const barW = Math.round((row.absPct / maxAbs) * 60);
            return (
              <tr key={row.name} className="border-b border-[color:var(--line)] last:border-0 hover:bg-[color:rgba(79,213,255,0.03)]">
                <td className="px-4 py-2.5 font-semibold text-[color:var(--ink)] w-36">{row.name}</td>
                <td className="px-2 py-2.5 tabular-nums text-right text-[color:var(--ink)]">{row.value}</td>
                <td className="px-2 py-2.5 tabular-nums text-right" style={{ color }}>{row.change}</td>
                <td className="px-2 py-2.5 tabular-nums text-right font-semibold" style={{ color }}>{row.pct}</td>
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

export function OverviewTab({ data, loading, error, commodities, bonds }: Props) {
  const [indexRange, setIndexRange] = useState<IndexRange>("d1");

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center py-16 text-sm text-[color:var(--ink-faint)]">
        Loading market data…
      </div>
    );
  }
  if (error && !data) {
    return (
      <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-400">{error}</div>
    );
  }
  if (!data) return null;

  return (
    <div className="space-y-8">
      {/* ── US Indices ──────────────────────────────────────────────── */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <SectionHeader title="US Indices" />
          <RangeSelector value={indexRange} onChange={setIndexRange} />
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {data.indices.map((q) => <IndexCard key={q.symbol} q={q} range={indexRange} />)}
        </div>
      </div>

      {/* ── VIX ─────────────────────────────────────────────────────── */}
      {data.vix && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <VixMeter vix={data.vix} />
          <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.5)] p-5 flex flex-col justify-center gap-2">
            <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Volatility Index</p>
            <p className="text-xs text-[color:var(--ink-faint)] leading-relaxed">
              VIX measures expected 30-day market volatility. Below 15 signals low fear; above 30 indicates elevated stress. Values above 40 are historically rare and mark extreme fear events.
            </p>
          </div>
        </div>
      )}

      {/* ── Commodities ─────────────────────────────────────────────── */}
      <div>
        <SectionHeader title="Commodities & Energy" count={commodities.data?.commodities.length} />
        {commodities.loading && !commodities.data ? (
          <div className="py-6 text-center text-xs text-[color:var(--ink-faint)]">Loading…</div>
        ) : commodities.data?.commodities.length ? (
          <CommoditiesTable commodities={commodities.data.commodities} />
        ) : null}
      </div>

      {/* ── Bonds & Rates ────────────────────────────────────────────── */}
      <div>
        <SectionHeader title="Bonds & Rates" />
        {bonds.loading && !bonds.data ? (
          <div className="py-6 text-center text-xs text-[color:var(--ink-faint)]">Loading…</div>
        ) : bonds.data && (bonds.data.yields.length > 0 || bonds.data.dxy) ? (
          <BondsTable yields={bonds.data.yields} dxy={bonds.data.dxy} />
        ) : null}
      </div>

      {/* ── Global Indices ───────────────────────────────────────────── */}
      {data.globalIndices.length > 0 && (
        <div>
          <SectionHeader title="Global Indices" count={data.globalIndices.length} />
          <GlobalIndexTable indices={data.globalIndices} />
        </div>
      )}

      <p className="text-right text-[11px] text-[color:var(--ink-faint)]">
        Updated {new Date(data.generatedAt).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })}
        {" · "}Commodity prices are ETF proxies. Bond yields via US Treasury.
      </p>
    </div>
  );
}
