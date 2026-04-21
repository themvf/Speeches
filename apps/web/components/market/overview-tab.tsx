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
import { InlineChart } from "./price-chart";

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

/* ── Formatters ─────────────────────────────────────────────────────── */

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

/* ── Shared atoms ───────────────────────────────────────────────────── */

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

function BoxHeader({ title, count }: { title: string; count?: number }) {
  return (
    <div className="border-b border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)] px-4 py-2.5">
      <span className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
        {title}
        {count !== undefined && <span className="ml-2 font-normal opacity-60">{count}</span>}
      </span>
    </div>
  );
}

/* ── MiniSparkline ──────────────────────────────────────────────────── */

function MiniSparkline({ prices, up }: { prices: number[]; up: boolean }) {
  if (prices.length < 2) return <div className="h-10 w-20 opacity-20 bg-[color:var(--line)] rounded" />;
  const W = 80; const H = 40;
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const range = max - min || 1;
  const xs = prices.map((_, i) => (i / (prices.length - 1)) * W);
  const ys = prices.map((p) => H - ((p - min) / range) * (H - 4) - 2);
  const color = up ? "#41d39d" : "#f87171";
  const id = `sg-${up ? "up" : "dn"}`;
  return (
    <svg width={W} height={H} className="shrink-0 overflow-visible" aria-hidden>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <polygon points={[...xs.map((x, i) => `${x.toFixed(1)},${ys[i].toFixed(1)}`), `${W},${H}`, `0,${H}`].join(" ")} fill={`url(#${id})`} />
      <polyline points={xs.map((x, i) => `${x.toFixed(1)},${ys[i].toFixed(1)}`).join(" ")} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" opacity="0.85" />
      <circle cx={xs[xs.length - 1]} cy={ys[ys.length - 1]} r="2" fill={color} />
    </svg>
  );
}

/* ── IndexCard ──────────────────────────────────────────────────────── */

function IndexCard({
  q, range, expanded, onToggle,
}: {
  q: MarketIndexQuote; range: IndexRange; expanded: boolean; onToggle: () => void;
}) {
  const displayPct = q.pcts[range] ?? q.pct;
  const color = displayPct >= 0 ? "#41d39d" : "#f87171";
  return (
    <div
      className={`rounded-xl border bg-[color:rgba(9,21,34,0.5)] p-4 flex flex-col gap-2 cursor-pointer transition-colors select-none ${
        expanded
          ? "border-[color:rgba(79,213,255,0.35)] bg-[color:rgba(79,213,255,0.06)]"
          : "border-[color:var(--line)] hover:border-[color:rgba(79,213,255,0.2)]"
      }`}
      onClick={onToggle}
      role="button"
      aria-expanded={expanded}
    >
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
        {q.sparkline.length >= 2 && <MiniSparkline prices={q.sparkline} up={displayPct >= 0} />}
      </div>
    </div>
  );
}

/* ── FearGauge ──────────────────────────────────────────────────────── */

function FearGauge({ vix }: { vix: VixQuote }) {
  const cx = 100, cy = 90, r = 72, sw = 13, nl = 56;
  const rad = (d: number) => (d * Math.PI) / 180;
  const arcD = (from: number, to: number) => {
    const sx = (cx + r * Math.cos(rad(from))).toFixed(2);
    const sy = (cy - r * Math.sin(rad(from))).toFixed(2);
    const ex = (cx + r * Math.cos(rad(to))).toFixed(2);
    const ey = (cy - r * Math.sin(rad(to))).toFixed(2);
    return `M ${sx} ${sy} A ${r} ${r} 0 0 1 ${ex} ${ey}`;
  };
  const needleDeg = 180 - (vix.gradientPct / 100) * 180;
  const nx = (cx + nl * Math.cos(rad(needleDeg))).toFixed(2);
  const ny = (cy - nl * Math.sin(rad(needleDeg))).toFixed(2);
  const lc = ({ GREED: "#41d39d", CALM: "#4fd5ff", CONCERN: "#f2ab43", PANIC: "#f87171" } as Record<string, string>)[vix.label] ?? "#fff";
  const SEGS = [
    { from: 180, to: 135, color: "#41d39d" },
    { from: 135, to: 90,  color: "#4fd5ff" },
    { from: 90,  to: 45,  color: "#f2ab43" },
    { from: 45,  to: 0,   color: "#f87171" },
  ];
  return (
    <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)] flex flex-col">
      <BoxHeader title="Fear & Greed — VIX" />
      <div className="flex flex-col items-center px-4 pt-3 pb-4 flex-1">
        <svg viewBox="0 0 200 100" className="w-full max-w-[230px]" aria-hidden>
          <path d={arcD(180, 0)} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth={sw} strokeLinecap="round" />
          {SEGS.map((s) => (
            <path key={s.from} d={arcD(s.from, s.to)} fill="none" stroke={s.color} strokeWidth={sw} strokeLinecap="butt" opacity={0.8} />
          ))}
          <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="white" strokeWidth="2.5" strokeLinecap="round" />
          <circle cx={cx} cy={cy} r="5" fill={lc} />
          <text x="16"  y={cy + 14} textAnchor="middle" fontSize="8" fill="rgba(255,255,255,0.35)" fontFamily="system-ui">GREED</text>
          <text x="184" y={cy + 14} textAnchor="middle" fontSize="8" fill="rgba(255,255,255,0.35)" fontFamily="system-ui">PANIC</text>
        </svg>
        <div className="text-center -mt-1">
          <div className="text-3xl font-bold tabular-nums text-white leading-none">{vix.value.toFixed(2)}</div>
          <div className="text-[11px] font-bold tracking-widest mt-1" style={{ color: lc }}>{vix.label}</div>
          <div className="text-xs tabular-nums mt-1">
            <span style={{ color: vix.pct >= 0 ? "#f87171" : "#41d39d" }}>
              {vix.change >= 0 ? "+" : ""}{vix.change.toFixed(2)} ({fmtPct(vix.pct)})
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── CommodityBox ───────────────────────────────────────────────────── */

function CommodityBox({ title, items }: { title: string; items: CommodityQuote[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const maxAbs = Math.max(...items.map((c) => Math.abs(c.pct)), 1);

  return (
    <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
      <BoxHeader title={title} />
      <table className="w-full text-xs">
        <tbody>
          {items.map((c) => {
            const color = c.up ? "#41d39d" : "#f87171";
            const barW  = Math.round((Math.abs(c.pct) / maxAbs) * 48);
            const isExp = expanded === c.symbol;
            return (
              <>
                <tr
                  key={c.symbol}
                  className={`border-b border-[color:var(--line)] cursor-pointer transition-colors ${
                    isExp ? "bg-[color:rgba(79,213,255,0.06)]" : "hover:bg-[color:rgba(79,213,255,0.03)]"
                  }`}
                  onClick={() => setExpanded(isExp ? null : c.symbol)}
                >
                  <td className="pl-4 pr-2 py-2.5 font-semibold text-[color:var(--ink)]">
                    <span className="mr-1.5 text-[color:var(--ink-faint)]">{isExp ? "▾" : "▸"}</span>
                    {c.name}
                  </td>
                  <td className="px-2 py-2.5 tabular-nums text-right text-[color:var(--ink)]">${fmtPrice(c.price)}</td>
                  <td className="px-2 py-2.5 tabular-nums text-right font-semibold" style={{ color }}>{fmtPct(c.pct)}</td>
                  <td className="pl-2 pr-4 py-2.5 w-14">
                    <div className="flex justify-end">
                      <div className="h-2.5 rounded-sm" style={{ width: barW, backgroundColor: color, opacity: 0.7 }} />
                    </div>
                  </td>
                </tr>
                {isExp && (
                  <tr key={`${c.symbol}-chart`}>
                    <td colSpan={4} className="p-4 bg-[color:rgba(9,21,34,0.3)] border-b border-[color:var(--line)]">
                      <InlineChart symbol={c.symbol} type="yahoo" name={c.name} up={c.up} />
                    </td>
                  </tr>
                )}
              </>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ── BondsBox ───────────────────────────────────────────────────────── */

function BondsBox({ yields, dxy }: { yields: TreasuryYield[]; dxy: MarketBondsData["dxy"] }) {
  const allPcts = [...yields.map((y) => Math.abs(y.pct)), Math.abs(dxy?.pct ?? 0)].filter(Boolean);
  const maxAbs = Math.max(...allPcts, 1);
  const rows = [
    ...yields.map((y) => ({
      key: y.label, name: y.label, value: y.rate.toFixed(2) + "%",
      pct: fmtPct(y.pct), up: y.up, absPct: Math.abs(y.pct), symbol: null,
    })),
    ...(dxy ? [{
      key: "dxy", name: "USD (UUP)", value: "$" + fmtPrice(dxy.price),
      pct: fmtPct(dxy.pct), up: dxy.up, absPct: Math.abs(dxy.pct), symbol: "UUP",
    }] : []),
  ];
  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
      <BoxHeader title="Bonds & Rates" />
      <table className="w-full text-xs">
        <tbody>
          {rows.map((row) => {
            const color = row.up ? "#41d39d" : "#f87171";
            const barW  = Math.round((row.absPct / maxAbs) * 48);
            const isExp = expanded === row.key;
            const clickable = !!row.symbol;
            return (
              <>
                <tr
                  key={row.key}
                  className={`border-b border-[color:var(--line)] last:border-0 transition-colors ${
                    clickable ? "cursor-pointer" : ""
                  } ${isExp ? "bg-[color:rgba(79,213,255,0.06)]" : clickable ? "hover:bg-[color:rgba(79,213,255,0.03)]" : ""}`}
                  onClick={() => clickable && setExpanded(isExp ? null : row.key)}
                >
                  <td className="pl-4 pr-2 py-2.5 font-semibold text-[color:var(--ink)]">
                    {clickable && <span className="mr-1.5 text-[color:var(--ink-faint)]">{isExp ? "▾" : "▸"}</span>}
                    {row.name}
                  </td>
                  <td className="px-2 py-2.5 tabular-nums text-right text-[color:var(--ink)]">{row.value}</td>
                  <td className="px-2 py-2.5 tabular-nums text-right font-semibold" style={{ color }}>{row.pct}</td>
                  <td className="pl-2 pr-4 py-2.5 w-14">
                    <div className="flex justify-end">
                      <div className="h-2.5 rounded-sm" style={{ width: barW, backgroundColor: color, opacity: 0.7 }} />
                    </div>
                  </td>
                </tr>
                {isExp && row.symbol && (
                  <tr key={`${row.key}-chart`}>
                    <td colSpan={4} className="p-4 bg-[color:rgba(9,21,34,0.3)] border-b border-[color:var(--line)]">
                      <InlineChart symbol={row.symbol} type="yahoo" name={row.name} up={row.up} />
                    </td>
                  </tr>
                )}
              </>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ── GlobalIndicesBox ───────────────────────────────────────────────── */

function GlobalIndicesBox({ indices }: { indices: MarketIndexQuote[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const maxAbs = Math.max(...indices.map((q) => Math.abs(q.pct)), 1);
  return (
    <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
      <BoxHeader title="Global Indices" count={indices.length} />
      <table className="w-full text-xs">
        <tbody>
          {indices.map((q) => {
            const color = q.up ? "#41d39d" : "#f87171";
            const barW  = Math.round((Math.abs(q.pct) / maxAbs) * 48);
            const isExp = expanded === q.symbol;
            return (
              <>
                <tr
                  key={q.symbol}
                  className={`border-b border-[color:var(--line)] last:border-0 cursor-pointer transition-colors ${
                    isExp ? "bg-[color:rgba(79,213,255,0.06)]" : "hover:bg-[color:rgba(79,213,255,0.03)]"
                  }`}
                  onClick={() => setExpanded(isExp ? null : q.symbol)}
                >
                  <td className="pl-4 pr-2 py-2.5 font-semibold text-[color:var(--ink)] w-28">
                    <span className="mr-1.5 text-[color:var(--ink-faint)]">{isExp ? "▾" : "▸"}</span>
                    {q.name}
                  </td>
                  <td className="px-2 py-2.5 tabular-nums text-right text-[color:var(--ink)]">{fmtPrice(q.price)}</td>
                  <td className="px-2 py-2.5 tabular-nums text-right font-semibold" style={{ color }}>{fmtPct(q.pct)}</td>
                  <td className="pl-2 pr-4 py-2.5 w-14">
                    <div className="flex justify-end">
                      <div className="h-2.5 rounded-sm" style={{ width: barW, backgroundColor: color, opacity: 0.7 }} />
                    </div>
                  </td>
                </tr>
                {isExp && (
                  <tr key={`${q.symbol}-chart`}>
                    <td colSpan={4} className="p-4 bg-[color:rgba(9,21,34,0.3)] border-b border-[color:var(--line)]">
                      <InlineChart symbol={q.symbol} type="yahoo" name={q.name} up={q.up} />
                    </td>
                  </tr>
                )}
              </>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ── OverviewTab ────────────────────────────────────────────────────── */

export function OverviewTab({ data, loading, error, commodities, bonds }: Props) {
  const [indexRange, setIndexRange] = useState<IndexRange>("d1");
  const [expandedIndex, setExpandedIndex] = useState<string | null>(null);

  if (loading && !data) {
    return <div className="flex items-center justify-center py-16 text-sm text-[color:var(--ink-faint)]">Loading market data…</div>;
  }
  if (error && !data) {
    return <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-400">{error}</div>;
  }
  if (!data) return null;

  const metals      = commodities.data?.commodities.filter((c) => c.category === "metals")      ?? [];
  const energy      = commodities.data?.commodities.filter((c) => c.category === "energy")      ?? [];
  const agriculture = commodities.data?.commodities.filter((c) => c.category === "agriculture") ?? [];
  const hasBonds    = bonds.data && (bonds.data.yields.length > 0 || bonds.data.dxy);
  const expandedQ   = data.indices.find((q) => q.symbol === expandedIndex);

  return (
    <div className="space-y-5">

      {/* ── US Indices ──────────────────────────────────────────── */}
      <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
        <div className="flex items-center justify-between border-b border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)] px-4 py-2.5">
          <span className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">US Indices</span>
          <RangeSelector value={indexRange} onChange={setIndexRange} />
        </div>
        <div className="grid grid-cols-2 gap-3 p-3 sm:grid-cols-4">
          {data.indices.map((q) => (
            <IndexCard
              key={q.symbol} q={q} range={indexRange}
              expanded={expandedIndex === q.symbol}
              onToggle={() => setExpandedIndex((prev) => prev === q.symbol ? null : q.symbol)}
            />
          ))}
        </div>
        {expandedQ && (
          <div className="border-t border-[color:var(--line)] bg-[color:rgba(9,21,34,0.3)] px-4 pt-3 pb-4">
            <InlineChart symbol={expandedIndex!} type="yahoo" name={expandedQ.name} up={expandedQ.up} />
          </div>
        )}
      </div>

      {/* ── Fear Gauge + Global Indices ─────────────────────────── */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {data.vix && <FearGauge vix={data.vix} />}
        {data.globalIndices.length > 0 && <GlobalIndicesBox indices={data.globalIndices} />}
      </div>

      {/* ── 2×2: Metals | Energy / Agriculture | Bonds ──────────── */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {metals.length > 0      && <CommodityBox title="Metals"      items={metals} />}
        {energy.length > 0      && <CommodityBox title="Energy"      items={energy} />}
        {agriculture.length > 0 && <CommodityBox title="Agriculture" items={agriculture} />}
        {hasBonds               && <BondsBox yields={bonds.data!.yields} dxy={bonds.data!.dxy} />}
      </div>

      {commodities.loading && !commodities.data && (
        <div className="py-4 text-center text-xs text-[color:var(--ink-faint)]">Loading commodities…</div>
      )}
      {bonds.loading && !bonds.data && (
        <div className="py-4 text-center text-xs text-[color:var(--ink-faint)]">Loading bonds…</div>
      )}

      <p className="text-right text-[11px] text-[color:var(--ink-faint)]">
        Updated {new Date(data.generatedAt).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })}
        {" · "}Commodity prices are futures · Bond yields via US Treasury
      </p>
    </div>
  );
}
