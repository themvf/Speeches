"use client";

import { useState } from "react";
import type { MarketSectorsData, SectorData, SectorStock } from "@/lib/server/types";
import { InlineChart } from "./price-chart";

const SECTOR_ETF: Record<string, string> = {
  "Technology":              "XLK",
  "Communication Services":  "XLC",
  "Consumer Discretionary":  "XLY",
  "Consumer Staples":        "XLP",
  "Energy":                  "XLE",
  "Financials":              "XLF",
  "Healthcare":              "XLV",
  "Industrials":             "XLI",
  "Materials":               "XLB",
  "Real Estate":             "XLRE",
  "Utilities":               "XLU",
};

interface Props {
  data: MarketSectorsData | null;
  loading: boolean;
  error: string | null;
}

type RangeId = "d1" | "w1" | "m1" | "m3" | "ytd";

const RANGES: { id: RangeId; label: string }[] = [
  { id: "d1",  label: "1D" },
  { id: "w1",  label: "1W" },
  { id: "m1",  label: "1M" },
  { id: "m3",  label: "3M" },
  { id: "ytd", label: "YTD" },
];

function fmtPct(n: number): string {
  const sign = n >= 0 ? "+" : "";
  return `${sign}${n.toFixed(2)}%`;
}

function PctBar({ pct, maxAbs }: { pct: number; maxAbs: number }) {
  const w = Math.round((Math.abs(pct) / maxAbs) * 80);
  const color = pct >= 0 ? "#41d39d" : "#f87171";
  return (
    <div className="flex items-center justify-end gap-1.5">
      <span className="tabular-nums text-xs font-semibold" style={{ color }}>{fmtPct(pct)}</span>
      <div className="h-3 rounded-sm shrink-0" style={{ width: w, backgroundColor: color, opacity: 0.7 }} />
    </div>
  );
}

function StockRow({ stock, maxAbs }: { stock: SectorStock; maxAbs: number }) {
  const color = stock.up ? "#41d39d" : "#f87171";
  const sign = stock.pct >= 0 ? "+" : "";
  const barW = Math.round((Math.abs(stock.pct) / maxAbs) * 60);

  return (
    <tr className="border-b border-[color:var(--line)] last:border-0 hover:bg-[color:rgba(79,213,255,0.03)]">
      <td className="pl-10 pr-2 py-2 w-16">
        <span className="text-xs font-bold text-[color:var(--accent)]">{stock.symbol}</span>
      </td>
      <td className="px-2 py-2 text-xs text-[color:var(--ink-faint)]">{stock.name}</td>
      <td className="px-2 py-2 tabular-nums text-xs text-right text-[color:var(--ink)]">
        ${stock.price.toFixed(2)}
      </td>
      <td className="px-2 py-2 tabular-nums text-xs text-right font-semibold" style={{ color }}>
        {sign}{stock.pct.toFixed(2)}%
      </td>
      <td className="pl-2 pr-4 py-2 w-20">
        <div className="flex justify-end">
          <div className="h-2.5 rounded-sm" style={{ width: barW, backgroundColor: color, opacity: 0.7 }} />
        </div>
      </td>
    </tr>
  );
}

function SectorRow({
  sector,
  range,
  expanded,
  onToggle,
  maxAbs,
}: {
  sector: SectorData;
  range: RangeId;
  expanded: boolean;
  onToggle: () => void;
  maxAbs: number;
}) {
  const pct = sector.pcts[range];
  const stockMax = sector.stocks.length > 0
    ? Math.max(...sector.stocks.map((s) => Math.abs(s.pct)), 1)
    : 1;

  return (
    <>
      <tr
        className="border-b border-[color:var(--line)] cursor-pointer hover:bg-[color:rgba(79,213,255,0.04)] transition-colors"
        onClick={onToggle}
      >
        <td className="px-4 py-3 text-xs font-semibold text-[color:var(--ink)]">
          <span className="mr-2 text-[color:var(--ink-faint)]">{expanded ? "[-]" : "[+]"}</span>
          {sector.name}
        </td>
        <td className="px-4 py-3 text-right">
          <PctBar pct={pct} maxAbs={maxAbs} />
        </td>
      </tr>
      {expanded && (
        <tr>
          <td colSpan={2} className="p-0 bg-[color:rgba(9,21,34,0.3)]">
            {/* ETF price chart */}
            {SECTOR_ETF[sector.name] && (
              <div className="px-4 pt-4 pb-3 border-b border-[color:var(--line)]">
                <InlineChart
                  symbol={SECTOR_ETF[sector.name]}
                  type="yahoo"
                  name={sector.name}
                  up={sector.pcts[range] >= 0}
                  label={`${SECTOR_ETF[sector.name]} ETF`}
                />
              </div>
            )}
            {/* Top stocks */}
            {sector.stocks.length > 0 && (
              <table className="w-full">
                <tbody>
                  {sector.stocks.map((s) => (
                    <StockRow key={s.symbol} stock={s} maxAbs={stockMax} />
                  ))}
                </tbody>
              </table>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

export function SectorsTab({ data, loading, error }: Props) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [range, setRange] = useState<RangeId>("d1");

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center py-16 text-sm text-[color:var(--ink-faint)]">
        Loading sectors…
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

  const sorted = [...data.sectors].sort((a, b) => b.pcts[range] - a.pcts[range]);
  const maxAbs = Math.max(...sorted.map((s) => Math.abs(s.pcts[range])), 1);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
          Market Sectors
        </p>

        <div className="flex items-center gap-2">
          {/* Range selector */}
          <div className="flex items-center gap-0.5 rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)] p-1">
            {RANGES.map(({ id, label }) => (
              <button
                key={id}
                type="button"
                onClick={() => setRange(id)}
                className={`rounded-lg px-3 py-1 text-xs font-medium transition-colors ${
                  range === id
                    ? "bg-[color:rgba(79,213,255,0.18)] text-[color:var(--ink)]"
                    : "text-[color:var(--ink-faint)] hover:text-[color:var(--ink)]"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <span className="text-xs text-[color:var(--ink-faint)]">
            {new Date(data.generatedAt).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
          </span>
        </div>
      </div>

      <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
        <table className="w-full">
          <tbody>
            {sorted.map((sector) => (
              <SectorRow
                key={sector.name}
                sector={sector}
                range={range}
                expanded={expandedId === sector.name}
                onToggle={() => setExpandedId((prev) => prev === sector.name ? null : sector.name)}
                maxAbs={maxAbs}
              />
            ))}
          </tbody>
        </table>
      </div>

      {range !== "d1" && (
        <p className="text-right text-[11px] text-[color:var(--ink-faint)]">
          Sector % shown for {RANGES.find((r) => r.id === range)?.label}. Individual stock % is always 1D.
        </p>
      )}
    </div>
  );
}
