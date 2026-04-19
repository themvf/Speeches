"use client";

import type { MarketMoversData, MoverQuote } from "@/lib/server/types";

interface Props {
  data: MarketMoversData | null;
  loading: boolean;
  error: string | null;
}

function MoverRow({ q, maxAbs }: { q: MoverQuote; maxAbs: number }) {
  const color = q.up ? "#41d39d" : "#f87171";
  const sign = q.pct >= 0 ? "+" : "";
  const barW = Math.round((Math.abs(q.pct) / maxAbs) * 80);

  return (
    <tr className="border-b border-[color:var(--line)] last:border-0 hover:bg-[color:rgba(79,213,255,0.04)]">
      <td className="pl-4 pr-2 py-2.5 w-8 text-xs text-[color:var(--ink-faint)] tabular-nums">{q.rank}</td>
      <td className="px-2 py-2.5 w-16">
        <span className="text-xs font-bold text-[color:var(--accent)]">{q.symbol}</span>
      </td>
      <td className="px-2 py-2.5 text-xs text-[color:var(--ink-faint)] max-w-[160px] truncate">{q.name}</td>
      <td className="px-2 py-2.5 tabular-nums text-xs text-right text-[color:var(--ink)]">
        ${q.price.toFixed(2)}
      </td>
      <td className="px-2 py-2.5 tabular-nums text-xs text-right font-semibold" style={{ color }}>
        {sign}{q.pct.toFixed(2)}%
      </td>
      <td className="pl-2 pr-4 py-2.5 w-24">
        <div className="flex justify-end">
          <div className="h-3 rounded-sm" style={{ width: barW, backgroundColor: color, opacity: 0.7 }} />
        </div>
      </td>
    </tr>
  );
}

function MoversList({ title, items, color }: { title: string; items: MoverQuote[]; color: string }) {
  const maxAbs = items.length > 0 ? Math.max(...items.map((q) => Math.abs(q.pct)), 1) : 1;

  return (
    <div>
      <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.12em]" style={{ color }}>
        {title}
      </p>
      <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
        <table className="w-full">
          <tbody>
            {items.length === 0 ? (
              <tr>
                <td className="px-4 py-6 text-center text-xs text-[color:var(--ink-faint)]">No data</td>
              </tr>
            ) : (
              items.map((q) => <MoverRow key={q.symbol} q={q} maxAbs={maxAbs} />)
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function MoversTab({ data, loading, error }: Props) {
  if (loading && !data) {
    return (
      <div className="flex items-center justify-center py-16 text-sm text-[color:var(--ink-faint)]">
        Loading movers…
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
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
          Market Movers
        </p>
        <span className="text-xs text-[color:var(--ink-faint)]">
          {new Date(data.generatedAt).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
        </span>
      </div>

      <MoversList title="Top Gainers" items={data.gainers} color="#41d39d" />
      <MoversList title="Top Losers"  items={data.losers}  color="#f87171" />
    </div>
  );
}
