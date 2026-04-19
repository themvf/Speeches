"use client";

import type { CryptoCoin, MarketCryptoData } from "@/lib/server/types";

interface Props {
  data: MarketCryptoData | null;
  loading: boolean;
  error: string | null;
}

function fmtPrice(n: number): string {
  if (n >= 1) return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (n >= 0.01) return n.toFixed(4);
  return n.toFixed(6);
}

function fmtLarge(n: number): string {
  if (n >= 1e12) return `$${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9)  return `$${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6)  return `$${(n / 1e6).toFixed(1)}M`;
  return `$${n.toLocaleString()}`;
}

function CryptoRow({ coin }: { coin: CryptoCoin }) {
  const color = coin.up ? "#41d39d" : "#f87171";
  const sign = coin.pct24h >= 0 ? "+" : "";
  const barW = Math.min(48, Math.round(Math.abs(coin.pct24h) * 2.5));

  return (
    <tr className="border-b border-[color:var(--line)] last:border-0 hover:bg-[color:rgba(79,213,255,0.04)] transition-colors">
      <td className="pl-4 pr-2 py-2.5 w-8 tabular-nums text-xs text-[color:var(--ink-faint)]">{coin.rank}</td>
      <td className="px-2 py-2.5 w-14">
        <span className="text-xs font-bold text-[color:var(--accent)]">{coin.symbol}</span>
      </td>
      <td className="px-2 py-2.5 text-xs text-[color:var(--ink-faint)] max-w-[120px] truncate">{coin.name}</td>
      <td className="px-2 py-2.5 tabular-nums text-xs text-right text-[color:var(--ink)]">${fmtPrice(coin.price)}</td>
      <td className="px-2 py-2.5 tabular-nums text-xs text-right font-semibold" style={{ color }}>
        {sign}{coin.pct24h.toFixed(2)}%
      </td>
      <td className="hidden sm:table-cell px-2 py-2.5 tabular-nums text-xs text-right text-[color:var(--ink-faint)]">
        {fmtLarge(coin.marketCap)}
      </td>
      <td className="hidden sm:table-cell px-2 py-2.5 tabular-nums text-xs text-right text-[color:var(--ink-faint)]">
        {fmtLarge(coin.volume24h)}
      </td>
      <td className="pl-2 pr-4 py-2.5 w-16">
        <div className="flex justify-end">
          <div className="h-3 rounded-sm" style={{ width: barW, backgroundColor: color, opacity: 0.7 }} />
        </div>
      </td>
    </tr>
  );
}

export function CryptoTab({ data, loading, error }: Props) {
  if (loading && !data) {
    return (
      <div className="flex items-center justify-center py-16 text-sm text-[color:var(--ink-faint)]">
        Loading crypto markets…
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
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
          Crypto Markets
          <span className="ml-2 font-normal text-[color:var(--ink-faint)]">{data.coins.length} coins</span>
        </p>
        <span className="text-xs text-[color:var(--ink-faint)]">
          {new Date(data.generatedAt).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
        </span>
      </div>

      <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
        <table className="w-full">
          <thead>
            <tr className="border-b border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)]">
              <th className="pl-4 pr-2 py-2 text-left text-[10px] font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)] w-8">#</th>
              <th className="px-2 py-2 text-left text-[10px] font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Name</th>
              <th className="px-2 py-2 text-left text-[10px] font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]"></th>
              <th className="px-2 py-2 text-right text-[10px] font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Price</th>
              <th className="px-2 py-2 text-right text-[10px] font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">24H %</th>
              <th className="hidden sm:table-cell px-2 py-2 text-right text-[10px] font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">MCAP</th>
              <th className="hidden sm:table-cell px-2 py-2 text-right text-[10px] font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Vol 24H</th>
              <th className="pl-2 pr-4 py-2 w-16"></th>
            </tr>
          </thead>
          <tbody>
            {data.coins.map((coin) => <CryptoRow key={coin.id} coin={coin} />)}
          </tbody>
        </table>
      </div>
    </div>
  );
}
