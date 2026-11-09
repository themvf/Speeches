"use client";

import { useEffect, useMemo, useState } from "react";
import { portfolioMetrics } from "@/lib/portfolio-calculations";
import type { Portfolio, PortfolioQuote } from "@/lib/portfolio-types";

const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
const MAX_COMPARISON_SYMBOLS = 500;

type SortKey = "return" | "value" | "name";

function pct(value: number | null): string {
  return value == null ? "—" : `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function tone(value: number | null): string {
  return value == null ? "text-[color:var(--ink-faint)]" : value >= 0 ? "text-emerald-400" : "text-rose-400";
}

function chunks<T>(values: T[], size: number): T[][] {
  const result: T[][] = [];
  for (let index = 0; index < values.length; index += size) result.push(values.slice(index, index + size));
  return result;
}

export function PortfolioComparison({ portfolios, onOpenPortfolio }: { portfolios: Portfolio[]; onOpenPortfolio: (portfolioId: string) => void }) {
  const [quotes, setQuotes] = useState<Map<string, PortfolioQuote>>(new Map());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [refreshedAt, setRefreshedAt] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("return");
  const allSymbols = useMemo(() => Array.from(new Set(portfolios.flatMap((portfolio) => portfolio.holdings.map((holding) => holding.symbol)))).sort(), [portfolios]);
  const requestedSymbols = useMemo(() => allSymbols.slice(0, MAX_COMPARISON_SYMBOLS), [allSymbols]);
  const symbolsKey = requestedSymbols.join(",");

  useEffect(() => {
    if (!symbolsKey) { setQuotes(new Map()); setRefreshedAt(null); return; }
    let cancelled = false;
    setLoading(true);
    setError("");
    void Promise.all(chunks(requestedSymbols, 100).map(async (symbols) => {
      const response = await fetch(`/api/portfolio-quotes?symbols=${encodeURIComponent(symbols.join(","))}`, { cache: "no-store" });
      const body = await response.json() as { data?: PortfolioQuote[]; error?: string };
      if (!response.ok) throw new Error(body.error || "Portfolio quotes could not be loaded.");
      return body.data ?? [];
    })).then((batches) => {
      if (cancelled) return;
      const rows = batches.flat();
      setQuotes(new Map(rows.map((quote) => [quote.symbol, quote])));
      const observations = rows.map((quote) => Date.parse(quote.asOf)).filter(Number.isFinite);
      setRefreshedAt(observations.length ? new Date(Math.min(...observations)).toISOString() : null);
    }).catch((reason) => {
      if (!cancelled) setError(reason instanceof Error ? reason.message : "Portfolio quotes could not be loaded.");
    }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [requestedSymbols, symbolsKey]);

  const rows = useMemo(() => portfolios.map((portfolio) => {
    const metrics = portfolioMetrics(portfolio, quotes);
    const symbolValues = new Map<string, number>();
    for (const holding of portfolio.holdings) {
      const quote = quotes.get(holding.symbol);
      if (quote) symbolValues.set(holding.symbol, (symbolValues.get(holding.symbol) ?? 0) + holding.shares * quote.price);
    }
    const topValue = symbolValues.size ? Math.max(...symbolValues.values()) : null;
    const topWeight = metrics.marketValue != null && topValue != null && metrics.marketValue > 0 ? (topValue / metrics.marketValue) * 100 : null;
    return { portfolio, metrics, topWeight };
  }).sort((a, b) => {
    if (sortKey === "name") return a.portfolio.name.localeCompare(b.portfolio.name);
    if (sortKey === "value") return (b.metrics.marketValue ?? -Infinity) - (a.metrics.marketValue ?? -Infinity);
    return (b.metrics.gainPct ?? -Infinity) - (a.metrics.gainPct ?? -Infinity);
  }), [portfolios, quotes, sortKey]);

  const completeRows = rows.filter((row) => row.metrics.marketValue != null);
  const best = completeRows.reduce<(typeof rows)[number] | null>((winner, row) => !winner || (row.metrics.gainPct ?? -Infinity) > (winner.metrics.gainPct ?? -Infinity) ? row : winner, null);
  const totalInvested = rows.reduce((sum, row) => sum + row.metrics.cost, 0);

  return <div className="space-y-5">
    <div className="panel overflow-hidden">
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-[color:var(--line)] p-5">
        <div><p className="text-[11px] font-semibold uppercase tracking-[.14em] text-cyan-300">Strategy comparison</p><h2 className="mt-1 text-xl font-semibold">Compare current portfolio snapshots</h2><p className="mt-1 max-w-2xl text-xs text-[color:var(--ink-faint)]">Every result uses the latest available quote. This compares current hypothetical outcomes—not historical returns or benchmark performance.</p></div>
        <div className="flex items-center gap-1 rounded-lg border border-[color:var(--line)] p-1 text-xs">{(["return", "value", "name"] as SortKey[]).map((key) => <button key={key} type="button" onClick={() => setSortKey(key)} className={`rounded-md px-3 py-1.5 capitalize ${sortKey === key ? "bg-cyan-400/15 text-cyan-200" : "text-[color:var(--ink-faint)] hover:text-[color:var(--ink)]"}`}>{key}</button>)}</div>
      </div>
      <div className="grid gap-3 p-5 sm:grid-cols-3">
        <div className="rounded-xl border border-[color:var(--line)] bg-white/[.02] p-4"><p className="text-[10px] font-semibold uppercase tracking-wider text-[color:var(--ink-faint)]">Strategies</p><p className="mt-2 text-2xl font-semibold">{portfolios.length}</p><p className="mt-1 text-xs text-[color:var(--ink-faint)]">{completeRows.length} fully priced</p></div>
        <div className="rounded-xl border border-[color:var(--line)] bg-white/[.02] p-4"><p className="text-[10px] font-semibold uppercase tracking-wider text-[color:var(--ink-faint)]">Total hypothetical investment</p><p className="mt-2 text-2xl font-semibold tabular-nums">{money.format(totalInvested)}</p><p className="mt-1 text-xs text-[color:var(--ink-faint)]">Across all saved portfolios</p></div>
        <div className="rounded-xl border border-[color:var(--line)] bg-white/[.02] p-4"><p className="text-[10px] font-semibold uppercase tracking-wider text-[color:var(--ink-faint)]">Highest current return</p><p className={`mt-2 text-2xl font-semibold tabular-nums ${tone(best?.metrics.gainPct ?? null)}`}>{pct(best?.metrics.gainPct ?? null)}</p><p className="mt-1 truncate text-xs text-[color:var(--ink-faint)]">{best?.portfolio.name ?? "Awaiting complete quotes"}</p></div>
      </div>
    </div>

    {error ? <div className="rounded-xl border border-amber-400/20 bg-amber-400/5 p-3 text-sm text-amber-300">{error} Saved trade data is unaffected.</div> : null}
    {allSymbols.length > MAX_COMPARISON_SYMBOLS ? <div className="rounded-xl border border-amber-400/20 bg-amber-400/5 p-3 text-sm text-amber-300">Comparison is limited to the first {MAX_COMPARISON_SYMBOLS} unique tickers; portfolios containing additional symbols are marked incomplete.</div> : null}

    <div className="panel overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[color:var(--line)] px-5 py-4"><div><h3 className="font-semibold">Portfolio results</h3><p className="mt-0.5 text-xs text-[color:var(--ink-faint)]">Select a portfolio to return to its positions.</p></div><span className="text-[11px] text-[color:var(--ink-faint)]">{loading ? "Loading prices…" : refreshedAt ? `Oldest quote ${new Date(refreshedAt).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}` : "No live prices"}</span></div>
      {portfolios.length > 1 ? <div className="overflow-x-auto"><table className="w-full min-w-[960px] text-sm"><thead className="bg-white/[.025] text-[10px] uppercase tracking-wider text-[color:var(--ink-faint)]"><tr><th className="px-4 py-3 text-left">Portfolio</th><th className="px-3 py-3 text-right">Positions</th><th className="px-3 py-3 text-right">Invested</th><th className="px-3 py-3 text-right">Market value</th><th className="px-3 py-3 text-right">Gain / loss</th><th className="px-3 py-3 text-right">Return</th><th className="px-3 py-3 text-right">Today</th><th className="px-3 py-3 text-right">Top ticker weight</th><th className="px-3 py-3 text-right">Coverage</th></tr></thead><tbody className="divide-y divide-[color:var(--line-soft)]">{rows.map(({ portfolio, metrics, topWeight }) => <tr key={portfolio.id} className="hover:bg-white/[.025]"><td className="px-4 py-3"><button type="button" onClick={() => onOpenPortfolio(portfolio.id)} className="font-semibold text-cyan-300 hover:text-cyan-200 hover:underline">{portfolio.name}</button></td><td className="px-3 py-3 text-right tabular-nums">{portfolio.holdings.length}</td><td className="px-3 py-3 text-right tabular-nums">{money.format(metrics.cost)}</td><td className="px-3 py-3 text-right tabular-nums">{metrics.marketValue == null ? "—" : money.format(metrics.marketValue)}</td><td className={`px-3 py-3 text-right tabular-nums ${tone(metrics.gain)}`}>{metrics.gain == null ? "—" : money.format(metrics.gain)}</td><td className={`px-3 py-3 text-right font-semibold tabular-nums ${tone(metrics.gainPct)}`}>{pct(metrics.gainPct)}</td><td className={`px-3 py-3 text-right tabular-nums ${tone(metrics.todayGain)}`}>{metrics.todayGain == null ? "—" : money.format(metrics.todayGain)}</td><td className="px-3 py-3 text-right tabular-nums">{topWeight == null ? "—" : `${topWeight.toFixed(1)}%`}</td><td className="px-3 py-3 text-right"><span className={`rounded-full border px-2 py-1 text-[10px] ${metrics.pricedHoldings === portfolio.holdings.length && portfolio.holdings.length > 0 ? "border-emerald-400/25 bg-emerald-400/10 text-emerald-300" : "border-amber-400/25 bg-amber-400/10 text-amber-300"}`}>{metrics.pricedHoldings}/{portfolio.holdings.length} priced</span></td></tr>)}</tbody></table></div> : <div className="px-6 py-14 text-center"><p className="font-semibold">Create a second portfolio to compare strategies</p><p className="mt-1 text-sm text-[color:var(--ink-faint)]">Your current portfolio remains saved and unchanged.</p></div>}
    </div>
    <p className="px-1 text-[11px] text-[color:var(--ink-faint)]">Comparison quotes are requested only when this view opens and do not create Neon writes. Missing prices hide affected portfolio totals rather than treating them as zero.</p>
  </div>;
}
