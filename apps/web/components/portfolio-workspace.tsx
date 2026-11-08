"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { usePortfolios } from "@/hooks/use-portfolios";
import { holdingMetrics, normalizeSymbol, portfolioMetrics } from "@/lib/portfolio-calculations";
import type { PortfolioHolding, PortfolioQuote } from "@/lib/portfolio-types";

const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2 });

function fmtMoney(value: number | null): string { return value == null ? "Unavailable" : money.format(value); }
function fmtPct(value: number | null): string { return value == null ? "—" : `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`; }
function gainClass(value: number | null): string { return value == null ? "text-[color:var(--ink-faint)]" : value >= 0 ? "text-emerald-400" : "text-rose-400"; }

type Tile = { holding: PortfolioHolding; quote?: PortfolioQuote; value: number; x: number; y: number; width: number; height: number };

function treemap(holdings: PortfolioHolding[], quotes: Map<string, PortfolioQuote>): Tile[] {
  const entries = holdings.map((holding) => ({ holding, quote: quotes.get(holding.symbol), value: holding.shares * (quotes.get(holding.symbol)?.price ?? holding.tradePrice) })).sort((a, b) => b.value - a.value);
  const tiles: Tile[] = [];
  function split(items: typeof entries, x: number, y: number, width: number, height: number, vertical: boolean): void {
    if (!items.length) return;
    if (items.length === 1) { tiles.push({ ...items[0], x, y, width, height }); return; }
    const total = items.reduce((sum, item) => sum + item.value, 0);
    let leftTotal = 0;
    let pivot = 0;
    while (pivot < items.length - 1 && leftTotal + items[pivot].value <= total / 2) { leftTotal += items[pivot].value; pivot += 1; }
    if (pivot === 0) { leftTotal = items[0].value; pivot = 1; }
    const ratio = leftTotal / total;
    if (vertical) {
      const leftWidth = width * ratio;
      split(items.slice(0, pivot), x, y, leftWidth, height, !vertical);
      split(items.slice(pivot), x + leftWidth, y, width - leftWidth, height, !vertical);
    } else {
      const topHeight = height * ratio;
      split(items.slice(0, pivot), x, y, width, topHeight, !vertical);
      split(items.slice(pivot), x, y + topHeight, width, height - topHeight, !vertical);
    }
  }
  split(entries, 0, 0, 100, 100, true);
  return tiles;
}

function tileColor(gainPct: number | null): string {
  if (gainPct == null) return "rgba(71, 94, 118, .72)";
  const strength = Math.min(0.88, 0.34 + Math.abs(gainPct) / 45);
  return gainPct >= 0 ? `rgba(20, 137, 91, ${strength})` : `rgba(190, 54, 72, ${strength})`;
}

function SummaryCard({ label, value, detail, tone }: { label: string; value: string; detail: string; tone?: string }) {
  return <div className="rounded-xl border border-[color:var(--line)] bg-[rgba(7,18,30,.72)] p-4">
    <p className="text-[11px] font-semibold uppercase tracking-[.12em] text-[color:var(--ink-faint)]">{label}</p>
    <p className={`mt-2 text-xl font-semibold tabular-nums ${tone ?? "text-[color:var(--ink)]"}`}>{value}</p>
    <p className="mt-1 text-xs text-[color:var(--ink-faint)]">{detail}</p>
  </div>;
}
function PositionForm({ onAdd, onCancel }: { onAdd: (input: { symbol: string; shares: number; tradePrice: number; tradeDate?: string }) => void; onCancel: () => void }) {
  const [symbol, setSymbol] = useState("");
  const [shares, setShares] = useState("");
  const [tradePrice, setTradePrice] = useState("");
  const [tradeDate, setTradeDate] = useState("");
  const [error, setError] = useState("");
  function submit(event: FormEvent) {
    event.preventDefault();
    const cleanSymbol = normalizeSymbol(symbol);
    const shareCount = Number(shares);
    const price = Number(tradePrice);
    if (!cleanSymbol || !Number.isFinite(shareCount) || shareCount <= 0 || !Number.isFinite(price) || price <= 0) {
      setError("Enter a valid ticker, share count, and trade price greater than zero."); return;
    }
    onAdd({ symbol: cleanSymbol, shares: shareCount, tradePrice: price, tradeDate: tradeDate || undefined });
  }
  return <form onSubmit={submit} className="rounded-xl border border-cyan-400/25 bg-cyan-400/[.045] p-4">
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
      <label className="text-xs text-[color:var(--ink-faint)]">Ticker<input autoFocus value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="AAPL" className="form-control mt-1 w-full px-3 py-2 uppercase" /></label>
      <label className="text-xs text-[color:var(--ink-faint)]">Shares<input inputMode="decimal" value={shares} onChange={(e) => setShares(e.target.value)} placeholder="10" className="form-control mt-1 w-full px-3 py-2" /></label>
      <label className="text-xs text-[color:var(--ink-faint)]">Trade price<input inputMode="decimal" value={tradePrice} onChange={(e) => setTradePrice(e.target.value)} placeholder="125.50" className="form-control mt-1 w-full px-3 py-2" /></label>
      <label className="text-xs text-[color:var(--ink-faint)]">Trade date <span className="opacity-70">(optional)</span><input type="date" value={tradeDate} onChange={(e) => setTradeDate(e.target.value)} className="form-control mt-1 w-full px-3 py-2" /></label>
      <div className="flex items-end gap-2"><button className="min-h-10 flex-1 rounded-lg bg-cyan-400 px-3 py-2 text-sm font-semibold text-slate-950">Add position</button><button type="button" onClick={onCancel} className="min-h-10 rounded-lg border border-[color:var(--line)] px-3 py-2 text-sm">Cancel</button></div>
    </div>
    {error ? <p className="mt-2 text-xs text-rose-400">{error}</p> : null}
  </form>;
}

function EditableNumber({ value, label, onCommit }: { value: number; label: string; onCommit: (value: number) => void }) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);
  function commit() { const parsed = Number(draft); if (Number.isFinite(parsed) && parsed > 0) onCommit(parsed); else setDraft(String(value)); }
  return <input aria-label={label} inputMode="decimal" value={draft} onChange={(e) => setDraft(e.target.value)} onBlur={commit} onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }} className="form-control w-24 px-2 py-1.5 text-right tabular-nums" />;
}

export function PortfolioWorkspace() {
  const store = usePortfolios();
  const [activeId, setActiveId] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [showAdd, setShowAdd] = useState(false);
  const [quotes, setQuotes] = useState<Map<string, PortfolioQuote>>(new Map());
  const [quoteTime, setQuoteTime] = useState<string | null>(null);
  const [quoteLoading, setQuoteLoading] = useState(false);
  const [quoteError, setQuoteError] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [undo, setUndo] = useState<{ message: string; restore: () => void } | null>(null);
  const [syncNotice, setSyncNotice] = useState("");
  const active = store.portfolios.find((portfolio) => portfolio.id === activeId) ?? store.portfolios[0];

  useEffect(() => { if (active && !activeId) setActiveId(active.id); }, [active, activeId]);
  useEffect(() => { if (activeId && !store.portfolios.some((item) => item.id === activeId)) setActiveId(store.portfolios[0]?.id ?? null); }, [activeId, store.portfolios]);

  const refreshQuotes = useCallback(async () => {
    if (!active?.holdings.length) { setQuotes(new Map()); setQuoteTime(null); return; }
    setQuoteLoading(true); setQuoteError("");
    try {
      const symbols = Array.from(new Set(active.holdings.map((holding) => holding.symbol))).join(",");
      const response = await fetch(`/api/portfolio-quotes?symbols=${encodeURIComponent(symbols)}`, { cache: "no-store" });
      const body = await response.json() as { data?: PortfolioQuote[]; missingSymbols?: string[]; error?: string };
      if (!response.ok) throw new Error(body.error || "Live quotes could not be loaded.");
      const nextQuotes = body.data ?? [];
      setQuotes(new Map(nextQuotes.map((quote) => [quote.symbol, quote])));
      const oldestObservation = nextQuotes.reduce<number | null>((oldest, quote) => {
        const observed = Date.parse(quote.asOf);
        if (!Number.isFinite(observed)) return oldest;
        return oldest == null ? observed : Math.min(oldest, observed);
      }, null);
      setQuoteTime(oldestObservation == null ? null : new Date(oldestObservation).toISOString());
      if (body.missingSymbols?.length) setQuoteError(`Quotes unavailable for ${body.missingSymbols.join(", ")}. Complete portfolio totals are hidden.`);
    } catch (error) { setQuoteError(error instanceof Error ? error.message : "Live quotes could not be loaded."); }
    finally { setQuoteLoading(false); }
  }, [active]);

  useEffect(() => { void refreshQuotes(); }, [refreshQuotes]);
  useEffect(() => {
    if (!autoRefresh || !active?.holdings.length) return;
    const timer = window.setInterval(() => {
      if (document.visibilityState !== "visible") return;
      if (quotes.size > 0 && [...quotes.values()].every((quote) => quote.status === "CLOSED")) return;
      void refreshQuotes();
    }, 5 * 60 * 1_000);
    return () => window.clearInterval(timer);
  }, [active?.holdings.length, autoRefresh, quotes, refreshQuotes]);
  useEffect(() => {
    if (!undo) return;
    const timer = window.setTimeout(() => setUndo(null), 8_000);
    return () => window.clearTimeout(timer);
  }, [undo]);
  useEffect(() => {
    if (!syncNotice) return;
    const timer = window.setTimeout(() => setSyncNotice(""), 6_000);
    return () => window.clearTimeout(timer);
  }, [syncNotice]);
  const metrics = useMemo(() => active ? portfolioMetrics(active, quotes) : null, [active, quotes]);
  const tiles = useMemo(() => active ? treemap(active.holdings, quotes) : [], [active, quotes]);

  if (!store.loaded) return <div className="panel p-8 text-sm text-[color:var(--ink-faint)]">Loading saved portfolios…</div>;

  function create(event: FormEvent) {
    event.preventDefault();
    const created = store.createPortfolio(newName);
    if (created) { setActiveId(created); setNewName(""); setShowCreate(false); setShowAdd(false); }
  }

  function removePortfolio() {
    if (!active) return;
    const removed = active;
    const index = store.portfolios.findIndex((portfolio) => portfolio.id === removed.id);
    store.deletePortfolio(removed.id);
    setUndo({
      message: `Removed ${removed.name}.`,
      restore: () => { store.restorePortfolio(removed, index); setActiveId(removed.id); },
    });
  }

  function removeHolding(holding: PortfolioHolding) {
    if (!active) return;
    const portfolioId = active.id;
    const index = active.holdings.findIndex((item) => item.id === holding.id);
    store.deleteHolding(portfolioId, holding.id);
    setUndo({ message: `Removed ${holding.symbol}.`, restore: () => store.restoreHolding(portfolioId, holding, index) });
  }

  async function connectAnotherDevice() {
    if (!store.syncLink) return;
    try {
      if (navigator.share) {
        await navigator.share({ title: "My portfolio workspace", text: "Open this private link once to sync this portfolio workspace.", url: store.syncLink });
        setSyncNotice("Private sync link shared.");
      } else {
        await navigator.clipboard.writeText(store.syncLink);
        setSyncNotice("Private sync link copied. Open it once on your other device.");
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      window.prompt("Copy this private sync link and open it once on your other device:", store.syncLink);
    }
  }

  return <div className="grid gap-5 lg:grid-cols-[240px_minmax(0,1fr)]">
    <aside className="panel h-fit overflow-hidden lg:sticky lg:top-24">
      <div className="border-b border-[color:var(--line)] p-4"><p className="text-xs font-semibold uppercase tracking-[.14em] text-[color:var(--ink-faint)]">Portfolios</p></div>
      <div className="space-y-1 p-2">{store.portfolios.map((portfolio) => <button key={portfolio.id} onClick={() => { setActiveId(portfolio.id); setShowAdd(false); }} className={`w-full rounded-lg px-3 py-3 text-left transition ${active?.id === portfolio.id ? "border border-cyan-400/30 bg-cyan-400/10 text-cyan-200" : "border border-transparent text-[color:var(--ink-soft)] hover:bg-white/5"}`}>
        <span className="block truncate text-sm font-semibold">{portfolio.name}</span><span className="mt-0.5 block text-[11px] opacity-65">{portfolio.holdings.length} {portfolio.holdings.length === 1 ? "position" : "positions"}</span>
      </button>)}</div>
      <div className="border-t border-[color:var(--line)] p-3">{showCreate ? <form onSubmit={create} className="space-y-2"><input autoFocus value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="Portfolio name" className="form-control w-full px-3 py-2 text-sm" /><div className="flex gap-2"><button className="flex-1 rounded-lg bg-cyan-400 px-2 py-2 text-xs font-semibold text-slate-950">Create</button><button type="button" onClick={() => setShowCreate(false)} className="rounded-lg border border-[color:var(--line)] px-2 py-2 text-xs">Cancel</button></div></form> : <button onClick={() => setShowCreate(true)} className="w-full rounded-lg border border-dashed border-[color:var(--line-strong)] px-3 py-2 text-sm font-semibold text-cyan-300 hover:bg-cyan-400/5">+ New portfolio</button>}</div>
    </aside>

    <section className="min-w-0 space-y-5">
      {active ? <>
        <div className="panel overflow-hidden">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[color:var(--line)] p-5">
            <div><div className="flex items-center gap-2"><h2 className="text-xl font-semibold">{active.name}</h2><span className="rounded-full border border-[color:var(--line)] px-2 py-0.5 text-[10px] uppercase tracking-wider text-[color:var(--ink-faint)]">{store.syncing ? "Syncing" : store.syncPending ? "Pending sync" : store.syncEnabled ? "Synced privately" : "Saved locally"}</span></div><p className="mt-1 text-xs text-[color:var(--ink-faint)]">A hypothetical strategy workspace. Cost basis uses your entered trade price; estimates use the latest available quote.</p>{syncNotice ? <p role="status" className="mt-2 text-xs text-emerald-300">{syncNotice}</p> : null}</div>
            <div className="flex flex-wrap gap-2"><button onClick={() => setShowAdd(true)} className="rounded-lg bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950">+ Add position</button><button onClick={() => void connectAnotherDevice()} disabled={!store.syncLink} className="rounded-lg border border-cyan-400/25 px-3 py-2 text-sm text-cyan-200 disabled:opacity-50">Connect another device</button><button onClick={() => void refreshQuotes()} disabled={quoteLoading} className="rounded-lg border border-[color:var(--line)] px-3 py-2 text-sm disabled:opacity-50">{quoteLoading ? "Refreshing…" : "Refresh prices"}</button><button onClick={() => setAutoRefresh((enabled) => !enabled)} aria-pressed={autoRefresh} className={`rounded-lg border px-3 py-2 text-sm ${autoRefresh ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-300" : "border-[color:var(--line)]"}`}>Live {autoRefresh ? "on" : "off"}</button><button onClick={() => { const name = window.prompt("Rename portfolio", active.name); if (name) store.renamePortfolio(active.id, name); }} className="rounded-lg border border-[color:var(--line)] px-3 py-2 text-sm">Rename</button>{store.portfolios.length > 1 ? <button onClick={removePortfolio} className="rounded-lg border border-rose-400/25 px-3 py-2 text-sm text-rose-300">Delete</button> : null}</div>
          </div>
          <div className="grid gap-3 p-5 sm:grid-cols-2 xl:grid-cols-4">
            <SummaryCard label="Amount invested" value={fmtMoney(metrics?.cost ?? 0)} detail={`${active.holdings.length} saved ${active.holdings.length === 1 ? "position" : "positions"}`} />
            <SummaryCard label="Market value" value={fmtMoney(metrics?.marketValue ?? null)} detail={metrics && metrics.pricedHoldings < active.holdings.length ? `${metrics.pricedHoldings} of ${active.holdings.length} positions priced` : quoteTime ? `Oldest quote ${new Date(quoteTime).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}` : "Add a position to begin"} />
            <SummaryCard label="Total gain / loss" value={fmtMoney(metrics?.gain ?? null)} detail={fmtPct(metrics?.gainPct ?? null)} tone={gainClass(metrics?.gain ?? null)} />
            <SummaryCard label="Today's change" value={fmtMoney(metrics?.todayGain ?? null)} detail="Shares × today's price change" tone={gainClass(metrics?.todayGain ?? null)} />
          </div>
        </div>

        {showAdd ? <PositionForm onCancel={() => setShowAdd(false)} onAdd={(input) => { store.addHolding(active.id, input); setShowAdd(false); }} /> : null}
        {quoteError ? <div className="rounded-xl border border-amber-400/20 bg-amber-400/5 p-3 text-sm text-amber-300">{quoteError} Your saved cost basis is unaffected.</div> : null}

        <div className="panel overflow-hidden">
          <div className="flex items-center justify-between border-b border-[color:var(--line)] px-5 py-4"><div><h3 className="font-semibold">Positions</h3><p className="mt-0.5 text-xs text-[color:var(--ink-faint)]">Edit shares or trade price inline; changes autosave.</p></div></div>
          {active.holdings.length ? <div className="overflow-x-auto"><table className="w-full min-w-[900px] text-sm"><thead className="bg-white/[.025] text-[11px] uppercase tracking-wider text-[color:var(--ink-faint)]"><tr><th className="px-4 py-3 text-left">Ticker</th><th className="px-3 py-3 text-right">Shares</th><th className="px-3 py-3 text-right">Trade price</th><th className="px-3 py-3 text-right">Cost basis</th><th className="px-3 py-3 text-right">Current price</th><th className="px-3 py-3 text-right">Market value</th><th className="px-3 py-3 text-right">Gain / loss</th><th className="px-3 py-3 text-right">Today</th><th className="px-3 py-3"><span className="sr-only">Actions</span></th></tr></thead><tbody className="divide-y divide-[color:var(--line-soft)]">{active.holdings.map((holding) => {
            const quote = quotes.get(holding.symbol); const row = holdingMetrics(holding, quote);
            return <tr key={holding.id} className="hover:bg-white/[.025]"><td className="px-4 py-3"><span className="font-semibold text-cyan-300">{holding.symbol}</span><span className="ml-2 text-xs text-[color:var(--ink-faint)]">{quote?.name ?? (quoteLoading ? "Loading…" : "Quote unavailable")}</span>{holding.tradeDate ? <span className="mt-0.5 block text-[10px] text-[color:var(--ink-faint)]">Bought {new Date(`${holding.tradeDate}T00:00:00`).toLocaleDateString()}</span> : null}</td><td className="px-3 py-3 text-right"><EditableNumber value={holding.shares} label={`${holding.symbol} shares`} onCommit={(shares) => store.updateHolding(active.id, holding.id, { shares })} /></td><td className="px-3 py-3 text-right"><EditableNumber value={holding.tradePrice} label={`${holding.symbol} trade price`} onCommit={(tradePrice) => store.updateHolding(active.id, holding.id, { tradePrice })} /></td><td className="px-3 py-3 text-right tabular-nums">{money.format(row.cost)}</td><td className="px-3 py-3 text-right tabular-nums">{quote ? money.format(quote.price) : "—"}</td><td className="px-3 py-3 text-right tabular-nums">{row.marketValue == null ? "—" : money.format(row.marketValue)}</td><td className={`px-3 py-3 text-right tabular-nums ${gainClass(row.gain)}`}>{row.gain == null ? "—" : <>{money.format(row.gain)}<span className="ml-1 text-xs">{fmtPct(row.gainPct)}</span></>}</td><td className={`px-3 py-3 text-right tabular-nums ${gainClass(row.todayGain)}`}>{row.todayGain == null ? "—" : money.format(row.todayGain)}</td><td className="px-3 py-3 text-right"><button onClick={() => removeHolding(holding)} className="rounded-md px-2 py-1 text-xs text-rose-300 hover:bg-rose-400/10">Remove</button></td></tr>;
          })}</tbody></table></div> : <div className="px-5 py-14 text-center"><p className="text-base font-semibold">No positions yet</p><p className="mt-1 text-sm text-[color:var(--ink-faint)]">Add a ticker, share count, and your trade price to start tracking performance.</p><button onClick={() => setShowAdd(true)} className="mt-4 rounded-lg bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950">Add your first position</button></div>}
        </div>

        {active.holdings.length ? <div className="panel overflow-hidden"><div className="flex flex-wrap items-center justify-between gap-2 border-b border-[color:var(--line)] px-5 py-4"><div><h3 className="font-semibold">Portfolio map</h3><p className="mt-0.5 text-xs text-[color:var(--ink-faint)]">Tile size = market value; color = gain or loss since your trade.</p></div><div className="flex items-center gap-3 text-[10px] uppercase tracking-wider text-[color:var(--ink-faint)]"><span className="inline-flex items-center gap-1"><i className="h-2.5 w-2.5 rounded-sm bg-rose-500" /> Loss</span><span className="inline-flex items-center gap-1"><i className="h-2.5 w-2.5 rounded-sm bg-slate-500" /> No quote</span><span className="inline-flex items-center gap-1"><i className="h-2.5 w-2.5 rounded-sm bg-emerald-500" /> Gain</span></div></div><div className="p-3"><div className="relative h-[420px] overflow-hidden rounded-lg border border-[color:var(--line)] bg-[#07111d]">{tiles.map((tile) => { const result = holdingMetrics(tile.holding, tile.quote); return <div key={tile.holding.id} className="absolute flex items-center justify-center overflow-hidden border border-[#07111d] p-2 text-center transition-opacity hover:opacity-90" style={{ left: `${tile.x}%`, top: `${tile.y}%`, width: `${tile.width}%`, height: `${tile.height}%`, background: tileColor(result.gainPct) }} title={`${tile.holding.symbol}: ${fmtMoney(result.marketValue ?? result.cost)}, ${fmtPct(result.gainPct)}`}><div><p className="font-semibold text-white" style={{ fontSize: tile.width > 25 && tile.height > 25 ? "1.5rem" : ".85rem" }}>{tile.holding.symbol}</p>{tile.width > 12 && tile.height > 12 ? <p className="text-xs tabular-nums text-white/85">{fmtPct(result.gainPct)}</p> : null}</div></div>; })}</div></div></div> : null}
        <p className="px-1 text-[11px] leading-relaxed text-[color:var(--ink-faint)]">Your private sync link is the key to this workspace: anyone with it can access the portfolios, and it cannot be recovered if every paired device and saved copy is lost. Quotes are delayed or latest-available observations from Yahoo Finance and may not reflect executable prices. Live mode refreshes visible, open-market portfolios every five minutes and pauses when the tab is hidden or all quotes report a closed market. Gains are hypothetical unrealized estimates before fees, taxes, dividends, splits, and currency effects.</p>
      </> : <div className="panel p-10 text-center"><p>Create a portfolio to begin.</p><button onClick={() => setShowCreate(true)} className="mt-3 rounded-lg bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950">New portfolio</button></div>}
    </section>
    {undo ? <div role="status" aria-live="polite" className="fixed bottom-5 left-1/2 z-50 flex -translate-x-1/2 items-center gap-4 rounded-xl border border-[color:var(--line-strong)] bg-[#0b1a2a] px-4 py-3 text-sm shadow-2xl"><span>{undo.message}</span><button onClick={() => { undo.restore(); setUndo(null); }} className="font-semibold text-cyan-300">Undo</button><button onClick={() => setUndo(null)} aria-label="Dismiss" className="text-[color:var(--ink-faint)]">×</button></div> : null}
  </div>;
}
// End portfolio workspace.
