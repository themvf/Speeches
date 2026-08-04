"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { MarketIndustriesData, TickerSearchEntry } from "@/lib/server/types";
import { TickerEventChart } from "./ticker-event-chart";

const MAX_RESULTS = 8;

// Market-page global search: ticker or company name, over the same tracked
// ~900-company universe the Industries tab and CBOE fundamentals lookup
// already use (industry-config.json). Persistent above the tab bar so it
// works no matter which tab is active. Selecting a result shows a quote +
// event-annotated chart inline, with jump-off buttons into the two tabs
// that already have deeper views for a single company.
export function TickerSearch({
  onViewIndustry,
  onViewFundamentals,
}: {
  onViewIndustry: (industry: string) => void;
  onViewFundamentals: (ticker: string) => void;
}) {
  const [index, setIndex] = useState<TickerSearchEntry[] | null>(null);
  const [indexError, setIndexError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const [selected, setSelected] = useState<TickerSearchEntry | null>(null);
  const [result, setResult] = useState<MarketIndustriesData | null>(null);
  const [resultLoading, setResultLoading] = useState(false);
  const [resultError, setResultError] = useState<string | null>(null);
  const loadedRef = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Lazy-fetch the index on first focus, matching this page's existing
  // "load on demand" convention (PeerTable, FundamentalsLookup) rather than
  // paying for it on every /market page load.
  function ensureIndexLoaded() {
    if (loadedRef.current) return;
    loadedRef.current = true;
    fetch("/api/market/search-index")
      .then((r) => r.json())
      .then((env) => {
        if (env.ok && env.data) setIndex(env.data.entries);
        else setIndexError(env.error ?? "Failed to load search index");
      })
      .catch(() => setIndexError("Network error"));
  }

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  const matches = useMemo(() => {
    const q = query.trim().toUpperCase();
    if (!q || !index) return [];
    const tickerHits: TickerSearchEntry[] = [];
    const nameHits: TickerSearchEntry[] = [];
    for (const entry of index) {
      if (entry.ticker === q) { tickerHits.unshift(entry); continue; }
      if (entry.ticker.startsWith(q)) { tickerHits.push(entry); continue; }
      if (entry.name.toUpperCase().includes(q)) nameHits.push(entry);
    }
    return [...tickerHits, ...nameHits].slice(0, MAX_RESULTS);
  }, [query, index]);

  function selectEntry(entry: TickerSearchEntry) {
    setSelected(entry);
    setQuery(entry.ticker);
    setOpen(false);
    setResult(null);
    setResultError(null);
    setResultLoading(true);
    fetch(`/api/market/industries?ticker=${encodeURIComponent(entry.ticker)}`)
      .then((r) => r.json())
      .then((env) => {
        if (env.ok && env.data) setResult(env.data);
        else setResultError(env.error ?? "Lookup failed");
      })
      .catch(() => setResultError("Network error"))
      .finally(() => setResultLoading(false));
  }

  function clear() {
    setSelected(null);
    setResult(null);
    setResultError(null);
    setQuery("");
  }

  const tickerResult = result?.tickerResult;

  return (
    <div ref={containerRef} className="relative">
      <div className="relative w-full max-w-sm">
        <input
          value={query}
          onFocus={() => { ensureIndexLoaded(); setOpen(true); }}
          onChange={(e) => { setQuery(e.target.value); setSelected(null); setOpen(true); setHighlight(0); }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setHighlight((h) => Math.min(h + 1, matches.length - 1)); }
            else if (e.key === "ArrowUp") { e.preventDefault(); setHighlight((h) => Math.max(h - 1, 0)); }
            else if (e.key === "Enter") { e.preventDefault(); const m = matches[highlight]; if (m) selectEntry(m); }
            else if (e.key === "Escape") setOpen(false);
          }}
          placeholder="Search ticker or company name…"
          aria-label="Search ticker or company name"
          className="w-full rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)] px-3 py-2 text-sm text-[color:var(--ink)] placeholder:text-[color:var(--ink-faint)]"
        />
        {selected && (
          <button
            type="button"
            onClick={clear}
            aria-label="Clear search"
            className="absolute right-2 top-1/2 -translate-y-1/2 text-[color:var(--ink-faint)] hover:text-[color:var(--ink)]"
          >
            ✕
          </button>
        )}
        {open && query.trim() && !selected && (
          <div className="absolute z-20 mt-1 w-full overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.98)] shadow-lg">
            {indexError && <p className="px-3 py-2 text-xs text-red-400">{indexError}</p>}
            {!indexError && !index && <p className="px-3 py-2 text-xs text-[color:var(--ink-faint)]">Loading…</p>}
            {index && matches.length === 0 && (
              <p className="px-3 py-2 text-xs text-[color:var(--ink-faint)]">No matches in the tracked ~{index.length}-company universe.</p>
            )}
            {matches.map((m, i) => (
              <button
                key={m.ticker}
                type="button"
                onMouseDown={(e) => { e.preventDefault(); selectEntry(m); }}
                onMouseEnter={() => setHighlight(i)}
                className={`flex w-full items-center justify-between gap-3 px-3 py-1.5 text-left text-xs ${
                  i === highlight ? "bg-[rgba(79,213,255,0.12)]" : ""
                }`}
              >
                <span>
                  <span className="font-bold text-[color:var(--accent)]">{m.ticker}</span>{" "}
                  <span className="text-[color:var(--ink-faint)]">{m.name}</span>
                </span>
                <span className="shrink-0 text-[10px] text-[color:var(--ink-faint)]">{m.industry}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {selected && (
        <div className="mt-3 rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
          {resultLoading && <p className="px-4 py-3 text-xs text-[color:var(--ink-faint)]">Loading {selected.ticker}…</p>}
          {resultError && <p className="px-4 py-3 text-xs text-red-400">{resultError}</p>}
          {!resultLoading && !resultError && tickerResult === null && (
            <p className="px-4 py-3 text-xs text-[color:var(--ink-faint)]">
              {selected.ticker} isn&apos;t in the tracked industry universe.
            </p>
          )}
          {tickerResult && (
            <>
              <div className="flex flex-wrap items-center justify-between gap-2 px-4 pt-3">
                <div className="flex items-baseline gap-2">
                  <span className="text-sm font-bold text-[color:var(--accent)]">{tickerResult.row.ticker}</span>
                  <span className="text-sm text-[color:var(--ink)]">{tickerResult.row.name}</span>
                  <span className="text-[10px] text-[color:var(--ink-faint)]">{tickerResult.industry}</span>
                </div>
                {tickerResult.row.price != null && (
                  <div className="flex items-baseline gap-2 tabular-nums">
                    <span className="text-sm font-semibold text-[color:var(--ink)]">
                      ${tickerResult.row.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </span>
                    {tickerResult.row.pricePct != null && (
                      <span
                        className="text-xs font-semibold"
                        style={{ color: tickerResult.row.pricePct >= 0 ? "#41d39d" : "#f87171" }}
                      >
                        {tickerResult.row.pricePct >= 0 ? "+" : ""}{tickerResult.row.pricePct.toFixed(2)}%
                      </span>
                    )}
                  </div>
                )}
              </div>
              <div className="flex flex-wrap gap-2 px-4 pt-2">
                <button
                  type="button"
                  onClick={() => onViewIndustry(tickerResult.industry)}
                  className="rounded-lg border border-[color:var(--line)] px-2.5 py-1 text-[11px] font-medium text-[color:var(--ink-soft)] hover:border-[color:rgba(79,213,255,0.35)] hover:text-[color:var(--accent)]"
                >
                  View industry peers →
                </button>
                <button
                  type="button"
                  onClick={() => onViewFundamentals(tickerResult.row.ticker)}
                  className="rounded-lg border border-[color:var(--line)] px-2.5 py-1 text-[11px] font-medium text-[color:var(--ink-soft)] hover:border-[color:rgba(79,213,255,0.35)] hover:text-[color:var(--accent)]"
                >
                  Full fundamentals →
                </button>
              </div>
              <TickerEventChart ticker={tickerResult.row.ticker} />
            </>
          )}
        </div>
      )}
    </div>
  );
}
