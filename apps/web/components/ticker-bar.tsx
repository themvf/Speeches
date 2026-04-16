"use client";

import { useEffect, useState } from "react";

type Quote = {
  symbol: string;
  name: string;
  price: number;
  change: number;
  pct: number;
  up: boolean;
};

export function TickerBar() {
  const [quotes, setQuotes] = useState<Quote[]>([]);

  async function fetchQuotes() {
    try {
      const res = await fetch("/api/market");
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data)) setQuotes(data);
      }
    } catch {
      // non-critical — silently skip
    }
  }

  useEffect(() => {
    fetchQuotes();
    const id = setInterval(fetchQuotes, 60_000);
    return () => clearInterval(id);
  }, []);

  const valid = quotes.filter((q) => q.price);
  if (valid.length === 0) return null;

  // Duplicate items so the scroll loops seamlessly
  const items = [...valid, ...valid];

  return (
    <div className="ticker-bar fixed bottom-0 left-0 right-0 z-40 overflow-hidden border-t border-[color:var(--line-soft)] bg-[color:rgba(4,11,20,0.9)] backdrop-blur-sm">
      <div className="flex h-8 items-center">
        {/* LIVE badge */}
        <div className="flex h-full flex-shrink-0 items-center gap-1.5 border-r border-[color:var(--line)] px-3">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[color:var(--ok)]" />
          <span className="text-[10px] font-bold uppercase tracking-[0.1em] text-[color:var(--ok)]">
            Live
          </span>
        </div>

        {/* Scrolling track */}
        <div className="relative flex-1 overflow-hidden">
          <div className="ticker-track flex w-max items-center gap-7 whitespace-nowrap">
            {items.map((q, i) => (
              <span
                key={`${q.symbol}-${i}`}
                className="flex items-center gap-1.5 text-[11px]"
              >
                <span className="font-mono font-bold text-[color:var(--accent)]">{q.symbol}</span>
                {q.name && q.name !== q.symbol && (
                  <span className="text-[10px] text-[color:var(--ink-faint)]">{q.name}</span>
                )}
                <span className="font-mono font-bold text-[color:var(--ink)]">
                  {(q.price ?? 0).toLocaleString("en-US", {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </span>
                <span
                  className={`font-mono text-[10px] font-semibold ${
                    q.up ? "text-[color:var(--ok)]" : "text-[color:var(--danger)]"
                  }`}
                >
                  {q.up ? "▲" : "▼"} {Math.abs(q.change ?? 0).toFixed(2)} (
                  {Math.abs(q.pct ?? 0).toFixed(2)}%)
                </span>
                <span className="text-[color:var(--line-strong)]" aria-hidden>
                  ·
                </span>
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
