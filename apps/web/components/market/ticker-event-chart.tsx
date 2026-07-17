"use client";

import { useEffect, useState } from "react";
import type { TickerEventsData } from "@/lib/server/types";

// SEC-51: price line annotated with every event layer the app already
// ingests - 8-K/Form 4 filings, earnings dates (with beat/miss outcomes),
// and Reddit attention (bottom strip). Hand-rolled SVG per repo convention.

const W = 660;
const H = 220;
const PAD_X = 42;
const PAD_TOP = 16;
const PAD_BOTTOM = 10;
const ATTN_H = 34;

function nearestX(candles: { t: number; c: number }[], targetSec: number): number {
  if (candles.length === 0) return 0;
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < candles.length; i++) {
    const d = Math.abs(candles[i]!.t - targetSec);
    if (d < bestDist) { bestDist = d; best = i; }
  }
  return best;
}

export function TickerEventChart({ ticker }: { ticker: string }) {
  const [data, setData] = useState<TickerEventsData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    fetch(`/api/market/ticker-events?ticker=${encodeURIComponent(ticker)}`)
      .then((r) => r.json())
      .then((env) => {
        if (cancelled) return;
        if (env.ok && env.data) setData(env.data);
        else setError(env.error ?? "Failed to load chart");
      })
      .catch(() => { if (!cancelled) setError("Network error"); });
    return () => { cancelled = true; };
  }, [ticker]);

  if (error) return <p className="px-4 py-3 text-xs text-red-400">{error}</p>;
  if (!data) return <p className="px-4 py-3 text-xs text-[color:var(--ink-faint)]">Loading {ticker} chart…</p>;
  if (data.candles.length < 2) return <p className="px-4 py-3 text-xs text-[color:var(--ink-faint)]">Not enough price history.</p>;

  const candles = data.candles;
  const lo = Math.min(...candles.map((p) => p.c));
  const hi = Math.max(...candles.map((p) => p.c));
  const span = hi - lo || 1;
  const x = (i: number) => PAD_X + (i / (candles.length - 1)) * (W - 2 * PAD_X);
  const y = (v: number) => PAD_TOP + (1 - (v - lo) / span) * (H - PAD_TOP - PAD_BOTTOM);
  const path = candles.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.c).toFixed(1)}`).join(" ");

  const dateToX = (iso: string) => x(nearestX(candles, Date.parse(iso.slice(0, 10) + "T16:00:00Z") / 1000));
  const dateToY = (iso: string) => y(candles[nearestX(candles, Date.parse(iso.slice(0, 10) + "T16:00:00Z") / 1000)]!.c);

  const maxMentions = Math.max(...data.attention.map((a) => a.mentions), 1);
  const firstDate = new Date(candles[0]!.t * 1000).toLocaleDateString("en-US", { month: "short", day: "numeric" });
  const lastDate = new Date(candles[candles.length - 1]!.t * 1000).toLocaleDateString("en-US", { month: "short", day: "numeric" });

  return (
    <div className="px-4 py-3">
      <svg viewBox={`0 0 ${W} ${H + ATTN_H}`} className="block w-full" role="img" aria-label={`${ticker} price with filings, earnings and attention events`}>
        {[0.25, 0.5, 0.75].map((f) => (
          <line key={f} x1={PAD_X} x2={W - PAD_X} y1={PAD_TOP + f * (H - PAD_TOP - PAD_BOTTOM)} y2={PAD_TOP + f * (H - PAD_TOP - PAD_BOTTOM)} stroke="var(--line)" strokeWidth={1} />
        ))}
        <text x={PAD_X - 6} y={y(hi) + 4} fontSize={9} textAnchor="end" fill="var(--ink-faint)">${hi.toFixed(hi >= 100 ? 0 : 2)}</text>
        <text x={PAD_X - 6} y={y(lo) + 4} fontSize={9} textAnchor="end" fill="var(--ink-faint)">${lo.toFixed(lo >= 100 ? 0 : 2)}</text>

        {/* earnings verticals first so markers draw on top */}
        {data.earnings.map((e) => {
          const ex = dateToX(e.date);
          const color = e.outcome === "beat" ? "#41d39d" : e.outcome === "miss" ? "#f87171" : "var(--ink-faint)";
          return (
            <g key={`e${e.date}`}>
              <title>{`Earnings ${e.date}${e.outcome ? ` — ${e.outcome}` : " (upcoming)"}`}</title>
              <line x1={ex} x2={ex} y1={PAD_TOP} y2={H - PAD_BOTTOM} stroke={color} strokeWidth={1} strokeDasharray="4 4" opacity={0.7} />
              <circle cx={ex} cy={PAD_TOP + 3} r={4} fill={e.resolved ? color : "var(--bg-elev-strong, #0b1826)"} stroke={color} strokeWidth={1.5} />
            </g>
          );
        })}

        <path d={path} fill="none" stroke="#4fd5ff" strokeWidth={1.8} strokeLinejoin="round" />

        {/* filing markers on the price line */}
        {data.filings.map((f, i) => {
          const fx = dateToX(f.filedAt);
          const fy = dateToY(f.filedAt);
          const isEightK = f.form === "8-K";
          const buy = /bought/i.test(f.label);
          const sell = /sold/i.test(f.label);
          const color = isEightK ? "#4fd5ff" : buy ? "#41d39d" : sell ? "#f87171" : "#a78bfa";
          return (
            <g key={`f${i}`}>
              <title>{`${f.form} ${f.filedAt.slice(0, 10)} — ${f.label}`}</title>
              {isEightK ? (
                <rect x={fx - 4} y={fy - 12 - 4} width={8} height={8} transform={`rotate(45 ${fx} ${fy - 12})`} fill={color} />
              ) : (
                <path
                  d={sell ? `M${fx - 5},${fy + 8} L${fx + 5},${fy + 8} L${fx},${fy + 16} Z` : `M${fx - 5},${fy - 16} L${fx + 5},${fy - 16} L${fx},${fy - 8} Z`}
                  fill={color}
                />
              )}
            </g>
          );
        })}

        {/* attention strip */}
        {data.attention.map((a) => {
          const ax = dateToX(a.date);
          const h = (a.mentions / maxMentions) * (ATTN_H - 6);
          return (
            <g key={`a${a.date}`}>
              <title>{`${a.date}: ${a.mentions} mentions`}</title>
              <rect x={ax - 1.5} y={H + ATTN_H - 2 - h} width={3} height={Math.max(h, 1)} fill="#fbbf24" opacity={0.35 + 0.6 * (a.mentions / maxMentions)} />
            </g>
          );
        })}
        <text x={PAD_X} y={H + ATTN_H - 2} fontSize={8} fill="var(--ink-faint)">{firstDate}</text>
        <text x={W - PAD_X} y={H + ATTN_H - 2} fontSize={8} textAnchor="end" fill="var(--ink-faint)">{lastDate}</text>
      </svg>
      <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[10px] text-[color:var(--ink-faint)]">
        <span><span className="mr-1 inline-block h-2 w-2 rotate-45" style={{ backgroundColor: "#4fd5ff" }} />8-K</span>
        <span><span className="mr-1 inline-block" style={{ width: 0, height: 0, borderLeft: "4px solid transparent", borderRight: "4px solid transparent", borderTop: "7px solid #f87171" }} />insider sell</span>
        <span><span className="mr-1 inline-block" style={{ width: 0, height: 0, borderLeft: "4px solid transparent", borderRight: "4px solid transparent", borderBottom: "7px solid #41d39d" }} />insider buy</span>
        <span className="text-[color:var(--ink-faint)]">┆ earnings (dot = beat/miss)</span>
        <span><span className="mr-1 inline-block h-2 w-2" style={{ backgroundColor: "#fbbf24", opacity: 0.7 }} />Reddit attention</span>
        {data.warning && <span className="text-amber-300">{data.warning}</span>}
      </div>
    </div>
  );
}
