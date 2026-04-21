"use client";

import { useEffect, useId, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type ChartPoint = { t: number; c: number };
type ChartRange = "1W" | "1M" | "3M" | "1Y";

const RANGES: ChartRange[] = ["1W", "1M", "3M", "1Y"];
const CUTOFF: Record<ChartRange, number> = { "1W": 7, "1M": 30, "3M": 90, "1Y": 365 };

// Module-level cache so data persists across component mounts
const chartCache = new Map<string, ChartPoint[]>();

function filterRange(prices: ChartPoint[], range: ChartRange): ChartPoint[] {
  const cutoff = Date.now() / 1000 - CUTOFF[range] * 86400;
  return prices.filter((p) => p.t >= cutoff);
}

function fmtDate(ts: number, range: ChartRange): string {
  const d = new Date(ts * 1000);
  if (range === "1Y") return d.toLocaleDateString("en-US", { month: "short" });
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function fmtAxis(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 10000) return `${(n / 1000).toFixed(0)}k`;
  if (abs >= 1000)  return `${(n / 1000).toFixed(1)}k`;
  if (abs >= 10)    return n.toFixed(0);
  if (abs >= 1)     return n.toFixed(2);
  return n.toFixed(4);
}

function fmtTooltip(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 1000) return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (abs >= 1)    return n.toFixed(2);
  return n.toFixed(4);
}

function TooltipContent({ active, payload }: { active?: boolean; payload?: { value: number; payload: ChartPoint }[] }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  const date = new Date(p.t * 1000).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  return (
    <div className="rounded-lg border border-[color:rgba(79,213,255,0.2)] bg-[color:rgba(9,21,34,0.96)] px-3 py-2 text-xs shadow-xl backdrop-blur-sm">
      <div className="text-[color:var(--ink-faint)] mb-0.5">{date}</div>
      <div className="font-bold tabular-nums text-white">{fmtTooltip(payload[0].value)}</div>
    </div>
  );
}

export function PriceChart({
  prices,
  up,
  defaultRange = "3M",
  label,
}: {
  prices: ChartPoint[];
  up: boolean;
  defaultRange?: ChartRange;
  label?: string;
}) {
  const [range, setRange] = useState<ChartRange>(defaultRange);
  const uid = useId().replace(/:/g, "");
  const gradId = `cg-${uid}`;
  const color = up ? "#41d39d" : "#f87171";
  const filtered = filterRange(prices, range);

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        {label ? (
          <span className="rounded border border-[color:rgba(79,213,255,0.2)] bg-[color:rgba(79,213,255,0.08)] px-2 py-0.5 text-[10px] font-bold tracking-wider text-[color:var(--accent)]">
            {label}
          </span>
        ) : <span />}
        <div className="flex gap-0.5">
        {RANGES.map((r) => (
          <button
            key={r}
            type="button"
            onClick={() => setRange(r)}
            className={`rounded-lg px-2.5 py-0.5 text-[11px] font-medium transition-colors ${
              range === r
                ? "bg-[color:rgba(79,213,255,0.18)] text-[color:var(--ink)]"
                : "text-[color:var(--ink-faint)] hover:text-[color:var(--ink)]"
            }`}
          >
            {r}
          </button>
        ))}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={filtered} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor={color} stopOpacity={0.25} />
              <stop offset="95%" stopColor={color} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
          <XAxis
            dataKey="t"
            tickFormatter={(ts) => fmtDate(ts, range)}
            tick={{ fontSize: 10, fill: "rgba(255,255,255,0.3)" }}
            tickLine={false}
            axisLine={false}
            interval="preserveStartEnd"
            minTickGap={40}
          />
          <YAxis
            orientation="right"
            tickFormatter={fmtAxis}
            tick={{ fontSize: 10, fill: "rgba(255,255,255,0.3)" }}
            tickLine={false}
            axisLine={false}
            domain={["auto", "auto"]}
            width={52}
          />
          <Tooltip content={<TooltipContent />} cursor={{ stroke: color, strokeWidth: 1, strokeOpacity: 0.35 }} />
          <Area
            type="monotone"
            dataKey="c"
            stroke={color}
            strokeWidth={1.5}
            fill={`url(#${gradId})`}
            dot={false}
            activeDot={{ r: 3, fill: color, stroke: "none" }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function InlineChart({
  symbol,
  type = "yahoo",
  name,
  up,
  label,
}: {
  symbol: string;
  type?: "yahoo" | "crypto";
  name: string;
  up: boolean;
  label?: string;
}) {
  const [prices, setPrices] = useState<ChartPoint[] | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const key = `${type}:${symbol}`;
    if (chartCache.has(key)) {
      setPrices(chartCache.get(key)!);
      setLoading(false);
      return;
    }
    setLoading(true);
    const qs = type === "crypto"
      ? `id=${encodeURIComponent(symbol)}&type=crypto`
      : `symbol=${encodeURIComponent(symbol)}&type=yahoo`;
    fetch(`/api/market/chart?${qs}`)
      .then((r) => r.json())
      .then((env) => {
        if (env.ok && env.data?.prices) {
          chartCache.set(key, env.data.prices);
          setPrices(env.data.prices);
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [symbol, type]);

  if (loading) {
    return (
      <div className="h-52 flex items-center justify-center text-xs text-[color:var(--ink-faint)]">
        Loading {name} chart…
      </div>
    );
  }
  if (!prices?.length) {
    return (
      <div className="h-16 flex items-center justify-center text-xs text-[color:var(--ink-faint)]">
        No chart data available
      </div>
    );
  }
  return <PriceChart prices={prices} up={up} label={label} />;
}
