"use client";

import type {
  MarketMacroData,
  MarketMacroIndicator,
  MarketMacroIndicatorId,
  MarketMacroPoint,
} from "@/lib/server/types";

interface Props {
  data: MarketMacroData | null;
  loading: boolean;
  error: string | null;
}

const SIGNALS: Record<MarketMacroIndicatorId, (indicator: MarketMacroIndicator) => string> = {
  real_gdp_growth: (indicator) => (indicator.change ?? 0) >= 0 ? "Growth accelerating" : "Growth slowing",
  cpi_inflation: (indicator) => (indicator.change ?? 0) <= 0 ? "Inflation cooling" : "Inflation heating",
  nonfarm_payrolls: (indicator) => indicator.value >= 0 ? "Jobs added" : "Jobs lost",
  unemployment_rate: (indicator) => (indicator.change ?? 0) <= 0 ? "Labor market firming" : "Labor market softening",
  effective_fed_funds: (indicator) => (indicator.change ?? 0) === 0 ? "Policy rate steady" : (indicator.change ?? 0) > 0 ? "Policy tightening" : "Policy easing",
  yield_curve_10y2y: (indicator) => indicator.value < 0 ? "Yield curve inverted" : "Yield curve positive",
};

function formatValue(indicator: MarketMacroIndicator, value = indicator.value): string {
  if (indicator.unit === "thousands") return `${value >= 0 ? "+" : ""}${Math.round(value).toLocaleString("en-US")}K`;
  return `${value.toFixed(2)}%`;
}

function formatChange(indicator: MarketMacroIndicator): string {
  if (indicator.change === null) return "No prior observation";
  if (indicator.unit === "thousands") {
    return `${indicator.change >= 0 ? "+" : ""}${Math.round(indicator.change).toLocaleString("en-US")}K vs prior`;
  }
  return `${indicator.change >= 0 ? "+" : ""}${indicator.change.toFixed(2)} pp vs prior`;
}

function formatObservationDate(value: string): string {
  const date = new Date(`${value}T00:00:00Z`);
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
}

function formatFredUpdatedAt(value: string): string {
  const date = new Date(value.replace(" ", "T"));
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

function MacroSparkline({ points }: { points: MarketMacroPoint[] }) {
  const visible = points.slice(-48);
  if (visible.length < 2) return null;
  const width = 240;
  const height = 72;
  const values = visible.map((point) => point.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const coordinates = visible.map((point, index) => {
    const x = (index / (visible.length - 1)) * width;
    const y = height - 6 - ((point.value - min) / range) * (height - 12);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-[72px] w-full" role="img" aria-label="Recent history">
      <line x1="0" y1={height - 1} x2={width} y2={height - 1} stroke="rgba(255,255,255,0.08)" />
      <polyline points={coordinates} fill="none" stroke="#4fd5ff" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

function MacroCard({ indicator }: { indicator: MarketMacroIndicator }) {
  const signal = SIGNALS[indicator.id](indicator);
  const inverted = indicator.id === "yield_curve_10y2y" && indicator.value < 0;
  return (
    <article className="flex min-h-[280px] flex-col rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.5)] p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">{indicator.label}</p>
          <p className="mt-1 text-[10px] text-[color:var(--ink-faint)]">{indicator.frequency} · {indicator.fredSeriesId}</p>
        </div>
        <span className={`rounded-full border px-2 py-1 text-[9px] font-semibold ${
          inverted
            ? "border-red-400/30 bg-red-400/10 text-red-300"
            : "border-[color:rgba(79,213,255,0.28)] bg-[color:rgba(79,213,255,0.08)] text-[color:var(--accent)]"
        }`}>
          {signal}
        </span>
      </div>

      <div className="mt-4 flex items-end justify-between gap-3">
        <div>
          <p className="text-3xl font-bold tabular-nums text-[color:var(--ink)]">{formatValue(indicator)}</p>
          <p className="mt-1 text-xs tabular-nums text-[color:var(--ink-faint)]">{formatChange(indicator)}</p>
        </div>
        <p className="text-right text-[10px] text-[color:var(--ink-faint)]">
          Observation<br />{formatObservationDate(indicator.observationDate)}
        </p>
      </div>

      <div className="mt-3"><MacroSparkline points={indicator.points} /></div>
      <p className="mt-2 flex-1 text-xs leading-5 text-[color:var(--ink-faint)]">{indicator.description}</p>
      <div className="mt-3 flex items-center justify-between gap-3 border-t border-[color:var(--line)] pt-3 text-[10px] text-[color:var(--ink-faint)]">
        <span>{indicator.lastUpdated ? `FRED updated ${formatFredUpdatedAt(indicator.lastUpdated)}` : "FRED update time unavailable"}</span>
        <a href={indicator.sourceUrl} target="_blank" rel="noreferrer" className="font-semibold text-[color:var(--accent)] hover:underline">Source</a>
      </div>
    </article>
  );
}

export function MacroTab({ data, loading, error }: Props) {
  if (loading && !data) {
    return <div className="flex items-center justify-center py-16 text-sm text-[color:var(--ink-faint)]">Loading FRED macro indicators…</div>;
  }
  if (error && !data) {
    return (
      <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-300">
        <p className="font-semibold">Macro data is unavailable</p>
        <p className="mt-1 text-xs text-red-200/80">{error}</p>
      </div>
    );
  }
  if (!data) return null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">US Macro Dashboard</p>
          <p className="mt-1 max-w-2xl text-xs leading-5 text-[color:var(--ink-faint)]">
            Growth, inflation, employment, policy, and yield-curve conditions. Values are cached server-side and may be revised by their originating agencies.
          </p>
        </div>
        <p className="text-[10px] text-[color:var(--ink-faint)]">
          Generated {new Date(data.generatedAt).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })} · {Math.round(data.cacheSeconds / 60)} min cache
        </p>
      </div>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {data.indicators.map((indicator) => <MacroCard key={indicator.id} indicator={indicator} />)}
      </div>
      <p className="text-right text-[10px] text-[color:var(--ink-faint)]">
        Source: Federal Reserve Bank of St. Louis FRED® · Release data can arrive after the scheduled publication time.
      </p>
    </div>
  );
}
