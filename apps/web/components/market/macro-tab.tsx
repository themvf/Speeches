"use client";

import type {
  MarketBondsData,
  MarketMacroCalendarData,
  MarketMacroData,
  MarketMacroGroup,
  MarketMacroIndicator,
  MarketMacroIndicatorId,
  MarketMacroPredictionsData,
  MarketMacroPoint,
  MarketRatesCreditData,
  MacroPredictionEvent,
} from "@/lib/server/types";
import type { RateTransmissionData } from "@/lib/rate-transmission";
import { signalFor } from "@/lib/macro-signals";
import { percentileContext } from "@/lib/macro-context";
import { MacroConditions } from "./macro-conditions";
import { MacroPredictionInline } from "./macro-prediction-panel";
import { MacroCalendar } from "./macro-calendar";
import { YieldCurve } from "./yield-curve";
import {
  daysUntil,
  formatCalendarDate,
  formatReleaseTime,
  localIsoDate,
  nextReleaseByIndicator,
  relativeDayLabel,
  type NextRelease,
} from "@/lib/macro-calendar-display";
import { RatesCreditSection } from "./rates-credit-section";
import { RateTransmissionSection } from "./rate-transmission";

interface Props {
  data: MarketMacroData | null;
  loading: boolean;
  error: string | null;
  predictions: {
    data: MarketMacroPredictionsData | null;
    loading: boolean;
    error: string | null;
  };
  calendar: {
    data: MarketMacroCalendarData | null;
    loading: boolean;
    error: string | null;
  };
  bonds: {
    data: MarketBondsData | null;
    loading: boolean;
    error: string | null;
  };
  ratesCredit: {
    data: MarketRatesCreditData | null;
    loading: boolean;
    error: string | null;
  };
  rateTransmission: {
    data: RateTransmissionData | null;
    loading: boolean;
    error: string | null;
  };
}

const GROUPS: Array<{ id: Exclude<MarketMacroGroup, "headline">; label: string; description: string }> = [
  { id: "activity", label: "Leading Activity", description: "Early reads on household demand and real-economy output." },
  { id: "inflation", label: "Inflation & Expectations", description: "Underlying prices, producer pressure, and market-implied inflation." },
  { id: "labor", label: "Labor Detail", description: "Layoffs, wages, participation, openings, and recession risk." },
  { id: "financial", label: "Financial Conditions & Liquidity", description: "The Treasury curve, credit spreads, real rates, funding, liquidity, and the dollar." },
  { id: "housing", label: "Housing", description: "Construction pipeline and household borrowing costs." },
];

function formatValue(indicator: MarketMacroIndicator, value = indicator.value): string {
  if (indicator.unit === "thousands") return `${value >= 0 ? "+" : ""}${Math.round(value).toLocaleString("en-US")}K`;
  if (indicator.unit === "thousands_level") {
    return Math.abs(value) >= 1_000 ? `${(value / 1_000).toFixed(2)}M` : `${Math.round(value).toLocaleString("en-US")}K`;
  }
  if (indicator.unit === "trillions") return `$${value.toFixed(2)}T`;
  if (indicator.unit === "index") return value.toFixed(2);
  return `${value.toFixed(2)}%`;
}

function formatChange(indicator: MarketMacroIndicator): string {
  if (indicator.change === null) return "No prior observation";
  const sign = indicator.change >= 0 ? "+" : "";
  if (indicator.unit === "thousands" || indicator.unit === "thousands_level") {
    return `${sign}${Math.round(indicator.change).toLocaleString("en-US")}K vs prior`;
  }
  if (indicator.unit === "trillions") return `${sign}$${indicator.change.toFixed(2)}T vs prior`;
  if (indicator.unit === "index") return `${sign}${indicator.change.toFixed(2)} vs prior`;
  return `${sign}${indicator.change.toFixed(2)} pp vs prior`;
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

function NextReleaseLine({ next, today }: { next: NextRelease | undefined; today: string }) {
  if (!next) return null;
  const formatted = formatCalendarDate(next.date, { month: "short", day: "numeric" });
  const time = formatReleaseTime(next.timeEt);
  return (
    <p className="mt-2 text-[10px] text-[color:var(--ink-faint)]">
      Next release <span className="font-semibold text-[color:var(--ink-soft)]">{formatted}</span>
      {time ? ` at ${time}` : ""}
      {" · "}
      {relativeDayLabel(daysUntil(next.date, today)).toLowerCase()}
    </p>
  );
}

function MacroCard({ indicator, contracts, nextRelease, today }: {
  indicator: MarketMacroIndicator;
  contracts: MacroPredictionEvent[];
  nextRelease: NextRelease | undefined;
  today: string;
}) {
  const signal = signalFor(indicator);
  const context = percentileContext(indicator);
  return (
    <article className="flex min-h-[280px] flex-col rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.5)] p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">{indicator.label}</p>
          <p className="mt-1 text-[10px] text-[color:var(--ink-faint)]">{indicator.frequency} · {indicator.fredSeriesId}</p>
        </div>
        <span className={`rounded-full border px-2 py-1 text-[9px] font-semibold ${
          signal.alert
            ? "border-red-400/30 bg-red-400/10 text-red-300"
            : "border-[color:rgba(79,213,255,0.28)] bg-[color:rgba(79,213,255,0.08)] text-[color:var(--accent)]"
        }`}>
          {signal.text}
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

      {context && (
        <p className="mt-2 text-[10px] text-[color:var(--ink-faint)]" title={`Percentile of ${context.sampleSize} observations ${context.window}`}>
          {context.summary}
        </p>
      )}

      <div className="mt-3"><MacroSparkline points={indicator.points} /></div>
      <p className="mt-2 flex-1 text-xs leading-5 text-[color:var(--ink-faint)]">{indicator.description}</p>
      <NextReleaseLine next={nextRelease} today={today} />
      <MacroPredictionInline events={contracts} />
      <div className="mt-3 flex items-center justify-between gap-3 border-t border-[color:var(--line)] pt-3 text-[10px] text-[color:var(--ink-faint)]">
        <span>{indicator.lastUpdated ? `FRED updated ${formatFredUpdatedAt(indicator.lastUpdated)}` : "FRED update time unavailable"}</span>
        <a href={indicator.sourceUrl} target="_blank" rel="noreferrer" className="font-semibold text-[color:var(--accent)] hover:underline">Source</a>
      </div>
    </article>
  );
}

function IndicatorGrid({ indicators, contracts, nextReleases, today }: {
  indicators: MarketMacroIndicator[];
  contracts: MacroPredictionEvent[];
  nextReleases: Map<MarketMacroIndicatorId, NextRelease>;
  today: string;
}) {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
      {[...indicators]
        .sort((left, right) => left.priority - right.priority)
        .map((indicator) => (
          <MacroCard
            key={indicator.id}
            indicator={indicator}
            contracts={contracts.filter((contract) => contract.indicatorIds.includes(indicator.id))}
            nextRelease={nextReleases.get(indicator.id)}
            today={today}
          />
        ))}
    </div>
  );
}

export function MacroTab({ data, loading, error, predictions, calendar, bonds, ratesCredit, rateTransmission }: Props) {
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

  const headline = data.indicators.filter((indicator) => indicator.group === "headline");
  const contracts = predictions.data?.events ?? [];
  const today = localIsoDate(new Date());
  // The curve renders inside the financial group so shape, credit and the real
  // yield are read together. Its 10Y-2Y figure is FRED's T10Y2Y - the same
  // number the card below shows - rather than one recomputed from Treasury XML.
  const curveYields = bonds.data?.yields ?? [];
  const curveSpread = data.indicators.find((indicator) => indicator.id === "yield_curve_10y2y")?.value ?? null;
  const nextReleases = nextReleaseByIndicator(calendar.data?.entries ?? [], today);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">US Macro Dashboard</p>
          <p className="mt-1 max-w-3xl text-xs leading-5 text-[color:var(--ink-faint)]">
            Headline conditions plus expandable views of activity, inflation, labor, liquidity, and housing. Values are cached server-side and may be revised by their originating agencies.
          </p>
        </div>
        <p className="text-[10px] text-[color:var(--ink-faint)]">
          Generated {new Date(data.generatedAt).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })} · {Math.round(data.cacheSeconds / 60)} min cache
        </p>
      </div>

      {predictions.error && !predictions.data && (
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-300">Market expectations are temporarily unavailable: {predictions.error}</div>
      )}

      <RatesCreditSection {...ratesCredit} />

      <MacroConditions indicators={data.indicators} />

      <MacroCalendar {...calendar} contracts={contracts} />

      <IndicatorGrid indicators={headline} contracts={contracts} nextReleases={nextReleases} today={today} />

      <div className="space-y-3">
        {GROUPS.map((group) => {
          const indicators = data.indicators.filter((indicator) => indicator.group === group.id);
          if (!indicators.length) return null;
          return (
            <div key={group.id} className="contents">
            <details className="group rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.3)]">
              <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-4 py-4 [&::-webkit-details-marker]:hidden">
                <span>
                  <span className="text-sm font-semibold text-[color:var(--ink)]">{group.label}</span>
                  <span className="ml-2 text-xs text-[color:var(--ink-faint)]">{group.description}</span>
                </span>
                <span className="flex shrink-0 items-center gap-2 text-xs text-[color:var(--ink-faint)]">
                  {indicators.length} indicators
                  <span aria-hidden="true" className="text-base transition-transform group-open:rotate-180">⌄</span>
                </span>
              </summary>
              <div className="space-y-4 border-t border-[color:var(--line)] p-4">
                {group.id === "financial" && curveYields.length >= 3 && (
                  <YieldCurve yields={curveYields} spread={curveSpread} spreadLabel="FRED T10Y2Y" />
                )}
                <IndicatorGrid indicators={indicators} contracts={contracts} nextReleases={nextReleases} today={today} />
              </div>
            </details>
            {group.id === "financial" && <RateTransmissionSection {...rateTransmission} />}
            </div>
          );
        })}
      </div>

      <p className="text-right text-[10px] text-[color:var(--ink-faint)]">
        Source: Federal Reserve Bank of St. Louis FRED® · Release data can arrive after the scheduled publication time.
      </p>
    </div>
  );
}
