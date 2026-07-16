"use client";

import type { MacroPredictionEvent, MacroPredictionTheme, MarketMacroPredictionsData } from "@/lib/server/types";

const THEME_LABELS: Record<MacroPredictionTheme, string> = {
  fed_policy: "Fed policy",
  growth: "Growth",
  inflation: "Inflation",
  labor: "Labor",
  recession: "Recession risk",
  housing: "Housing",
};

function probability(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function money(value: number): string {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function change(value: number | null): string | null {
  if (value == null || value === 0) return null;
  const points = value * 100;
  return `${points > 0 ? "+" : ""}${points.toFixed(1)} pp today`;
}

function MatchBadge({ event }: { event: MacroPredictionEvent }) {
  const exact = event.matchKind === "exact_series";
  return (
    <span className={`rounded-full border px-2 py-0.5 text-[9px] font-semibold ${exact ? "border-emerald-400/25 bg-emerald-400/10 text-emerald-300" : "border-amber-400/25 bg-amber-400/10 text-amber-300"}`}>
      {exact ? "Exact series" : "Related signal"}
    </span>
  );
}

export function MacroPredictionInline({ events }: { events: MacroPredictionEvent[] }) {
  if (!events.length) return null;
  const leader = events[0].leadingOutcome;
  return (
    <details className="group mt-3 rounded-lg border border-[color:rgba(79,213,255,0.2)] bg-[color:rgba(79,213,255,0.05)]">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-3 py-2 text-xs [&::-webkit-details-marker]:hidden">
        <span className="font-semibold text-[color:var(--ink-soft)]">Market expectations</span>
        <span className="flex items-center gap-2 text-[color:var(--ink-faint)]">
          {leader && <span>{leader.label} <strong className="text-[color:var(--accent)]">{probability(leader.probability)}</strong></span>}
          <span aria-hidden="true" className="transition-transform group-open:rotate-180">⌄</span>
        </span>
      </summary>
      <div className="space-y-3 border-t border-[color:var(--line)] px-3 py-3">
        {events.map((event) => (
          <div key={event.mappingKey} className="space-y-1.5">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <a href={event.url} target="_blank" rel="noreferrer" className="text-xs font-semibold text-[color:var(--ink)] hover:text-[color:var(--accent)] hover:underline">{event.title} ↗</a>
              <MatchBadge event={event} />
            </div>
            <div className="flex flex-wrap gap-1.5">
              {event.outcomes.filter((outcome) => !outcome.closed).slice(0, 6).map((outcome) => (
                <span key={outcome.marketId} className="rounded bg-[color:rgba(15,32,50,0.8)] px-2 py-1 text-[10px] text-[color:var(--ink-faint)]">
                  {outcome.label} <strong className="text-[color:var(--ink-soft)]">{probability(outcome.probability)}</strong>
                </span>
              ))}
            </div>
            <p className="text-[10px] leading-4 text-[color:var(--ink-faint)]">{event.matchNote}</p>
          </div>
        ))}
      </div>
    </details>
  );
}

function EventCard({ event }: { event: MacroPredictionEvent }) {
  return (
    <article className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.5)] p-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">{THEME_LABELS[event.theme]}</p>
          <a href={event.url} target="_blank" rel="noreferrer" className="mt-1 block text-sm font-semibold text-[color:var(--ink)] hover:text-[color:var(--accent)] hover:underline">{event.title} ↗</a>
        </div>
        <MatchBadge event={event} />
      </div>
      <div className="mt-3 space-y-2">
        {event.outcomes.filter((outcome) => !outcome.closed).map((outcome) => (
          <div key={outcome.marketId}>
            <div className="mb-1 flex items-center justify-between gap-3 text-xs">
              <span className="truncate text-[color:var(--ink-soft)]">{outcome.label}</span>
              <span className="shrink-0 font-semibold tabular-nums text-[color:var(--ink)]">{probability(outcome.probability)}</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-[color:rgba(255,255,255,0.06)]"><div className="h-full rounded-full bg-[color:var(--accent)]" style={{ width: probability(outcome.probability) }} /></div>
            <div className="mt-1 flex justify-between text-[9px] text-[color:var(--ink-faint)]"><span>{change(outcome.oneDayChange)}</span><span>Vol {money(outcome.volume)}</span></div>
          </div>
        ))}
      </div>
      <p className="mt-3 border-t border-[color:var(--line)] pt-3 text-[10px] leading-4 text-[color:var(--ink-faint)]">{event.matchNote}</p>
    </article>
  );
}

interface MacroViewProps {
  data: MarketMacroPredictionsData | null;
  loading: boolean;
  error: string | null;
}

export function MacroPredictionMarketsView({ data, loading, error }: MacroViewProps) {
  if (loading && !data) return <p className="py-12 text-center text-sm text-[color:var(--ink-faint)]">Loading macro contracts…</p>;
  if (error && !data) return <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-300">{error}</div>;
  if (!data) return null;
  return (
    <div className="space-y-3">
      {data.warning && <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-300">{data.warning}</div>}
      {data.events.length ? <div className="grid grid-cols-1 gap-4 lg:grid-cols-2"><>{data.events.map((event) => <EventCard key={event.mappingKey} event={event} />)}</></div> : <p className="py-8 text-center text-sm text-[color:var(--ink-faint)]">No supported US macro contracts are active.</p>}
      <p className="text-[10px] text-[color:var(--ink-faint)]">Live US, English-language contracts via {data.source} · {Math.round(data.cacheSeconds / 60)} min cache. Market prices are probabilities, not forecasts or investment advice. Earnings sharp-wallet scores are intentionally not applied.</p>
    </div>
  );
}
