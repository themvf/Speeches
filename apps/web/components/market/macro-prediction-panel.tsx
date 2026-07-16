"use client";

import { useMemo, useState } from "react";
import type { MacroPredictionEvent, MacroPredictionTheme, MacroSharpArchetype, MacroSharpCohort, MarketMacroPredictionsData } from "@/lib/server/types";

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
  const [view, setView] = useState<"contracts" | "wallets">("contracts");
  if (loading && !data) return <p className="py-12 text-center text-sm text-[color:var(--ink-faint)]">Loading macro contracts…</p>;
  if (error && !data) return <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-300">{error}</div>;
  if (!data) return null;
  const tracking = data.walletTracking;
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <p className="text-[10px] text-[color:var(--ink-faint)]">Recurring US macro releases only</p>
        <div className="flex overflow-hidden rounded-lg border border-[color:var(--line)]">
          {(["contracts", "wallets"] as const).map((id) => (
            <button key={id} type="button" onClick={() => setView(id)} className={`px-3 py-1 text-xs font-medium ${view === id ? "bg-[rgba(79,213,255,0.12)] text-[color:var(--ink)]" : "text-[color:var(--ink-faint)]"}`}>
              {id === "contracts" ? "Contracts" : "Macro sharps"}
            </button>
          ))}
        </div>
      </div>
      {data.warning && <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-300">{data.warning}</div>}
      {view === "contracts" && (data.events.length ? <div className="grid grid-cols-1 gap-4 lg:grid-cols-2"><>{data.events.map((event) => <EventCard key={event.mappingKey} event={event} />)}</></div> : <p className="py-8 text-center text-sm text-[color:var(--ink-faint)]">No supported US macro contracts are active.</p>)}
      {view === "wallets" && tracking && <MacroSharpWallets tracking={tracking} />}
      {view === "wallets" && !tracking && <p className="py-8 text-center text-sm text-[color:var(--ink-faint)]">Macro wallet scoring is initializing.</p>}
      <p className="text-[10px] text-[color:var(--ink-faint)]">Live US, English-language contracts via {data.source} · {Math.round(data.cacheSeconds / 60)} min cache. Each release counts once across all brackets. Only positions entered at least one hour before publication contribute to predictive share. Research context only.</p>
    </div>
  );
}

const ARCHETYPE: Record<MacroSharpArchetype, { label: string; color: string }> = {
  early_sharp: { label: "Early sharp", color: "#41d39d" },
  release_scalper: { label: "Release scalper", color: "#fbbf24" },
  longshot: { label: "Longshot", color: "#a78bfa" },
  unclassified: { label: "Building sample", color: "var(--ink-faint)" },
};

function MacroSharpWallets({ tracking }: { tracking: NonNullable<MarketMacroPredictionsData["walletTracking"]> }) {
  const [cohort, setCohort] = useState<MacroSharpCohort | "all">("all");
  const wallets = useMemo(() => tracking.wallets.filter((wallet) => cohort === "all" || wallet.cohort === cohort).slice(0, 40), [tracking.wallets, cohort]);
  return (
    <div className="space-y-4">
      {tracking.warning && <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-300">{tracking.warning}</div>}
      <div className="grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-6">
        {tracking.cohorts.map((item) => (
          <button key={item.id} type="button" onClick={() => setCohort(item.id)} className={`rounded-xl border p-3 text-left ${cohort === item.id ? "border-[color:var(--accent)] bg-[color:rgba(79,213,255,0.08)]" : "border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]"}`}>
            <p className="text-xs font-semibold text-[color:var(--ink)]">{item.label}</p>
            <p className="mt-1 text-[10px] text-[color:var(--ink-faint)]">{item.cadence} · max sample {item.observations}</p>
            <p className="mt-2 text-lg font-bold tabular-nums text-[color:var(--accent)]">{item.qualifiedWallets}</p>
            <p className="text-[9px] text-[color:var(--ink-faint)]">qualified wallets</p>
          </button>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-2 text-[10px]">
        <button type="button" onClick={() => setCohort("all")} className={`rounded px-2 py-1 ${cohort === "all" ? "bg-[rgba(79,213,255,0.12)] text-[color:var(--ink)]" : "text-[color:var(--ink-faint)]"}`}>All cohorts</button>
        <span className="text-[color:var(--ink-faint)]">Qualification: {tracking.minCohortEvents} releases in one family; generalist: {tracking.generalistMinEvents} across {tracking.generalistMinCohorts}+ families.</span>
      </div>
      <div className="overflow-x-auto rounded-xl border border-[color:var(--line)]">
        <table className="w-full min-w-[760px] text-xs">
          <thead><tr className="border-b border-[color:var(--line)] text-[10px] uppercase tracking-[0.08em] text-[color:var(--ink-faint)]"><th className="px-3 py-2 text-left">Wallet</th><th className="px-3 py-2 text-left">Specialty</th><th className="px-3 py-2 text-left">Class</th><th className="px-3 py-2 text-right">Events</th><th className="px-3 py-2 text-right">Win</th><th className="px-3 py-2 text-right">P&amp;L</th><th className="px-3 py-2 text-right">ROI</th><th className="px-3 py-2 text-right">Pre-release</th></tr></thead>
          <tbody>{wallets.map((wallet) => { const style = ARCHETYPE[wallet.archetype]; return <tr key={`${wallet.wallet}-${wallet.cohort}`} className="border-b border-[color:var(--line)] last:border-0"><td className="px-3 py-2"><a href={`https://polymarket.com/profile/${wallet.wallet}`} target="_blank" rel="noreferrer" className="font-medium text-[color:var(--ink-soft)] hover:text-[color:var(--accent)] hover:underline">{wallet.name}</a></td><td className="px-3 py-2 text-[color:var(--ink-faint)]">{wallet.cohortLabel}</td><td className="px-3 py-2"><span className="rounded px-1.5 py-0.5 text-[10px] font-semibold" style={{ color: style.color, backgroundColor: `color-mix(in srgb, ${style.color} 12%, transparent)` }}>{style.label}</span></td><td className="px-3 py-2 text-right tabular-nums">{wallet.events}</td><td className="px-3 py-2 text-right tabular-nums">{Math.round(wallet.winRate * 100)}%</td><td className="px-3 py-2 text-right font-semibold tabular-nums" style={{ color: wallet.pnlUsd >= 0 ? "#41d39d" : "#f87171" }}>{money(wallet.pnlUsd)}</td><td className="px-3 py-2 text-right tabular-nums">{wallet.roi == null ? "—" : `${Math.round(wallet.roi * 100)}%`}</td><td className="px-3 py-2 text-right tabular-nums">{wallet.predictiveShare == null ? "—" : `${Math.round(wallet.predictiveShare * 100)}%`}</td></tr>; })}</tbody>
        </table>
      </div>
      {!wallets.length && <p className="py-6 text-center text-xs text-[color:var(--ink-faint)]">No wallet has accumulated history in this cohort yet.</p>}
    </div>
  );
}
