"use client";

import type {
  MacroCalendarEntry,
  MacroPredictionEvent,
  MarketMacroCalendarData,
} from "@/lib/server/types";
import {
  daysUntil,
  formatCalendarDate,
  localIsoDate,
  matchContract,
  relativeDayLabel,
} from "@/lib/macro-calendar-display";

interface Props {
  data: MarketMacroCalendarData | null;
  loading: boolean;
  error: string | null;
  contracts: MacroPredictionEvent[];
}

/** Rows shown before the rest of the horizon is folded into a disclosure. */
const VISIBLE_ROWS = 8;

function probability(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function CalendarRow({ entry, today, contract }: {
  entry: MacroCalendarEntry;
  today: string;
  contract: MacroPredictionEvent | null;
}) {
  const days = daysUntil(entry.date, today);
  const imminent = days <= 1;
  return (
    <li className="flex flex-wrap items-start gap-x-4 gap-y-2 border-t border-[color:var(--line)] px-4 py-3 first:border-t-0">
      <div className="w-[120px] shrink-0">
        <p className="text-xs font-semibold tabular-nums text-[color:var(--ink)]">{formatCalendarDate(entry.date, { weekday: "short", month: "short", day: "numeric" })}</p>
        <p className={`text-[10px] ${imminent ? "font-semibold text-[color:var(--accent)]" : "text-[color:var(--ink-faint)]"}`}>
          {relativeDayLabel(days)}
        </p>
      </div>

      <div className="min-w-[200px] flex-1">
        <a
          href={entry.releaseUrl}
          target="_blank"
          rel="noreferrer"
          className="text-xs font-semibold text-[color:var(--ink)] hover:text-[color:var(--accent)] hover:underline"
        >
          {entry.releaseName} ↗
        </a>
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {entry.indicators.map((indicator) => (
            <span
              key={indicator.id}
              className="rounded bg-[color:rgba(15,32,50,0.8)] px-2 py-0.5 text-[10px] text-[color:var(--ink-faint)]"
            >
              {indicator.label}
            </span>
          ))}
        </div>
      </div>

      {contract?.leadingOutcome && (
        <a
          href={contract.url}
          target="_blank"
          rel="noreferrer"
          className="shrink-0 rounded-lg border border-[color:rgba(79,213,255,0.2)] bg-[color:rgba(79,213,255,0.05)] px-2.5 py-1.5 text-right"
          title={contract.matchNote}
        >
          <span className="block text-[9px] uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Market lean</span>
          <span className="block text-[11px] text-[color:var(--ink-soft)]">
            {contract.leadingOutcome.label}{" "}
            <strong className="text-[color:var(--accent)]">{probability(contract.leadingOutcome.probability)}</strong>
          </span>
        </a>
      )}
    </li>
  );
}

export function MacroCalendar({ data, loading, error, contracts }: Props) {
  if (loading && !data) {
    return (
      <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.3)] px-4 py-6 text-xs text-[color:var(--ink-faint)]">
        Loading the FRED release calendar…
      </div>
    );
  }
  if (error && !data) {
    return (
      <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-xs text-amber-300">
        Release calendar is temporarily unavailable: {error}
      </div>
    );
  }
  if (!data) return null;

  const today = localIsoDate(new Date());
  const upcoming = data.entries.filter((entry) => entry.date >= today);
  const visible = upcoming.slice(0, VISIBLE_ROWS);
  const remaining = upcoming.slice(VISIBLE_ROWS);
  const rowFor = (entry: MacroCalendarEntry) => (
    <CalendarRow
      key={`${entry.date}-${entry.releaseId}`}
      entry={entry}
      today={today}
      contract={matchContract(entry, contracts)}
    />
  );

  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.3)]">
      <div className="flex flex-wrap items-end justify-between gap-3 px-4 py-3">
        <div>
          <p className="text-sm font-semibold text-[color:var(--ink)]">Release Calendar</p>
          <p className="mt-1 text-xs text-[color:var(--ink-faint)]">
            Scheduled publication dates for the indicators on this page, over the next {data.horizonDays} days.
          </p>
        </div>
        <p className="text-[10px] text-[color:var(--ink-faint)]">{upcoming.length} scheduled releases</p>
      </div>

      {data.warnings?.length ? (
        <p className="border-t border-[color:var(--line)] px-4 py-2 text-[10px] text-amber-300">
          {data.warnings.join(" ")}
        </p>
      ) : null}

      {upcoming.length === 0 ? (
        <p className="border-t border-[color:var(--line)] px-4 py-4 text-xs text-[color:var(--ink-faint)]">
          No scheduled releases published for this window yet.
        </p>
      ) : (
        <>
          <ul className="border-t border-[color:var(--line)]">{visible.map(rowFor)}</ul>
          {remaining.length > 0 && (
            <details className="group border-t border-[color:var(--line)]">
              <summary className="flex cursor-pointer list-none items-center justify-between px-4 py-3 text-xs text-[color:var(--ink-faint)] [&::-webkit-details-marker]:hidden">
                <span>Show the remaining {remaining.length} scheduled releases</span>
                <span aria-hidden="true" className="text-base transition-transform group-open:rotate-180">⌄</span>
              </summary>
              <ul className="border-t border-[color:var(--line)]">{remaining.map(rowFor)}</ul>
            </details>
          )}
        </>
      )}

      <p className="border-t border-[color:var(--line)] px-4 py-2 text-[10px] leading-4 text-[color:var(--ink-faint)]">
        Dates only — FRED does not publish a release time of day. Dates come from the source agencies and do not
        necessarily reflect when the data lands in FRED. Daily interest-rate and exchange-rate refreshes are excluded.
      </p>
    </section>
  );
}
