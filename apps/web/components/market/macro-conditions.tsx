"use client";

import type { MarketMacroIndicator } from "@/lib/server/types";
import { assessConditions, summarizeConditions, type ConditionState } from "@/lib/macro-context";

// Cross-indicator condition reads. Every card states the convention it rests
// on and the readings that drove it, so nothing here is a claim the reader
// cannot check against a number on the same page.

const STATE_STYLES: Record<ConditionState, { dot: string; chip: string }> = {
  alert: { dot: "#f87171", chip: "border-red-400/30 bg-red-400/10 text-red-300" },
  watch: { dot: "#e0a94a", chip: "border-amber-400/30 bg-amber-400/10 text-amber-300" },
  calm: { dot: "#41d39d", chip: "border-emerald-400/25 bg-emerald-400/10 text-emerald-300" },
  neutral: { dot: "var(--ink-faint)", chip: "border-[color:var(--line)] bg-[color:rgba(15,32,50,0.6)] text-[color:var(--ink-soft)]" },
};

export function MacroConditions({ indicators }: { indicators: MarketMacroIndicator[] }) {
  const conditions = assessConditions(indicators);
  if (!conditions.length) return null;
  const summary = summarizeConditions(conditions);

  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.3)]">
      <div className="flex flex-wrap items-baseline justify-between gap-2 px-4 py-3">
        <div>
          <p className="text-sm font-semibold text-[color:var(--ink)]">What the readings say</p>
          {summary && <p className="mt-1 max-w-3xl text-xs leading-5 text-[color:var(--ink-soft)]">{summary}</p>}
        </div>
        <p className="text-[10px] uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
          Research context only
        </p>
      </div>

      <div className="grid grid-cols-1 gap-px border-t border-[color:var(--line)] bg-[color:var(--line)] sm:grid-cols-2 xl:grid-cols-3">
        {conditions.map((condition) => {
          const styles = STATE_STYLES[condition.state];
          return (
            <article key={condition.id} className="flex flex-col gap-2 bg-[color:rgba(9,21,34,0.55)] p-4">
              <div className="flex items-start justify-between gap-3">
                <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
                  {condition.label}
                </p>
                <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[9px] font-semibold ${styles.chip}`}>
                  <span
                    aria-hidden="true"
                    className="mr-1 inline-block h-1.5 w-1.5 rounded-full align-middle"
                    style={{ backgroundColor: styles.dot }}
                  />
                  {condition.headline}
                </span>
              </div>

              <p className="text-xs leading-5 text-[color:var(--ink-faint)]">{condition.meaning}</p>

              <dl className="mt-auto flex flex-col gap-1 border-t border-[color:var(--line)] pt-2">
                {condition.drivers.map((driver) => (
                  <div key={driver.label} className="flex items-baseline justify-between gap-3">
                    <dt className="text-[10px] text-[color:var(--ink-faint)]">{driver.label}</dt>
                    <dd className="text-[11px] font-semibold tabular-nums text-[color:var(--ink-soft)]">{driver.value}</dd>
                  </div>
                ))}
              </dl>
            </article>
          );
        })}
      </div>

      <p className="border-t border-[color:var(--line)] px-4 py-2 text-[10px] leading-4 text-[color:var(--ink-faint)]">
        These describe current conditions and the conventions economists read them by. They are not forecasts and not
        investment advice. Each state is derived from the readings shown beside it.
      </p>
    </section>
  );
}
