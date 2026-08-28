"use client";

import type { BreadthPair, MarketBreadth as MarketBreadthData } from "@/lib/server/types";

// Participation, beside the indices. Every claim here is checkable against the
// two percentages printed next to it - the same convention the macro
// conditions strip uses.

const TONE_STYLES: Record<BreadthPair["tone"], { chip: string; dot: string }> = {
  broad: { chip: "border-emerald-400/25 bg-emerald-400/10 text-emerald-300", dot: "#41d39d" },
  narrow: { chip: "border-amber-400/30 bg-amber-400/10 text-amber-300", dot: "#e0a94a" },
  even: { chip: "border-[color:var(--line)] bg-[color:rgba(15,32,50,0.6)] text-[color:var(--ink-soft)]", dot: "var(--ink-faint)" },
};

const TONE_LABEL: Record<BreadthPair["tone"], string> = {
  broad: "Broad",
  narrow: "Narrow",
  even: "Even",
};

function pct(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function pp(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)} pp`;
}

function toneColor(value: number): string {
  return value >= 0 ? "#41d39d" : "#f87171";
}

export function MarketBreadth({ breadth }: { breadth: MarketBreadthData | null }) {
  if (!breadth || breadth.pairs.length === 0) return null;

  return (
    <section className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-[color:var(--line)] px-4 py-2.5">
        <h3 className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-soft)]">
          Participation
        </h3>
        <p className="text-[11px] text-[color:var(--ink-soft)]">{breadth.summary}</p>
      </div>

      <div className="grid grid-cols-1 gap-px bg-[color:var(--line)] sm:grid-cols-2">
        {breadth.pairs.map((pair) => {
          const styles = TONE_STYLES[pair.tone];
          return (
            <article key={pair.id} className="flex flex-col gap-2 bg-[color:rgba(9,21,34,0.55)] p-4">
              <div className="flex items-start justify-between gap-3">
                <p className="text-xs font-semibold text-[color:var(--ink)]">{pair.label}</p>
                <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[9px] font-semibold ${styles.chip}`}>
                  <span
                    aria-hidden="true"
                    className="mr-1 inline-block h-1.5 w-1.5 rounded-full align-middle"
                    style={{ backgroundColor: styles.dot }}
                  />
                  {TONE_LABEL[pair.tone]} {pp(pair.spreadPp)}
                </span>
              </div>

              <dl className="flex flex-wrap gap-x-5 gap-y-1">
                <div className="flex items-baseline gap-1.5">
                  <dt className="text-[10px] text-[color:var(--ink-faint)]">Index ({pair.capSymbol})</dt>
                  <dd className="text-sm font-semibold tabular-nums" style={{ color: toneColor(pair.capPct) }}>
                    {pct(pair.capPct)}
                  </dd>
                </div>
                <div className="flex items-baseline gap-1.5">
                  <dt className="text-[10px] text-[color:var(--ink-faint)]">Average stock ({pair.equalSymbol})</dt>
                  <dd className="text-sm font-semibold tabular-nums" style={{ color: toneColor(pair.equalPct) }}>
                    {pct(pair.equalPct)}
                  </dd>
                </div>
              </dl>

              <p className="text-[11px] leading-4 text-[color:var(--ink-faint)]">{pair.reading}</p>
            </article>
          );
        })}
      </div>

      {breadth.smallVsLarge && (
        <div className="flex flex-wrap items-baseline justify-between gap-2 border-t border-[color:var(--line)] px-4 py-2 text-[11px] text-[color:var(--ink-faint)]">
          <span>Small caps vs large caps</span>
          <span className="tabular-nums">
            Russell 2000{" "}
            <strong style={{ color: toneColor(breadth.smallVsLarge.smallPct) }}>{pct(breadth.smallVsLarge.smallPct)}</strong>
            {" · "}S&amp;P 500{" "}
            <strong style={{ color: toneColor(breadth.smallVsLarge.largePct) }}>{pct(breadth.smallVsLarge.largePct)}</strong>
            {" · "}
            <span className="text-[color:var(--ink-soft)]">{pp(breadth.smallVsLarge.spreadPp)}</span>
          </span>
        </div>
      )}

      <p className="border-t border-[color:var(--line)] px-4 py-2 text-[10px] leading-4 text-[color:var(--ink-faint)]">
        A proxy for breadth, not an advance/decline line: it compares equal-weighted and cap-weighted versions of the
        same index, so it shows whether the average constituent kept up with the headline number. Research context only.
      </p>
    </section>
  );
}
