"use client";

import { useState } from "react";
import type { AttributionResult, RateTransmissionData } from "@/lib/rate-transmission";

interface Props {
  data: RateTransmissionData | null;
  loading: boolean;
  error: string | null;
}

const WINDOW_LABELS = ["1M", "3M", "6M", "12M"] as const;

/**
 * Leg colours, shared by the bars and their legend so the two cannot disagree.
 *
 * These are literals rather than `text-amber-300` because every numbered amber
 * utility in this app is dead: tailwind.config.ts sets `colors.amber` to the
 * string "var(--amber)", which replaces Tailwind's amber scale instead of
 * extending it, so `amber-300` matches no rule and the swatch silently
 * inherited the ink colour while its bar rendered orange.
 */
const LEG_COLORS = { base: "#4fd5ff", spread: "#f59e0b" } as const;

function pct(value: number): string {
  return `${value.toFixed(2)}%`;
}

/**
 * A spread is a difference between two yields, so it is percentage points, not
 * a percentage. Rendering 10Y-2Y of 0.62 as "0.62%" reads as a yield.
 */
function pp(value: number): string {
  return `${value.toFixed(2)} pp`;
}

function bp(value: number): string {
  return `${value >= 0 ? "+" : ""}${Math.round(value)} bp`;
}

function CurveCard({ label, description, value }: { label: string; description: string; value: number | null }) {
  return (
    <div className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.45)] p-3">
      <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">{label}</p>
      <p className="mt-1 text-xl font-bold tabular-nums text-[color:var(--ink)]">{value === null ? "—" : pp(value)}</p>
      <p className="mt-1 text-[10px] text-[color:var(--ink-faint)]">{description}</p>
    </div>
  );
}

function AttributionBars({ value }: { value: AttributionResult }) {
  const max = Math.max(Math.abs(value.baseBp), Math.abs(value.spreadBp), 1);
  const bar = (amount: number, color: string, y: number) => {
    const width = 45 * Math.abs(amount) / max;
    const x = amount >= 0 ? 50 : 50 - width;
    return <rect x={x} y={y} width={width} height="7" rx="2" fill={color} />;
  };
  return (
    <svg viewBox="0 0 100 30" className="h-20 w-full" role="img" aria-label="Mortgage rate change attributed to Treasury yield and spread changes">
      <line x1="50" y1="1" x2="50" y2="29" stroke="rgba(255,255,255,0.28)" strokeWidth="0.6" />
      {bar(value.baseBp, LEG_COLORS.base, 5)}
      {bar(value.spreadBp, LEG_COLORS.spread, 18)}
    </svg>
  );
}

export function RateTransmissionSection({ data, loading, error }: Props) {
  const [window, setWindow] = useState<(typeof WINDOW_LABELS)[number]>("3M");
  const windowEntry = data?.attribution.find((entry) => entry.window === window) ?? null;
  const attribution = windowEntry?.mortgage ?? null;
  const corporateAttribution = windowEntry?.corporate ?? null;
  const mortgage = data?.levels.mortgage ?? null;
  // Tolerates a payload from the build that predates the corporate row.
  const corporate = data?.levels.corporate?.available ? data.levels.corporate.level : null;
  return (
    <details className="group rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.3)]">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-4 py-4 [&::-webkit-details-marker]:hidden">
        <span>
          <span className="text-sm font-semibold text-[color:var(--ink)]">Rate Transmission</span>
          <span className="ml-2 text-xs text-[color:var(--ink-faint)]">What is moving borrowing costs: benchmarks or spreads?</span>
        </span>
        <span aria-hidden="true" className="text-base text-[color:var(--ink-faint)] transition-transform group-open:rotate-180">⌄</span>
      </summary>
      <div className="space-y-5 border-t border-[color:var(--line)] p-4">
        {loading && !data && <p className="text-xs text-[color:var(--ink-faint)]">Loading rate transmission…</p>}
        {error && !data && <p className="text-xs text-red-300">Rate transmission is unavailable: {error}</p>}
        {data && (
          <>
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Levels decomposition</p>
              <div className="mt-2 grid gap-3 lg:grid-cols-2">
                <div className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.45)] p-4">
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <p className="text-sm font-semibold text-[color:var(--ink)]">30Y mortgage</p>
                    <p className="text-lg font-bold tabular-nums text-[color:var(--ink)]">{mortgage ? pct(mortgage.rate) : "—"}</p>
                  </div>
                  <p className="mt-2 text-xs tabular-nums text-[color:var(--ink-soft)]">
                    {mortgage ? `${pct(mortgage.base)} 10Y Treasury + ${pp(mortgage.spread)} mortgage spread` : "Mortgage or Treasury input unavailable"}
                  </p>
                  {mortgage?.spreadContext && (
                    <p className="mt-2 text-[10px] text-[color:var(--ink-faint)]" title={`Percentile of ${mortgage.sampleSize} aligned observations`}>
                      Spread: {mortgage.spreadContext.summary}
                    </p>
                  )}
                </div>
                {corporate ? (
                  <div className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.45)] p-4">
                    <div className="flex flex-wrap items-baseline justify-between gap-2">
                      <p className="text-sm font-semibold text-[color:var(--ink)]">Baa corporate</p>
                      <p className="text-lg font-bold tabular-nums text-[color:var(--ink)]">{pct(corporate.rate)}</p>
                    </div>
                    <p className="mt-2 text-xs tabular-nums text-[color:var(--ink-soft)]">
                      {`${pct(corporate.base)} 10Y Treasury + ${pp(corporate.spread)} Baa spread`}
                    </p>
                    {corporate.spreadContext && (
                      <p className="mt-2 text-[10px] text-[color:var(--ink-faint)]" title={`Percentile of ${corporate.sampleSize} aligned observations`}>
                        Spread: {corporate.spreadContext.summary}
                      </p>
                    )}
                  </div>
                ) : (
                  <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-4">
                    <p className="text-sm font-semibold text-amber-200">Baa corporate</p>
                    <p className="mt-2 text-xs leading-5 text-amber-100/70">{data.levels.corporate.reason}</p>
                  </div>
                )}
              </div>
            </div>

            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Curve transmission</p>
              <div className="mt-2 grid grid-cols-2 gap-3 lg:grid-cols-4">
                <CurveCard label="Short tail" description="2Y − 3M" value={data.curve.shortTail?.value ?? null} />
                <CurveCard label="Belly" description="10Y − 2Y" value={data.curve.belly?.value ?? null} />
                <CurveCard label="Long tail" description="30Y − 10Y" value={data.curve.longTail?.value ?? null} />
                <CurveCard label="Policy gap" description="2Y − effective fed funds" value={data.curve.policyGap?.value ?? null} />
              </div>
            </div>

            <div>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Change attribution</p>
                <div className="flex gap-1">
                  {WINDOW_LABELS.map((label) => (
                    <button key={label} type="button" onClick={() => setWindow(label)} className={`rounded-md border px-2 py-1 text-[10px] font-semibold ${window === label ? "border-[color:var(--accent)] bg-[color:rgba(79,213,255,0.12)] text-[color:var(--accent)]" : "border-[color:var(--line)] text-[color:var(--ink-faint)]"}`}>{label}</button>
                  ))}
                </div>
              </div>
              <div className="mt-2 grid gap-3 lg:grid-cols-2">
                {([["30Y mortgage", attribution], ["Baa corporate", corporateAttribution]] as const).map(([label, value]) => (
                  <div key={label} className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.45)] p-4">
                    {value ? (
                      <>
                        <p className="text-sm text-[color:var(--ink-soft)]">{label} change: <span className="font-bold tabular-nums text-[color:var(--ink)]">{bp(value.totalBp)}</span></p>
                        <AttributionBars value={value} />
                        <div className="flex flex-wrap gap-4 text-xs tabular-nums text-[color:var(--ink-soft)]">
                          <span><span aria-hidden="true" style={{ color: LEG_COLORS.base }}>■</span> Treasury {bp(value.baseBp)}</span>
                          <span><span aria-hidden="true" style={{ color: LEG_COLORS.spread }}>■</span> Spread {bp(value.spreadBp)}</span>
                        </div>
                        <p className="mt-2 text-[10px] text-[color:var(--ink-faint)]">{value.startDate} to {value.endDate}. Components sum to the total by construction.</p>
                      </>
                    ) : (
                      <>
                        <p className="text-sm text-[color:var(--ink-soft)]">{label}</p>
                        <p className="mt-2 text-xs text-[color:var(--ink-faint)]">Not enough aligned history for this window.</p>
                      </>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {data.warnings.length > 0 && <p className="text-xs text-amber-300">Partial data: {data.warnings.join(" · ")}</p>}
            <p className="text-[10px] leading-4 text-[color:var(--ink-faint)]">
              Descriptive accounting decomposition, not a causal estimate. Sources: {data.sources.map((source, index) => <span key={source.seriesId}>{index ? ", " : ""}<a className="text-[color:var(--accent)] hover:underline" href={source.url} target="_blank" rel="noreferrer">{source.seriesId}</a></span>)}.
            </p>
          </>
        )}
      </div>
    </details>
  );
}
