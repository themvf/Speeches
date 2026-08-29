"use client";

import { useState } from "react";
import type { AttributionResult, RateTransmissionData } from "@/lib/rate-transmission";

interface Props {
  data: RateTransmissionData | null;
  loading: boolean;
  error: string | null;
}

const WINDOW_LABELS = ["1M", "3M", "6M", "12M", "YTD"] as const;

function pct(value: number): string {
  return `${value.toFixed(2)}%`;
}

function bp(value: number): string {
  return `${value >= 0 ? "+" : ""}${Math.round(value)} bp`;
}

function regimeLabel(regime: NonNullable<RateTransmissionData["creditResearch"]>["regime"]): string {
  return ({
    restrictive_financing: "Rates and risk premium both tightening",
    rates_led_tightening: "Rates-led tightening",
    flight_to_quality: "Flight-to-quality pattern",
    broad_easing: "Rates and risk premium both easing",
    mixed: "Mixed cross-asset signal",
  } as const)[regime];
}

function appetiteLabel(appetite: NonNullable<RateTransmissionData["creditResearch"]>["appetite"]): string {
  return appetite === "risk_averse" ? "Below-average risk appetite" : appetite === "supportive" ? "Above-average risk appetite" : "Near-average risk appetite";
}

function leadLagCopy(result: NonNullable<RateTransmissionData["leadLag"]["mortgageTreasury"]>): string {
  if (result.verdict === "no_clear_lead") return "No clear lead: the strongest lag does not materially improve on contemporaneous correlation.";
  const leader = result.verdict === "treasury_leads" ? "Treasury" : "Mortgage";
  return `${leader} changes led by ${Math.abs(result.bestLag)} ${Math.abs(result.bestLag) === 1 ? "week" : "weeks"} in this sample.`;
}

function CurveCard({ label, description, value }: { label: string; description: string; value: number | null }) {
  return (
    <div className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.45)] p-3">
      <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">{label}</p>
      <p className="mt-1 text-xl font-bold tabular-nums text-[color:var(--ink)]">{value === null ? "—" : pct(value)}</p>
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
      {bar(value.baseBp, "#4fd5ff", 5)}
      {bar(value.spreadBp, "#f59e0b", 18)}
    </svg>
  );
}

export function RateTransmissionSection({ data, loading, error }: Props) {
  const [window, setWindow] = useState<(typeof WINDOW_LABELS)[number]>("3M");
  const attribution = data?.attribution.find((entry) => entry.window === window)?.mortgage ?? null;
  const mortgage = data?.levels.mortgage ?? null;
  const passThrough = data?.passThrough?.mortgage ?? null;
  const leadLag = data?.leadLag?.mortgageTreasury ?? null;
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
                    {mortgage ? `${pct(mortgage.base)} 10Y Treasury + ${pct(mortgage.spread)} mortgage spread` : "Mortgage or Treasury input unavailable"}
                  </p>
                  {mortgage?.spreadPercentile !== null && mortgage?.spreadPercentile !== undefined && (
                    <p className="mt-2 text-[10px] text-[color:var(--ink-faint)]">Spread is at the {Math.round(mortgage.spreadPercentile)}th percentile of {mortgage.sampleSize} aligned observations.</p>
                  )}
                </div>
                <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-4">
                  <p className="text-sm font-semibold text-amber-200">Rating-specific corporate debt and High Yield</p>
                  <p className="mt-2 text-xs leading-5 text-amber-100/70">Unavailable pending a licensed corporate-yield source. {data.levels.corporate.reason}</p>
                </div>
              </div>
            </div>

            {data.creditResearch && (
              <div>
                <div className="flex flex-wrap items-end justify-between gap-2">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Public corporate-credit context</p>
                    <p className="mt-1 text-[10px] text-[color:var(--ink-faint)]">Federal Reserve research series · senior unsecured nonfinancial corporate bonds · monthly and revisable</p>
                  </div>
                  <p className="text-[10px] text-[color:var(--ink-faint)]">Observation {data.creditResearch.observationDate}</p>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-3 lg:grid-cols-4">
                  <CurveCard label="Corporate spread" description="Gilchrist–Zakrajšek aggregate" value={data.creditResearch.corporateSpread} />
                  <CurveCard label="Default-risk component" description="Spread less excess premium" value={data.creditResearch.defaultRiskComponent} />
                  <CurveCard label="Excess bond premium" description={appetiteLabel(data.creditResearch.appetite)} value={data.creditResearch.excessBondPremium} />
                  <div className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.45)] p-3">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Fed research model</p>
                    <p className="mt-1 text-xl font-bold tabular-nums text-[color:var(--ink)]">{(data.creditResearch.recessionProbability * 100).toFixed(1)}%</p>
                    <p className="mt-1 text-[10px] text-[color:var(--ink-faint)]">Model-implied 12-month recession probability</p>
                  </div>
                </div>
                <div className="mt-3 rounded-lg border border-[color:rgba(79,213,255,0.25)] bg-[color:rgba(79,213,255,0.06)] p-3">
                  <p className="text-sm font-semibold text-[color:var(--ink)]">{regimeLabel(data.creditResearch.regime)}</p>
                  <p className="mt-1 text-xs text-[color:var(--ink-faint)]">
                    Three-month move: 10Y Treasury {data.creditResearch.treasuryChange3mBp === null ? "n/a" : bp(data.creditResearch.treasuryChange3mBp)}; excess bond premium {data.creditResearch.ebpChange3mBp === null ? "n/a" : bp(data.creditResearch.ebpChange3mBp)}. EBP is at the {Math.round(data.creditResearch.ebpPercentile)}th percentile of its history.
                  </p>
                </div>
              </div>
            )}

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
              <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Pass-through and timing</p>
              <div className="mt-2 grid gap-3 lg:grid-cols-2">
                <div className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.45)] p-4">
                  <p className="text-sm font-semibold text-[color:var(--ink)]">Mortgage pass-through</p>
                  {passThrough ? (
                    <>
                      <p className="mt-2 text-xl font-bold tabular-nums text-[color:var(--ink)]">β {passThrough.beta.toFixed(2)} <span className="text-sm font-medium text-[color:var(--ink-faint)]">± {passThrough.standardError.toFixed(2)}</span></p>
                      <p className="mt-1 text-xs text-[color:var(--ink-faint)]">R² {passThrough.rSquared.toFixed(2)} · n={passThrough.n} weekly changes · rolling {passThrough.window}-week window</p>
                      <p className="mt-2 text-[10px] leading-4 text-[color:var(--ink-faint)]">A 100 bp weekly 10Y Treasury move was associated with approximately {Math.round(passThrough.beta * 100)} bp in the mortgage rate within this window. Association, not a structural causal estimate.</p>
                    </>
                  ) : <p className="mt-2 text-xs text-[color:var(--ink-faint)]">Suppressed: fewer than 30 usable weekly changes or insufficient variation.</p>}
                </div>
                <div className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.45)] p-4">
                  <p className="text-sm font-semibold text-[color:var(--ink)]">Treasury ↔ mortgage timing</p>
                  {leadLag ? (
                    <>
                      <p className="mt-2 text-xl font-bold tabular-nums text-[color:var(--ink)]">r {leadLag.correlation.toFixed(2)}</p>
                      <p className="mt-1 text-xs text-[color:var(--ink-faint)]">{leadLagCopy(leadLag)}</p>
                      <p className="mt-2 text-[10px] leading-4 text-[color:var(--ink-faint)]">Weekly yield-level changes, ±4-week search, n={leadLag.n}. A lead is named only when |r|≥0.20 and exceeds lag zero by at least 0.05; correlation does not identify the mechanism.</p>
                    </>
                  ) : <p className="mt-2 text-xs text-[color:var(--ink-faint)]">Insufficient aligned history for guarded lead/lag analysis.</p>}
                </div>
              </div>
            </div>

            <div>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Mortgage attribution</p>
                <div className="flex gap-1">
                  {WINDOW_LABELS.map((label) => (
                    <button key={label} type="button" onClick={() => setWindow(label)} className={`rounded-md border px-2 py-1 text-[10px] font-semibold ${window === label ? "border-[color:var(--accent)] bg-[color:rgba(79,213,255,0.12)] text-[color:var(--accent)]" : "border-[color:var(--line)] text-[color:var(--ink-faint)]"}`}>{label}</button>
                  ))}
                </div>
              </div>
              {attribution ? (
                <div className="mt-2 rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.45)] p-4">
                  <p className="text-sm text-[color:var(--ink-soft)]">Mortgage rate change: <span className="font-bold tabular-nums text-[color:var(--ink)]">{bp(attribution.totalBp)}</span></p>
                  <AttributionBars value={attribution} />
                  <div className="flex flex-wrap gap-4 text-xs tabular-nums">
                    <span className="text-[color:var(--accent)]">■ Treasury {bp(attribution.baseBp)}</span>
                    <span className="text-amber-300">■ Spread {bp(attribution.spreadBp)}</span>
                  </div>
                  <p className="mt-2 text-[10px] text-[color:var(--ink-faint)]">{attribution.startDate} to {attribution.endDate}. Components sum to the total by construction.</p>
                </div>
              ) : <p className="mt-2 text-xs text-[color:var(--ink-faint)]">Not enough aligned history for this window.</p>}
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
