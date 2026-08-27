"use client";

import type {
  MarketRatesCreditData,
  MarketRatesCreditMetric,
  MarketRatesCreditTone,
} from "@/lib/server/types";

interface Props {
  data: MarketRatesCreditData | null;
  loading: boolean;
  error: string | null;
}

const TONE_STYLES: Record<MarketRatesCreditTone, string> = {
  positive: "border-emerald-400/25 bg-emerald-400/[0.07] text-emerald-300",
  neutral: "border-[color:var(--line)] bg-[color:rgba(9,21,34,0.52)] text-[color:var(--accent)]",
  negative: "border-amber-400/25 bg-amber-400/[0.07] text-amber-300",
};

function basisPoints(value: number | null): string {
  if (value === null) return "—";
  const amount = value * 100;
  return `${amount >= 0 ? "+" : ""}${amount.toFixed(0)}`;
}

function percentile(value: number | null): string {
  return value === null ? "—" : `${Math.round(value)}th`;
}

function observationDate(value: string): string {
  return new Date(`${value}T00:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}

function CurveChart({ metrics }: { metrics: MarketRatesCreditMetric[] }) {
  if (metrics.length < 2) return null;
  const width = 760;
  const height = 220;
  const left = 38;
  const right = 16;
  const top = 18;
  const bottom = 34;
  const currentValues = metrics.map((metric) => metric.value);
  const monthAgoValues = metrics.map((metric) => metric.change1m === null ? metric.value : metric.value - metric.change1m);
  const allValues = [...currentValues, ...monthAgoValues];
  const rawMin = Math.min(...allValues);
  const rawMax = Math.max(...allValues);
  const padding = Math.max((rawMax - rawMin) * 0.18, 0.12);
  const min = rawMin - padding;
  const max = rawMax + padding;
  const range = max - min || 1;
  const x = (index: number) => left + (index / (metrics.length - 1)) * (width - left - right);
  const y = (value: number) => top + ((max - value) / range) * (height - top - bottom);
  const points = (values: number[]) => values.map((value, index) => `${x(index).toFixed(1)},${y(value).toFixed(1)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="mt-4 w-full" role="img" aria-label="Treasury yield curve now compared with one month ago">
      {[0, 0.5, 1].map((position) => {
        const gridY = top + position * (height - top - bottom);
        const value = max - position * range;
        return (
          <g key={position}>
            <line x1={left} y1={gridY} x2={width - right} y2={gridY} stroke="rgba(255,255,255,0.08)" />
            <text x={left - 7} y={gridY + 4} textAnchor="end" fill="rgba(214,228,240,0.48)" fontSize="10">{value.toFixed(1)}%</text>
          </g>
        );
      })}
      <polyline points={points(monthAgoValues)} fill="none" stroke="rgba(214,228,240,0.42)" strokeWidth="2" strokeDasharray="5 5" />
      <polyline points={points(currentValues)} fill="none" stroke="#4fd5ff" strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" />
      {currentValues.map((value, index) => (
        <g key={metrics[index].id}>
          <circle cx={x(index)} cy={y(value)} r="3.5" fill="#4fd5ff" />
          <text x={x(index)} y={height - 10} textAnchor="middle" fill="rgba(214,228,240,0.58)" fontSize="10">{metrics[index].shortLabel}</text>
        </g>
      ))}
    </svg>
  );
}

function CreditTable({ title, metrics }: { title: string; metrics: MarketRatesCreditMetric[] }) {
  return (
    <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.42)]">
      <div className="border-b border-[color:var(--line)] px-4 py-3">
        <h4 className="text-sm font-semibold text-[color:var(--ink)]">{title}</h4>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] text-left text-xs">
          <thead className="text-[10px] uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
            <tr>
              <th className="px-4 py-2 font-medium">Segment</th>
              <th className="px-3 py-2 text-right font-medium">OAS</th>
              <th className="px-3 py-2 text-right font-medium">1D bp</th>
              <th className="px-3 py-2 text-right font-medium">1W bp</th>
              <th className="px-3 py-2 text-right font-medium">1M bp</th>
              <th className="px-3 py-2 text-right font-medium">Percentile</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((metric) => (
              <tr key={metric.id} className="border-t border-[color:var(--line)] text-[color:var(--ink-faint)]">
                <td className="px-4 py-3">
                  <a href={metric.sourceUrl} target="_blank" rel="noreferrer" className="font-semibold text-[color:var(--ink)] hover:text-[color:var(--accent)]">{metric.shortLabel}</a>
                  <span className="ml-2 text-[10px]">{observationDate(metric.observationDate)}</span>
                </td>
                <td className="px-3 py-3 text-right font-semibold tabular-nums text-[color:var(--ink)]">{(metric.value * 100).toFixed(0)} bp</td>
                {[metric.change1d, metric.change1w, metric.change1m].map((change, index) => (
                  <td key={index} className={`px-3 py-3 text-right tabular-nums ${change !== null && change > 0 ? "text-amber-300" : change !== null && change < 0 ? "text-emerald-300" : ""}`}>
                    {basisPoints(change)}
                  </td>
                ))}
                <td className="px-3 py-3 text-right tabular-nums">{percentile(metric.percentile)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function RatesCreditSection({ data, loading, error }: Props) {
  if (loading && !data) {
    return <div className="rounded-xl border border-[color:var(--line)] p-6 text-sm text-[color:var(--ink-faint)]">Loading rates and credit intelligence…</div>;
  }
  if (error && !data) {
    return (
      <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4 text-sm text-amber-300">
        <p className="font-semibold">Rates and credit intelligence is unavailable</p>
        <p className="mt-1 text-xs text-amber-200/80">{error}</p>
      </div>
    );
  }
  if (!data) return null;

  return (
    <section className="space-y-4 rounded-2xl border border-[color:rgba(79,213,255,0.22)] bg-[color:rgba(6,17,28,0.6)] p-4 md:p-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--accent)]">Rates & Credit Pulse</p>
          <h2 className="mt-1 text-xl font-semibold text-[color:var(--ink)]">Cross-market conditions, not isolated prints</h2>
          <p className="mt-1 max-w-3xl text-xs leading-5 text-[color:var(--ink-faint)]">Curve classification, real yields, credit quality tiers, and historical context derived from daily public observations.</p>
        </div>
        <p className="text-[10px] text-[color:var(--ink-faint)]">Generated {new Date(data.generatedAt).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}</p>
      </div>

      {data.warnings.map((warning) => <div key={warning} className="rounded-lg border border-amber-400/20 bg-amber-400/[0.06] px-3 py-2 text-xs text-amber-200">{warning}</div>)}

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        {data.signals.map((signal) => (
          <article key={signal.id} className={`rounded-xl border p-4 ${TONE_STYLES[signal.tone]}`}>
            <p className="text-[10px] font-semibold uppercase tracking-[0.1em] opacity-70">{signal.label}</p>
            <p className="mt-1 text-lg font-bold text-[color:var(--ink)]">{signal.state}</p>
            <p className="mt-2 text-xs leading-5 text-[color:var(--ink-faint)]">{signal.summary}</p>
          </article>
        ))}
      </div>

      {data.drivers.length > 0 && (
        <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.42)] p-4">
          <p className="text-[10px] font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">Largest one-month drivers</p>
          <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-2">
            {data.drivers.map((driver) => (
              <div key={driver.label} className="flex gap-3 text-xs leading-5">
                <span className={`font-bold ${driver.tone === "negative" ? "text-amber-300" : driver.tone === "positive" ? "text-emerald-300" : "text-[color:var(--accent)]"}`}>{driver.label}</span>
                <span className="text-[color:var(--ink-faint)]">{driver.detail}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(260px,1fr)]">
        <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.42)] p-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-[color:var(--ink)]">Treasury curve</h3>
              <p className="mt-1 text-xs text-[color:var(--ink-faint)]">Current curve <span className="text-[color:var(--accent)]">—</span> versus one month ago <span className="opacity-60">- - -</span></p>
            </div>
            <span className="text-[10px] text-[color:var(--ink-faint)]">Daily constant maturity</span>
          </div>
          <CurveChart metrics={data.treasuryCurve} />
          <div className="mt-2 grid grid-cols-5 gap-2 sm:grid-cols-10">
            {data.treasuryCurve.map((metric) => (
              <a key={metric.id} href={metric.sourceUrl} target="_blank" rel="noreferrer" className="text-center hover:text-[color:var(--accent)]">
                <p className="text-[10px] text-[color:var(--ink-faint)]">{metric.shortLabel}</p>
                <p className="mt-0.5 text-xs font-semibold tabular-nums text-[color:var(--ink)]">{metric.value.toFixed(2)}%</p>
                <p className={`text-[9px] tabular-nums ${(metric.change1m ?? 0) > 0 ? "text-amber-300" : (metric.change1m ?? 0) < 0 ? "text-emerald-300" : "text-[color:var(--ink-faint)]"}`}>{basisPoints(metric.change1m)} bp</p>
              </a>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.42)] p-4">
          <h3 className="text-sm font-semibold text-[color:var(--ink)]">Real yields</h3>
          <p className="mt-1 text-xs text-[color:var(--ink-faint)]">Inflation-indexed Treasury yields and their one-month move.</p>
          <div className="mt-4 space-y-3">
            {data.realYields.map((metric) => (
              <a key={metric.id} href={metric.sourceUrl} target="_blank" rel="noreferrer" className="flex items-center justify-between rounded-lg border border-[color:var(--line)] px-3 py-3 hover:border-[color:var(--line-strong)]">
                <span>
                  <span className="block text-xs font-semibold text-[color:var(--ink)]">{metric.shortLabel}</span>
                  <span className="text-[10px] text-[color:var(--ink-faint)]">{observationDate(metric.observationDate)}</span>
                </span>
                <span className="text-right">
                  <span className="block text-base font-bold tabular-nums text-[color:var(--ink)]">{metric.value.toFixed(2)}%</span>
                  <span className={`text-[10px] tabular-nums ${(metric.change1m ?? 0) > 0 ? "text-amber-300" : "text-emerald-300"}`}>{basisPoints(metric.change1m)} bp / 1M</span>
                </span>
              </a>
            ))}
          </div>
        </div>
      </div>

      {data.creditDataStatus === "enabled" ? (
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
          <CreditTable title="Investment grade spreads" metrics={data.investmentGrade} />
          <CreditTable title="High yield spreads" metrics={data.highYield} />
        </div>
      ) : (
        <div className="rounded-xl border border-dashed border-[color:var(--line-strong)] bg-[color:rgba(9,21,34,0.32)] p-5">
          <h3 className="text-sm font-semibold text-[color:var(--ink)]">Corporate spread matrix ready for an authorized source</h3>
          <p className="mt-2 max-w-3xl text-xs leading-5 text-[color:var(--ink-faint)]">The IG, AAA, AA, A, BBB, HY, BB, B, and CCC series are implemented but disabled by default. Enable the ICE data flag only for a deployment whose internal-use or redistribution rights permit these values to be displayed.</p>
        </div>
      )}

      <p className="text-right text-[10px] leading-4 text-[color:var(--ink-faint)]">Source: FRED®. OAS series are ICE BofA indices distributed through FRED. Percentiles and z-scores use the observations returned by the source and may reflect a rolling history window.</p>
    </section>
  );
}
