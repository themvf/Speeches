"use client";

import type { TreasuryYield } from "@/lib/server/types";

/**
 * The Treasury curve as a shape rather than a list. Four tenors cannot show a
 * kink or an inversion; thirteen can, and the whole point of the curve is
 * where it bends.
 *
 * X is log-spaced on maturity: evenly spacing thirteen tenors would give the
 * 1M-to-1Y front end, where policy expectations live, a twelfth of the width
 * it deserves while stretching the 20Y-to-30Y segment across a third of the
 * plot. Hand-rolled SVG, matching every other chart in this app.
 */
export function YieldCurve({ yields, spread, spreadLabel }: {
  yields: TreasuryYield[];
  /**
   * The canonical 10Y-2Y value. Passed in rather than derived here: FRED's
   * T10Y2Y already owns that number on the Macro card, and this plot is built
   * from Treasury XML, so computing a second one invites two figures on the
   * same screen disagreeing by a day. Shape comes from these points; the
   * number comes from one source.
   */
  spread?: number | null;
  spreadLabel?: string;
}) {
  const points = [...yields].sort((left, right) => left.months - right.months);
  if (points.length < 3) return null;

  const width = 520;
  const height = 190;
  const padLeft = 38;
  const padRight = 12;
  const padTop = 14;
  const padBottom = 26;

  const rates = points.map((point) => point.rate);
  const minRate = Math.min(...rates);
  const maxRate = Math.max(...rates);
  // A flat curve should read as flat, not as noise amplified to fill the box.
  const pad = Math.max((maxRate - minRate) * 0.15, 0.1);
  const low = minRate - pad;
  const high = maxRate + pad;

  const logs = points.map((point) => Math.log(point.months));
  const minLog = Math.min(...logs);
  const maxLog = Math.max(...logs);
  const logSpan = maxLog - minLog || 1;

  const x = (months: number) =>
    padLeft + ((Math.log(months) - minLog) / logSpan) * (width - padLeft - padRight);
  const y = (rate: number) =>
    padTop + (1 - (rate - low) / (high - low || 1)) * (height - padTop - padBottom);

  const line = points.map((point) => `${x(point.months).toFixed(1)},${y(point.rate).toFixed(1)}`).join(" ");

  // Gridlines at the rate extremes plus the midpoint - enough to read a level
  // off the plot without drawing a full grid behind a single line.
  const gridRates = [low + (high - low) * 0.5, minRate, maxRate];

  // Label only the tenors a reader actually anchors on, so the axis does not
  // collide with itself at the short end.
  const labelled = new Set(["1M", "3M", "1Y", "2Y", "5Y", "10Y", "30Y"]);

  return (
    <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-[color:var(--line)] px-4 py-2.5">
        <h3 className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-soft)]">
          Treasury Yield Curve
        </h3>
        {typeof spread === "number" && (
          <p className="text-[11px] tabular-nums text-[color:var(--ink-faint)]">
            10Y&minus;2Y{" "}
            <strong className={spread < 0 ? "text-red-300" : "text-[color:var(--ink-soft)]"}>
              {spread >= 0 ? "+" : ""}{spread.toFixed(2)} pp
            </strong>
            {spread < 0 && " · inverted"}
            {spreadLabel && <span className="ml-1 text-[color:var(--ink-faint)]">· {spreadLabel}</span>}
          </p>
        )}
      </div>

      <div className="overflow-x-auto px-2 py-3">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="h-[190px] w-full min-w-[320px]"
          role="img"
          aria-label={`Treasury yield curve from ${points[0].tenor} at ${points[0].rate.toFixed(2)} percent to ${points[points.length - 1].tenor} at ${points[points.length - 1].rate.toFixed(2)} percent`}
        >
          {gridRates.map((rate) => (
            <g key={rate}>
              <line
                x1={padLeft} y1={y(rate)} x2={width - padRight} y2={y(rate)}
                stroke="rgba(255,255,255,0.07)" strokeWidth="1"
              />
              <text
                x={padLeft - 6} y={y(rate) + 3} textAnchor="end"
                fill="var(--ink-faint)" fontSize="9" style={{ fontVariantNumeric: "tabular-nums" }}
              >
                {rate.toFixed(2)}
              </text>
            </g>
          ))}

          <polyline
            points={line}
            fill="none"
            stroke="#4fd5ff"
            strokeWidth="2"
            strokeLinejoin="round"
            strokeLinecap="round"
          />

          {points.map((point) => (
            <g key={point.tenor}>
              <circle cx={x(point.months)} cy={y(point.rate)} r="2.6" fill="#4fd5ff">
                <title>{`${point.tenor} · ${point.rate.toFixed(2)}%`}</title>
              </circle>
              {labelled.has(point.tenor) && (
                <text
                  x={x(point.months)} y={height - 8} textAnchor="middle"
                  fill="var(--ink-faint)" fontSize="9"
                >
                  {point.tenor}
                </text>
              )}
            </g>
          ))}
        </svg>
      </div>

      <p className="border-t border-[color:var(--line)] px-4 py-2 text-[10px] text-[color:var(--ink-faint)]">
        {points.length} tenors · maturity axis is log-scaled · source US Treasury
      </p>
    </div>
  );
}
