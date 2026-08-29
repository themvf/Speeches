"use client";

import { useState } from "react";
import type {
  MarketRateTransmission,
  RateAttribution,
  RateTransmissionTargetBlock,
} from "@/lib/server/types";

/**
 * The transmission subsection of the Rates & Credit workspace.
 *
 * What borrowers pay, split into the Treasury yield underneath and the spread
 * on top; then how much of the recent move came from each, whether moves are
 * reaching borrowers, and which side moves first.
 *
 * Its data arrives inside the workspace payload rather than from a route of its
 * own, so every figure here shares an observation date with the curve above it.
 *
 * Leg colours are literals shared by the bars and their legend so the two
 * cannot disagree, and are kept off the page's semantic palette - cyan, green,
 * red and amber all already mean something on this tab.
 */
const LEG = { base: "#7c93ff", spread: "#f0609e" } as const;

/** Never scale the bars off a move so small the rounding dominates. */
const MIN_BAR_SCALE_BP = 25;

const pct = (value: number) => `${value.toFixed(2)}%`;
/** A spread is a difference between two yields: percentage points, not percent. */
const pp = (value: number) => `${value.toFixed(2)} pp`;
const bp = (value: number) => `${value >= 0 ? "+" : "−"}${Math.abs(Math.round(value))} bp`;

function formatDate(value: string): string {
  const date = new Date(`${value}T00:00:00Z`);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
}

/**
 * One leg as a bar either side of zero. Deliberately not stacked: the legs
 * routinely carry opposite signs - a Treasury selloff met by a narrowing
 * spread is the ordinary case - and stacking would hide exactly that.
 */
function LegBar({ valueBp, scaleBp, color, label }: { valueBp: number; scaleBp: number; color: string; label: string }) {
  const half = 50;
  const width = Math.min(Math.abs(valueBp) / scaleBp, 1) * half;
  return (
    <div className="flex items-center gap-3">
      <span className="w-[92px] shrink-0 text-[10px] text-[color:var(--ink-faint)]">{label}</span>
      <span className="relative h-3 flex-1 rounded-sm bg-[color:rgba(15,32,50,0.7)]">
        <span aria-hidden="true" className="absolute inset-y-0 left-1/2 w-px bg-[color:var(--line)]" />
        <span
          className="absolute inset-y-0 rounded-sm"
          style={{ left: `${valueBp >= 0 ? half : half - width}%`, width: `${Math.max(width, 0.4)}%`, backgroundColor: color }}
        />
      </span>
      <span className="w-[64px] shrink-0 text-right text-[11px] font-semibold tabular-nums text-[color:var(--ink-soft)]">
        {bp(valueBp)}
      </span>
    </div>
  );
}

function Attribution({ value }: { value: RateAttribution }) {
  const scale = Math.max(Math.abs(value.baseBp), Math.abs(value.spreadBp), MIN_BAR_SCALE_BP);
  return (
    <div className="mt-3 flex flex-col gap-2 border-t border-[color:var(--line)] pt-3">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-[10px] text-[color:var(--ink-faint)]">
          {formatDate(value.startDate)} to {formatDate(value.endDate)}
        </span>
        <span className="text-sm font-semibold tabular-nums text-[color:var(--ink)]">{bp(value.totalBp)}</span>
      </div>
      <LegBar valueBp={value.baseBp} scaleBp={scale} color={LEG.base} label="Treasury" />
      <LegBar valueBp={value.spreadBp} scaleBp={scale} color={LEG.spread} label="Spread" />
      <p className="text-[10px] text-[color:var(--ink-faint)]">Components sum to the total by construction.</p>
    </div>
  );
}

/** An estimate, so it always shows the sample it rests on. */
function Estimates({ target }: { target: RateTransmissionTargetBlock }) {
  if (!target.passThrough && !target.leadLag && !target.leadLagNote) return null;
  const reach = target.passThrough;
  return (
    <div className="mt-3 flex flex-col gap-2 border-t border-[color:var(--line)] pt-3">
      {reach && (
        <div>
          <p className="text-[11px] text-[color:var(--ink-soft)]">
            About <span className="font-semibold tabular-nums text-[color:var(--ink)]">{Math.round(reach.beta * 100)}%</span>{" "}
            of each Treasury move has reached this borrower.
          </p>
          <p className="text-[10px] tabular-nums text-[color:var(--ink-faint)]">
            ± {Math.round(reach.stdError * 100)} pts · explains {Math.round(reach.rSquared * 100)}% of the variation · {reach.windowLabel}
          </p>
          {reach.lagNote && <p className="text-[10px] leading-4 text-[color:var(--ink-faint)]">{reach.lagNote}</p>}
        </div>
      )}
      {!target.leadLag && target.leadLagNote && (
        <p className="text-[10px] leading-4 text-[color:var(--ink-faint)]">{target.leadLagNote}</p>
      )}
      {target.leadLag && (
        <div>
          <p className="text-[11px] text-[color:var(--ink-soft)]">{target.leadLag.verdict}</p>
          <p className="text-[10px] tabular-nums text-[color:var(--ink-faint)]">
            Timing only, not cause · correlation {target.leadLag.correlation.toFixed(2)} over {target.leadLag.observations} {target.leadLag.periodLabel}
          </p>
        </div>
      )}
    </div>
  );
}

function TargetCard({ target, baseLabel, windowLabel }: {
  target: RateTransmissionTargetBlock;
  baseLabel: string;
  windowLabel: string;
}) {
  const level = target.level;
  const attributed = target.attribution.find((entry) => entry.window === windowLabel)?.value ?? null;

  if (!level) {
    return (
      <div className="rounded-lg border border-amber-500/25 bg-amber-500/5 p-4">
        <p className="text-sm font-semibold text-amber-200">{target.label}</p>
        <p className="mt-2 text-xs leading-5 text-amber-100/80">{target.unavailableReason}</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.45)] p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-sm font-semibold text-[color:var(--ink)]">{target.label}</p>
        <p className="text-lg font-bold tabular-nums text-[color:var(--ink)]">{pct(level.rate)}</p>
      </div>
      <p className="mt-1 text-xs tabular-nums text-[color:var(--ink-soft)]">
        <span style={{ color: LEG.base }}>{pct(level.base)}</span> {baseLabel}
        {" + "}
        <span style={{ color: LEG.spread }}>{pp(level.spread)}</span> spread
      </p>
      <p className="mt-1 text-[10px] text-[color:var(--ink-faint)]">
        {formatDate(level.observationDate)}
        {level.baseObservationDate !== level.observationDate ? ` · Treasury as of ${formatDate(level.baseObservationDate)}` : ""}
      </p>
      {level.spreadContext && (
        <p className="mt-2 text-[10px] text-[color:var(--ink-faint)]" title={`Percentile of ${level.sampleSize} aligned observations`}>
          Spread: {level.spreadContext.summary}
        </p>
      )}
      {attributed ? <Attribution value={attributed} /> : (
        <p className="mt-3 border-t border-[color:var(--line)] pt-3 text-[10px] text-[color:var(--ink-faint)]">
          Not enough aligned history for this window.
        </p>
      )}
      <Estimates target={target} />
    </div>
  );
}

export function RateTransmissionPanel({ transmission }: { transmission: MarketRateTransmission | null }) {
  // Tolerates a payload from a build either side of this one.
  const windows = transmission?.windows ?? [];
  const [windowLabel, setWindowLabel] = useState<string>("3M");
  if (!transmission) return null;
  const active = windows.includes(windowLabel) ? windowLabel : windows[0] ?? "";

  return (
    <div className="space-y-4 rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.45)] p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-[color:var(--ink)]">What borrowers pay</h3>
          <p className="mt-1 max-w-2xl text-xs leading-5 text-[color:var(--ink-faint)]">
            Each borrowing rate split into the Treasury yield underneath it and the spread on top, then what moved it.
          </p>
        </div>
        <div className="flex items-center gap-1">
          {windows.map((label) => (
            <button
              key={label}
              type="button"
              onClick={() => setWindowLabel(label)}
              className={`rounded-md border px-2 py-1 text-[10px] font-semibold transition ${
                label === active
                  ? "border-[color:rgba(79,213,255,0.35)] bg-[color:rgba(79,213,255,0.12)] text-[color:var(--accent)]"
                  : "border-[color:var(--line)] text-[color:var(--ink-faint)] hover:text-[color:var(--ink-soft)]"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        {transmission.targets.map((target) => (
          <TargetCard key={target.id} target={target} baseLabel={transmission.baseLabel} windowLabel={active} />
        ))}
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {transmission.curve.map((tail) => (
          <div key={tail.id} className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)] p-3">
            <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">{tail.label}</p>
            <p className="mt-1 text-lg font-bold tabular-nums text-[color:var(--ink)]">
              {tail.reading ? pp(tail.reading.value) : "—"}
            </p>
            <p className="mt-1 text-[10px] text-[color:var(--ink-faint)]">{tail.description}</p>
          </div>
        ))}
      </div>

      {transmission.notes.map((note) => (
        <p key={note} className="text-[10px] leading-4 text-[color:var(--ink-faint)]">{note}</p>
      ))}
    </div>
  );
}
