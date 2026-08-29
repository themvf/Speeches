import type {
  MarketMacroPoint,
  RateAttribution,
  RateCurveReading,
  RateDecomposition,
  RateLeadLag,
  RatePassThrough,
} from "./server/types.ts";
import { percentileOfPoints } from "./macro-context.ts";

/**
 * Rate transmission: how Treasury moves reach mortgage and corporate borrowers.
 *
 * Every borrowing rate is a Treasury yield plus a spread, and the two have
 * different causes. This module separates them, then measures two things about
 * the relationship.
 *
 * The three layers differ in how wrong they can be, and are treated
 * accordingly:
 *
 *   1. The level split and the window attribution are ACCOUNTING. Rate = base +
 *      spread is definitional; there is no coefficient to get wrong.
 *   2. Pass-through is an ESTIMATE. It carries its standard error, its fit and
 *      its sample size, and disappears below a minimum rather than degrading.
 *   3. Lead/lag is an ESTIMATE ABOUT TIMING and the easiest to fool yourself
 *      with. See the warning on `leadLag` before touching it.
 *
 * Pure functions only - no fetching, no React - so the route and the panel
 * cannot compute different numbers from the same inputs.
 */

/**
 * A base observation this far before the target observation is a data gap
 * rather than a holiday. Treasury series skip weekends and federal holidays,
 * so gaps up to four days are ordinary; beyond a week, carrying the last value
 * forward invents a spread rather than reporting one.
 */
export const MAX_BASE_STALENESS_DAYS = 7;

/** Below this a pass-through estimate is noise with a decimal point. */
export const MIN_PASS_THROUGH_OBSERVATIONS = 30;

/** Below this a lead/lag correlation is not worth naming a leader over. */
export const MIN_LEAD_LAG_CORRELATION = 0.2;

/** A lag must beat lag zero by this much before we call it a lead. */
const LEAD_MARGIN = 0.03;

export const RATE_TRANSMISSION_SERIES = [
  { id: "DGS3MO", label: "3M Treasury", license: "public-domain-citation-requested" },
  { id: "DGS2", label: "2Y Treasury", license: "public-domain-citation-requested" },
  { id: "DGS10", label: "10Y Treasury", license: "public-domain-citation-requested" },
  { id: "DGS30", label: "30Y Treasury", license: "public-domain-citation-requested" },
  { id: "DFF", label: "Effective fed funds", license: "public-domain" },
  { id: "MORTGAGE30US", label: "30Y mortgage", license: "existing-project-source" },
  // Moody's Baa over the 10Y. FRED tags this *citation required* - satisfied by
  // the tab's existing FRED attribution - not pre-approval required, which is
  // the ICE BofA OAS tag that genuinely blocks a public route.
  { id: "BAA10Y", label: "Baa corporate spread over 10Y", license: "citation-required" },
] as const;

export interface AlignedRatePoint {
  date: string;
  target: number;
  base: number;
  baseDate: string;
  spread: number;
}

export function daysBetween(from: string, to: string): number {
  return Math.round((Date.parse(`${to}T00:00:00Z`) - Date.parse(`${from}T00:00:00Z`)) / 86_400_000);
}

function ordered(points: readonly MarketMacroPoint[]): MarketMacroPoint[] {
  return [...points].sort((left, right) => left.date.localeCompare(right.date));
}

/**
 * Pair each target observation with the last base observation on or before it.
 *
 * This as-of join is the load-bearing part of the module. The mortgage rate is
 * weekly (Thursdays); Treasury yields are daily with holes on holidays. Walking
 * the two arrays in step by index does not throw and does not look wrong - it
 * silently pairs each rate with an arbitrary yield. On real data that produced
 * a spread wrong by up to 54 basis points, trending where the truth was flat.
 */
export function alignAsOf(
  basePoints: readonly MarketMacroPoint[],
  targetPoints: readonly MarketMacroPoint[],
  maxStalenessDays = MAX_BASE_STALENESS_DAYS,
): AlignedRatePoint[] {
  const base = ordered(basePoints);
  const target = ordered(targetPoints);
  const aligned: AlignedRatePoint[] = [];
  let baseIndex = 0;
  let latestBase: MarketMacroPoint | null = null;

  for (const point of target) {
    while (baseIndex < base.length && base[baseIndex].date <= point.date) {
      latestBase = base[baseIndex];
      baseIndex += 1;
    }
    if (!latestBase) continue;
    if (daysBetween(latestBase.date, point.date) > maxStalenessDays) continue;
    aligned.push({
      date: point.date,
      target: point.value,
      base: latestBase.value,
      baseDate: latestBase.date,
      spread: point.value - latestBase.value,
    });
  }
  return aligned;
}

/** Rebuild a yield level from a published spread plus its base. */
export function levelFromSpread(
  spread: readonly MarketMacroPoint[],
  base: readonly MarketMacroPoint[],
): MarketMacroPoint[] {
  return alignAsOf(base, spread).map((point) => ({ date: point.date, value: point.target + point.base }));
}

export function decompose(points: readonly AlignedRatePoint[]): RateDecomposition | null {
  const current = points.at(-1);
  if (!current) return null;
  // One implementation of "where does this sit in its own history", shared with
  // the indicator cards. It always names the window, which matters most here:
  // the aligned sample is only as deep as the shorter of the two series.
  const spreadPoints = points
    .filter((point) => Number.isFinite(point.spread))
    .map((point) => ({ date: point.date, value: point.spread }));
  const spreadContext = percentileOfPoints(spreadPoints, current.spread);
  return {
    observationDate: current.date,
    baseObservationDate: current.baseDate,
    rate: current.target,
    base: current.base,
    spread: current.spread,
    spreadPercentile: spreadContext?.percentile ?? null,
    spreadContext,
    sampleSize: spreadPoints.length,
    historyStart: points[0]?.date ?? null,
  };
}

/**
 * Split a window's move into its Treasury leg and its spread leg.
 *
 * Rounding happens here, not at render, because the panel states the identity
 * in words. Rounding each leg independently at the last moment lets a
 * 12.5 / 8.4 / 4.1 split display as 13 = 8 + 4, and a caption claiming the
 * components sum to the total while the pixels disagree is worse than no
 * caption. The spread leg absorbs the rounding as the remainder.
 */
export function attributeWindow(points: readonly AlignedRatePoint[], days: number): RateAttribution | null {
  const end = points.at(-1);
  if (!end) return null;
  const cutoff = new Date(`${end.date}T00:00:00Z`);
  cutoff.setUTCDate(cutoff.getUTCDate() - days);
  const cutoffDate = cutoff.toISOString().slice(0, 10);
  const start = points.filter((point) => point.date <= cutoffDate).at(-1);
  if (!start) return null;

  const totalBp = Math.round((end.target - start.target) * 100);
  const baseBp = Math.round((end.base - start.base) * 100);
  return { startDate: start.date, endDate: end.date, totalBp, baseBp, spreadBp: totalBp - baseBp };
}

export function latestDifference(
  base: readonly MarketMacroPoint[],
  target: readonly MarketMacroPoint[],
): RateCurveReading | null {
  const current = alignAsOf(base, target).at(-1);
  return current
    ? { value: current.spread, observationDate: current.date, baseObservationDate: current.baseDate }
    : null;
}

// ── Estimates ────────────────────────────────────────────────────────────────

/** Period-over-period changes. Regressing on levels would fit a trend, not a relationship. */
function changes(values: readonly number[]): number[] {
  return values.slice(1).map((value, index) => value - values[index]);
}

function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function correlation(xs: readonly number[], ys: readonly number[]): number | null {
  if (xs.length < 3 || xs.length !== ys.length) return null;
  const mx = mean(xs);
  const my = mean(ys);
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < xs.length; i += 1) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  if (sxx === 0 || syy === 0) return null;
  return sxy / Math.sqrt(sxx * syy);
}

/**
 * How much of a Treasury move reaches this borrower.
 *
 * A least-squares fit of the borrower's rate changes on the Treasury's, over
 * the most recent `window` observations. A slope near 1 means moves pass
 * through roughly intact; well below 1 means the spread is absorbing them.
 *
 * `lagPeriods` shifts the Treasury change back, for a borrowing rate that is
 * measured later than the yield it responds to. This is not optional tuning:
 * Freddie Mac surveys lenders about quotes made earlier in the week, so a
 * same-week comparison mechanically misses most of the response. On real data
 * the same-week fit came out at 25% with an R-squared of 0.08 while the
 * one-week-lagged relationship correlated at 0.67 - so the unlagged number
 * would have understated pass-through by roughly a factor of three and looked
 * like a finding about lenders rather than an artefact of survey timing.
 *
 * Returns null rather than a thin estimate: below the minimum sample, and when
 * the Treasury barely moved over the window (no variation to explain), there is
 * no honest number to report.
 */
export function passThrough(
  points: readonly AlignedRatePoint[],
  window: number,
  windowLabel: string,
  lagPeriods = 0,
  lagNote: string | null = null,
): RatePassThrough | null {
  const recent = points.slice(-(window + 1 + lagPeriods));
  const allBase = changes(recent.map((point) => point.base));
  const allTarget = changes(recent.map((point) => point.target));
  // Pair each target change with the base change `lagPeriods` earlier.
  const xs = lagPeriods > 0 ? allBase.slice(0, allBase.length - lagPeriods) : allBase;
  const ys = lagPeriods > 0 ? allTarget.slice(lagPeriods) : allTarget;
  if (xs.length < MIN_PASS_THROUGH_OBSERVATIONS) return null;

  const mx = mean(xs);
  const my = mean(ys);
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < xs.length; i += 1) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
  }
  if (sxx === 0) return null;

  const beta = sxy / sxx;
  const intercept = my - beta * mx;
  let sse = 0;
  let sst = 0;
  for (let i = 0; i < xs.length; i += 1) {
    sse += (ys[i] - (intercept + beta * xs[i])) ** 2;
    sst += (ys[i] - my) ** 2;
  }
  const stdError = Math.sqrt(sse / Math.max(xs.length - 2, 1) / sxx);
  return {
    beta,
    stdError,
    rSquared: sst === 0 ? 0 : Math.max(0, 1 - sse / sst),
    observations: xs.length,
    windowLabel: `${xs.length} ${windowLabel}`,
    lagPeriods,
    lagNote: lagPeriods > 0 ? lagNote : null,
  };
}

/**
 * Which of two borrowing rates tends to move first.
 *
 * **Both arguments must be yield LEVELS. Never pass a spread and the yield it
 * was derived from.** A spread is defined as one yield minus another, so
 * correlating it against its own subtrahend carries a built-in loading of -1
 * and will always "show" the two moving in opposite directions. That is
 * arithmetic, not economics, and it produces a confident, meaningless result.
 *
 * The same trap has a mirror image that is easier to miss: a yield LEVEL
 * rebuilt as `spread + base` also contains the base by construction, so it is
 * guaranteed to correlate positively with it. Only a series observed
 * independently of the base can answer this question. Callers must check that
 * before calling; `independentOfBase` in the route config is where that is
 * decided.
 *
 * A positive lag means `first` leads `second` by that many observation periods.
 * The verdict names a leader only when the correlation clears a floor and beats
 * lag zero by a margin; otherwise it says so. Timing only - never cause.
 */
export function leadLag(
  first: readonly MarketMacroPoint[],
  second: readonly MarketMacroPoint[],
  maxLag: number,
  periodLabel: string,
  firstLabel: string,
  secondLabel: string,
): RateLeadLag | null {
  // Inner join on exact dates: both inputs are daily series on the same
  // publication calendar, so an as-of join here would pair a date with itself.
  const secondByDate = new Map(second.map((point) => [point.date, point.value]));
  const common = ordered(first).flatMap((point) => {
    const other = secondByDate.get(point.date);
    return other === undefined ? [] : [{ a: point.value, b: other }];
  });
  const da = changes(common.map((pair) => pair.a));
  const db = changes(common.map((pair) => pair.b));
  if (da.length < MIN_PASS_THROUGH_OBSERVATIONS) return null;

  let best = { lag: 0, r: 0 };
  let atZero = 0;
  for (let lag = -maxLag; lag <= maxLag; lag += 1) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < da.length; i += 1) {
      const j = i + lag;
      if (j < 0 || j >= db.length) continue;
      xs.push(da[i]);
      ys.push(db[j]);
    }
    const r = correlation(xs, ys);
    if (r === null) continue;
    if (lag === 0) atZero = r;
    if (Math.abs(r) > Math.abs(best.r)) best = { lag, r };
  }

  /** "1 week", not "1 weeks". */
  const periods = (count: number) =>
    `${count} ${Math.abs(count) === 1 ? periodLabel.replace(/s$/, "") : periodLabel}`;

  const strong = Math.abs(best.r) >= MIN_LEAD_LAG_CORRELATION;
  const beatsZero = Math.abs(best.r) - Math.abs(atZero) >= LEAD_MARGIN;
  let verdict: string;
  if (!strong) {
    verdict = "No clear timing relationship in this window.";
  } else if (best.lag === 0 || !beatsZero) {
    verdict = `${firstLabel} and ${secondLabel} move together, same day.`;
  } else if (best.lag > 0) {
    verdict = `${firstLabel} has moved ${periods(best.lag)} ahead of ${secondLabel}.`;
  } else {
    verdict = `${secondLabel} has moved ${periods(Math.abs(best.lag))} ahead of ${firstLabel}.`;
  }

  return {
    bestLagPeriods: strong && beatsZero ? best.lag : 0,
    correlation: best.r,
    observations: da.length,
    periodLabel,
    verdict,
  };
}
