import type { MarketMacroPoint } from "./server/types.ts";
import { percentileOfPoints, type PercentileContext } from "./macro-context.ts";

/**
 * A base observation this far before the target observation is a data gap
 * rather than a holiday. Treasury series skip weekends and federal holidays,
 * so gaps up to four days are ordinary; beyond a week, carrying the last
 * value forward invents a spread rather than reporting one.
 */
export const MAX_BASE_STALENESS_DAYS = 7;

export const RATE_TRANSMISSION_SERIES = [
  { id: "DGS3MO", label: "3M Treasury", license: "public-domain-citation-requested" },
  { id: "DGS2", label: "2Y Treasury", license: "public-domain-citation-requested" },
  { id: "DGS10", label: "10Y Treasury", license: "public-domain-citation-requested" },
  { id: "DGS30", label: "30Y Treasury", license: "public-domain-citation-requested" },
  { id: "DFF", label: "Effective fed funds", license: "public-domain" },
  { id: "MORTGAGE30US", label: "30Y mortgage", license: "existing-project-source" },
  // Moody's Baa over the 10Y. FRED tags this *citation required* - satisfied by
  // the tab's existing FRED attribution - not pre-approval required, which is
  // the ICE BofA OAS tag that genuinely blocks a public route. It already
  // renders as the Financial Conditions "Baa Credit Spread" card on this same
  // page, so omitting it here was the stricter reading of the wrong licence.
  { id: "BAA10Y", label: "Baa corporate spread over 10Y", license: "citation-required" },
] as const;

export interface AlignedRatePoint {
  date: string;
  target: number;
  base: number;
  baseDate: string;
  spread: number;
}

export interface RateDecomposition {
  observationDate: string;
  baseObservationDate: string;
  rate: number;
  base: number;
  spread: number;
  spreadPercentile: number | null;
  /** Full sentence naming the window, e.g. "Higher than 71% of readings since Aug 2021". */
  spreadContext: PercentileContext | null;
  sampleSize: number;
  historyStart: string | null;
}

export interface AttributionResult {
  startDate: string;
  endDate: string;
  /**
   * Whole basis points, with `totalBp === baseBp + spreadBp` exactly.
   *
   * Rounding happens here rather than at render because the panel states the
   * identity in words. Rounding each leg independently at the last moment lets
   * a 12.5 / 8.4 / 4.1 split display as 13 = 8 + 4, and a caption that claims
   * the components sum to the total while the pixels disagree is worse than no
   * caption. The spread leg absorbs the rounding as the remainder.
   */
  totalBp: number;
  baseBp: number;
  spreadBp: number;
}

export interface CurveReading {
  value: number;
  observationDate: string;
  baseObservationDate: string;
}

export interface RateTransmissionData {
  asOf: string;
  generatedAt: string;
  levels: {
    mortgage: RateDecomposition | null;
    corporate: {
      available: boolean;
      reason: string;
      level: RateDecomposition | null;
    };
  };
  curve: {
    shortTail: CurveReading | null;
    belly: CurveReading | null;
    longTail: CurveReading | null;
    policyGap: CurveReading | null;
  };
  attribution: Array<{
    window: "1M" | "3M" | "6M" | "12M";
    mortgage: AttributionResult | null;
    corporate: AttributionResult | null;
  }>;
  warnings: string[];
  sources: Array<{ seriesId: string; label: string; url: string }>;
}

function ordered(points: MarketMacroPoint[]): MarketMacroPoint[] {
  return [...points].sort((left, right) => left.date.localeCompare(right.date));
}

/** Align each target observation with the latest base observation available on or before it. */
export function daysBetween(from: string, to: string): number {
  return Math.round((Date.parse(`${to}T00:00:00Z`) - Date.parse(`${from}T00:00:00Z`)) / 86_400_000);
}

export function alignAsOf(
  basePoints: MarketMacroPoint[],
  targetPoints: MarketMacroPoint[],
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

export function decompose(points: AlignedRatePoint[]): RateDecomposition | null {
  const current = points.at(-1);
  if (!current) return null;
  // One implementation of "where does this sit in its own history", shared with
  // the indicator cards. It always names the window, which matters most exactly
  // where this panel is weakest: the aligned sample is only as deep as the
  // shorter of the two series.
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

export function attributeWindow(points: AlignedRatePoint[], days: number): AttributionResult | null {
  const end = points.at(-1);
  if (!end) return null;
  const cutoff = new Date(`${end.date}T00:00:00Z`);
  cutoff.setUTCDate(cutoff.getUTCDate() - days);
  const cutoffDate = cutoff.toISOString().slice(0, 10);
  const start = points.filter((point) => point.date <= cutoffDate).at(-1);
  if (!start) return null;
  const totalBp = Math.round((end.target - start.target) * 100);
  const baseBp = Math.round((end.base - start.base) * 100);
  return {
    startDate: start.date,
    endDate: end.date,
    totalBp,
    baseBp,
    spreadBp: totalBp - baseBp,
  };
}

export function latestDifference(base: MarketMacroPoint[], target: MarketMacroPoint[]): CurveReading | null {
  const current = alignAsOf(base, target).at(-1);
  return current ? {
    value: current.spread,
    observationDate: current.date,
    baseObservationDate: current.baseDate,
  } : null;
}
