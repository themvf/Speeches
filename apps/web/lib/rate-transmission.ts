import type { MarketMacroPoint } from "./server/types.ts";

export const RATE_TRANSMISSION_SERIES = [
  { id: "DGS3MO", label: "3M Treasury", license: "public-domain-citation-requested" },
  { id: "DGS2", label: "2Y Treasury", license: "public-domain-citation-requested" },
  { id: "DGS10", label: "10Y Treasury", license: "public-domain-citation-requested" },
  { id: "DGS30", label: "30Y Treasury", license: "public-domain-citation-requested" },
  { id: "DFF", label: "Effective fed funds", license: "public-domain" },
  { id: "MORTGAGE30US", label: "30Y mortgage", license: "existing-project-source" },
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
  sampleSize: number;
  historyStart: string | null;
}

export interface AttributionResult {
  startDate: string;
  endDate: string;
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
      available: false;
      reason: string;
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
  }>;
  warnings: string[];
  sources: Array<{ seriesId: string; label: string; url: string }>;
}

function ordered(points: MarketMacroPoint[]): MarketMacroPoint[] {
  return [...points].sort((left, right) => left.date.localeCompare(right.date));
}

/** Align each target observation with the latest base observation available on or before it. */
export function alignAsOf(basePoints: MarketMacroPoint[], targetPoints: MarketMacroPoint[]): AlignedRatePoint[] {
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
  const spreads = points.map((point) => point.spread).filter(Number.isFinite);
  const spreadPercentile = spreads.length >= 12
    ? 100 * spreads.filter((value) => value <= current.spread).length / spreads.length
    : null;
  return {
    observationDate: current.date,
    baseObservationDate: current.baseDate,
    rate: current.target,
    base: current.base,
    spread: current.spread,
    spreadPercentile,
    sampleSize: spreads.length,
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
  return {
    startDate: start.date,
    endDate: end.date,
    totalBp: (end.target - start.target) * 100,
    baseBp: (end.base - start.base) * 100,
    spreadBp: (end.spread - start.spread) * 100,
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
