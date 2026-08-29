import type { MarketMacroPoint } from "./server/types.ts";
import type { FedCreditResearchPoint } from "./server/fed-credit-research.ts";

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

export interface PassThroughResult {
  beta: number;
  standardError: number;
  rSquared: number;
  n: number;
  window: number;
}

export interface LeadLagResult {
  bestLag: number;
  correlation: number;
  lagZeroCorrelation: number;
  n: number;
  unit: "weeks";
  verdict: "treasury_leads" | "mortgage_leads" | "no_clear_lead";
}

export interface CreditResearchSnapshot {
  observationDate: string;
  corporateSpread: number;
  excessBondPremium: number;
  defaultRiskComponent: number;
  recessionProbability: number;
  ebpPercentile: number;
  ebpChange3mBp: number | null;
  treasuryChange3mBp: number | null;
  appetite: "risk_averse" | "neutral" | "supportive";
  regime: "restrictive_financing" | "rates_led_tightening" | "flight_to_quality" | "broad_easing" | "mixed";
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
    window: "1M" | "3M" | "6M" | "12M" | "YTD";
    mortgage: AttributionResult | null;
  }>;
  passThrough: { mortgage: PassThroughResult | null };
  leadLag: { mortgageTreasury: LeadLagResult | null };
  creditResearch: CreditResearchSnapshot | null;
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

export function attributeYearToDate(points: AlignedRatePoint[]): AttributionResult | null {
  const end = points.at(-1);
  if (!end) return null;
  const yearStart = `${end.date.slice(0, 4)}-01-01`;
  const start = points.filter((point) => point.date < yearStart).at(-1);
  if (!start) return null;
  return {
    startDate: start.date,
    endDate: end.date,
    totalBp: (end.target - start.target) * 100,
    baseBp: (end.base - start.base) * 100,
    spreadBp: (end.spread - start.spread) * 100,
  };
}

interface RateChange {
  x: number;
  y: number;
}

function changes(points: AlignedRatePoint[]): RateChange[] {
  return points.slice(1).map((point, index) => ({
    x: point.base - points[index].base,
    y: point.target - points[index].target,
  }));
}

export function rollingOls(points: AlignedRatePoint[], window = 52, minimum = 30): PassThroughResult | null {
  const sample = changes(points).slice(-window);
  const n = sample.length;
  if (n < minimum) return null;
  const meanX = sample.reduce((sum, point) => sum + point.x, 0) / n;
  const meanY = sample.reduce((sum, point) => sum + point.y, 0) / n;
  const sxx = sample.reduce((sum, point) => sum + (point.x - meanX) ** 2, 0);
  const sxy = sample.reduce((sum, point) => sum + (point.x - meanX) * (point.y - meanY), 0);
  const syy = sample.reduce((sum, point) => sum + (point.y - meanY) ** 2, 0);
  if (sxx <= Number.EPSILON || syy <= Number.EPSILON || n <= 2) return null;
  const beta = sxy / sxx;
  const intercept = meanY - beta * meanX;
  const residualSumSquares = sample.reduce((sum, point) => sum + (point.y - intercept - beta * point.x) ** 2, 0);
  return {
    beta,
    standardError: Math.sqrt((residualSumSquares / (n - 2)) / sxx),
    rSquared: Math.max(0, Math.min(1, 1 - residualSumSquares / syy)),
    n,
    window,
  };
}

function correlation(pairs: RateChange[]): number | null {
  if (pairs.length < 2) return null;
  const meanX = pairs.reduce((sum, point) => sum + point.x, 0) / pairs.length;
  const meanY = pairs.reduce((sum, point) => sum + point.y, 0) / pairs.length;
  const sxx = pairs.reduce((sum, point) => sum + (point.x - meanX) ** 2, 0);
  const syy = pairs.reduce((sum, point) => sum + (point.y - meanY) ** 2, 0);
  if (sxx <= Number.EPSILON || syy <= Number.EPSILON) return null;
  const sxy = pairs.reduce((sum, point) => sum + (point.x - meanX) * (point.y - meanY), 0);
  return sxy / Math.sqrt(sxx * syy);
}

export function crossCorrelate(
  points: AlignedRatePoint[],
  maxLag = 4,
  minimumPairs = 20,
  threshold = 0.2,
  improvement = 0.05,
): LeadLagResult | null {
  const sample = changes(points);
  const candidates: Array<{ lag: number; correlation: number; n: number }> = [];
  for (let lag = -maxLag; lag <= maxLag; lag += 1) {
    const pairs: RateChange[] = [];
    for (let index = 0; index < sample.length; index += 1) {
      const targetIndex = index + lag;
      if (targetIndex < 0 || targetIndex >= sample.length) continue;
      pairs.push({ x: sample[index].x, y: sample[targetIndex].y });
    }
    if (pairs.length < minimumPairs) continue;
    const value = correlation(pairs);
    if (value !== null) candidates.push({ lag, correlation: value, n: pairs.length });
  }
  const lagZero = candidates.find((candidate) => candidate.lag === 0);
  if (!lagZero || !candidates.length) return null;
  const best = candidates.reduce((current, candidate) => Math.abs(candidate.correlation) > Math.abs(current.correlation) ? candidate : current);
  const clear = best.lag !== 0
    && Math.abs(best.correlation) >= threshold
    && Math.abs(best.correlation) - Math.abs(lagZero.correlation) >= improvement;
  return {
    bestLag: best.lag,
    correlation: best.correlation,
    lagZeroCorrelation: lagZero.correlation,
    n: best.n,
    unit: "weeks",
    verdict: !clear ? "no_clear_lead" : best.lag > 0 ? "treasury_leads" : "mortgage_leads",
  };
}

function valueAsOf(points: MarketMacroPoint[], date: string): MarketMacroPoint | null {
  return ordered(points).filter((point) => point.date <= date).at(-1) ?? null;
}

export function buildCreditResearch(
  points: FedCreditResearchPoint[],
  treasury10y: MarketMacroPoint[],
): CreditResearchSnapshot | null {
  const orderedCredit = [...points].sort((left, right) => left.date.localeCompare(right.date));
  const current = orderedCredit.at(-1);
  if (!current) return null;
  const startDate = new Date(`${current.date}T00:00:00Z`);
  startDate.setUTCMonth(startDate.getUTCMonth() - 3);
  const start = orderedCredit.filter((point) => point.date <= startDate.toISOString().slice(0, 10)).at(-1) ?? null;
  const treasuryEnd = valueAsOf(treasury10y, current.date);
  const treasuryStart = start ? valueAsOf(treasury10y, start.date) : null;
  const ebpChange3mBp = start ? (current.excessBondPremium - start.excessBondPremium) * 100 : null;
  const treasuryChange3mBp = treasuryEnd && treasuryStart ? (treasuryEnd.value - treasuryStart.value) * 100 : null;
  const ebpValues = orderedCredit.map((point) => point.excessBondPremium);
  const ebpPercentile = 100 * ebpValues.filter((value) => value <= current.excessBondPremium).length / ebpValues.length;
  const appetite = current.excessBondPremium >= 0.25 ? "risk_averse" : current.excessBondPremium <= -0.25 ? "supportive" : "neutral";
  const ratesUp = (treasuryChange3mBp ?? 0) >= 15;
  const ratesDown = (treasuryChange3mBp ?? 0) <= -15;
  const creditTighter = (ebpChange3mBp ?? 0) >= 10;
  const creditEasier = (ebpChange3mBp ?? 0) <= -10;
  const regime = ratesUp && creditTighter
    ? "restrictive_financing"
    : ratesUp && !creditTighter
      ? "rates_led_tightening"
      : ratesDown && creditTighter
        ? "flight_to_quality"
        : ratesDown && creditEasier
          ? "broad_easing"
          : "mixed";
  return {
    observationDate: current.date,
    corporateSpread: current.corporateSpread,
    excessBondPremium: current.excessBondPremium,
    defaultRiskComponent: current.defaultRiskComponent,
    recessionProbability: current.recessionProbability,
    ebpPercentile,
    ebpChange3mBp,
    treasuryChange3mBp,
    appetite,
    regime,
  };
}
