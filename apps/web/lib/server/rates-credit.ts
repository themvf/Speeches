import type {
  MarketRatesCreditData,
  MarketRatesCreditDriver,
  MarketRatesCreditGroup,
  MarketRatesCreditMetric,
  MarketRatesCreditPoint,
  MarketRatesCreditSignal,
  MarketRatesCreditTone,
} from "./types.ts";

const FRED_API_BASE = "https://api.stlouisfed.org/fred";
export const RATES_CREDIT_CACHE_SECONDS = 15 * 60;

export interface RatesCreditDefinition {
  id: string;
  seriesId: string;
  label: string;
  shortLabel: string;
  group: MarketRatesCreditGroup;
  tenorYears?: number;
}

interface FredObservationPayload {
  observations?: Array<{ date?: string; value?: string }>;
}

export const RATES_CREDIT_DEFINITIONS: readonly RatesCreditDefinition[] = [
  { id: "treasury_3m", seriesId: "DGS3MO", label: "3-Month Treasury", shortLabel: "3M", group: "treasury", tenorYears: 0.25 },
  { id: "treasury_6m", seriesId: "DGS6MO", label: "6-Month Treasury", shortLabel: "6M", group: "treasury", tenorYears: 0.5 },
  { id: "treasury_1y", seriesId: "DGS1", label: "1-Year Treasury", shortLabel: "1Y", group: "treasury", tenorYears: 1 },
  { id: "treasury_2y", seriesId: "DGS2", label: "2-Year Treasury", shortLabel: "2Y", group: "treasury", tenorYears: 2 },
  { id: "treasury_3y", seriesId: "DGS3", label: "3-Year Treasury", shortLabel: "3Y", group: "treasury", tenorYears: 3 },
  { id: "treasury_5y", seriesId: "DGS5", label: "5-Year Treasury", shortLabel: "5Y", group: "treasury", tenorYears: 5 },
  { id: "treasury_7y", seriesId: "DGS7", label: "7-Year Treasury", shortLabel: "7Y", group: "treasury", tenorYears: 7 },
  { id: "treasury_10y", seriesId: "DGS10", label: "10-Year Treasury", shortLabel: "10Y", group: "treasury", tenorYears: 10 },
  { id: "treasury_20y", seriesId: "DGS20", label: "20-Year Treasury", shortLabel: "20Y", group: "treasury", tenorYears: 20 },
  { id: "treasury_30y", seriesId: "DGS30", label: "30-Year Treasury", shortLabel: "30Y", group: "treasury", tenorYears: 30 },
  { id: "real_5y", seriesId: "DFII5", label: "5-Year Real Yield", shortLabel: "5Y real", group: "real_yield", tenorYears: 5 },
  { id: "real_10y", seriesId: "DFII10", label: "10-Year Real Yield", shortLabel: "10Y real", group: "real_yield", tenorYears: 10 },
  { id: "real_30y", seriesId: "DFII30", label: "30-Year Real Yield", shortLabel: "30Y real", group: "real_yield", tenorYears: 30 },
  { id: "ig_broad", seriesId: "BAMLC0A0CM", label: "US Corporate OAS", shortLabel: "IG", group: "credit_ig" },
  { id: "ig_aaa", seriesId: "BAMLC0A1CAAA", label: "AAA Corporate OAS", shortLabel: "AAA", group: "credit_ig" },
  { id: "ig_aa", seriesId: "BAMLC0A2CAA", label: "AA Corporate OAS", shortLabel: "AA", group: "credit_ig" },
  { id: "ig_a", seriesId: "BAMLC0A3CA", label: "A Corporate OAS", shortLabel: "A", group: "credit_ig" },
  { id: "ig_bbb", seriesId: "BAMLC0A4CBBB", label: "BBB Corporate OAS", shortLabel: "BBB", group: "credit_ig" },
  { id: "hy_broad", seriesId: "BAMLH0A0HYM2", label: "US High Yield OAS", shortLabel: "HY", group: "credit_hy" },
  { id: "hy_bb", seriesId: "BAMLH0A1HYBB", label: "BB High Yield OAS", shortLabel: "BB", group: "credit_hy" },
  { id: "hy_b", seriesId: "BAMLH0A2HYB", label: "B High Yield OAS", shortLabel: "B", group: "credit_hy" },
  { id: "hy_ccc", seriesId: "BAMLH0A3HYC", label: "CCC & Lower High Yield OAS", shortLabel: "CCC", group: "credit_hy" },
];

function parseObservations(payload: FredObservationPayload): MarketRatesCreditPoint[] {
  return (payload.observations ?? [])
    .flatMap((observation) => {
      const value = Number(observation.value);
      return observation.date && Number.isFinite(value) ? [{ date: observation.date, value }] : [];
    })
    .sort((left, right) => left.date.localeCompare(right.date));
}

function changeFrom(points: MarketRatesCreditPoint[], observationsBack: number): number | null {
  if (points.length <= observationsBack) return null;
  return points[points.length - 1].value - points[points.length - 1 - observationsBack].value;
}

function percentile(values: number[], current: number): number | null {
  if (values.length < 20) return null;
  return (values.filter((value) => value <= current).length / values.length) * 100;
}

function zScore(values: number[], current: number): number | null {
  if (values.length < 20) return null;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / values.length;
  const standardDeviation = Math.sqrt(variance);
  return standardDeviation > 0 ? (current - mean) / standardDeviation : 0;
}

export function buildRatesCreditMetric(
  definition: RatesCreditDefinition,
  rawPoints: MarketRatesCreditPoint[],
): MarketRatesCreditMetric {
  const points = [...rawPoints].sort((left, right) => left.date.localeCompare(right.date));
  const current = points.at(-1);
  if (!current) throw new Error(`FRED returned no observations for ${definition.seriesId}`);
  const values = points.map((point) => point.value);
  return {
    id: definition.id,
    fredSeriesId: definition.seriesId,
    label: definition.label,
    shortLabel: definition.shortLabel,
    group: definition.group,
    tenorYears: definition.tenorYears ?? null,
    value: current.value,
    change1d: changeFrom(points, 1),
    change1w: changeFrom(points, 5),
    change1m: changeFrom(points, 21),
    percentile: percentile(values, current.value),
    zScore: zScore(values, current.value),
    observationDate: current.date,
    points,
    sourceUrl: `https://fred.stlouisfed.org/series/${definition.seriesId}`,
  };
}

function metricById(metrics: MarketRatesCreditMetric[], id: string): MarketRatesCreditMetric | undefined {
  return metrics.find((metric) => metric.id === id);
}

function bp(value: number | null): string {
  if (value === null) return "n/a";
  const basisPoints = value * 100;
  return `${basisPoints >= 0 ? "+" : ""}${basisPoints.toFixed(0)} bp`;
}

function absoluteBp(value: number): string {
  return `${Math.abs(value * 100).toFixed(0)} bp`;
}

function makeRatesSignal(metrics: MarketRatesCreditMetric[]): MarketRatesCreditSignal {
  const twoYear = metricById(metrics, "treasury_2y");
  const tenYear = metricById(metrics, "treasury_10y");
  const changes = [twoYear?.change1m, tenYear?.change1m].filter((value): value is number => value !== null && value !== undefined);
  if (changes.length < 2) {
    return { id: "rates", label: "Rates", state: "Insufficient data", tone: "neutral", summary: "Both 2Y and 10Y one-month histories are required to classify the rates regime." };
  }
  const average = changes.length ? changes.reduce((sum, value) => sum + value, 0) / changes.length : 0;
  if (average >= 0.1) {
    return { id: "rates", label: "Rates", state: "Yields rising", tone: "negative", summary: `The 2Y/10Y complex rose ${absoluteBp(average)} on average over one month.` };
  }
  if (average <= -0.1) {
    return { id: "rates", label: "Rates", state: "Yields falling", tone: "positive", summary: `The 2Y/10Y complex fell ${absoluteBp(average)} on average over one month.` };
  }
  return { id: "rates", label: "Rates", state: "Range-bound", tone: "neutral", summary: `The average one-month 2Y/10Y move is ${bp(average)}.` };
}

export function classifyCurve(metrics: MarketRatesCreditMetric[]): MarketRatesCreditSignal {
  const twoYear = metricById(metrics, "treasury_2y");
  const tenYear = metricById(metrics, "treasury_10y");
  if (!twoYear || !tenYear || twoYear.change1m === null || tenYear.change1m === null) {
    return { id: "curve", label: "Curve", state: "Insufficient data", tone: "neutral", summary: "The 2Y and 10Y history needed to classify the curve is incomplete." };
  }

  const shortMove = twoYear.change1m;
  const longMove = tenYear.change1m;
  const slope = tenYear.value - twoYear.value;
  const slopeChange = longMove - shortMove;
  const threshold = 0.02;
  let state = "Parallel / mixed";

  if (shortMove >= threshold && longMove >= threshold) {
    state = slopeChange >= threshold ? "Bear steepener" : slopeChange <= -threshold ? "Bear flattener" : "Parallel selloff";
  } else if (shortMove <= -threshold && longMove <= -threshold) {
    state = slopeChange >= threshold ? "Bull steepener" : slopeChange <= -threshold ? "Bull flattener" : "Parallel rally";
  } else if (slopeChange >= threshold) {
    state = "Curve steepening";
  } else if (slopeChange <= -threshold) {
    state = "Curve flattening";
  }

  const tone: MarketRatesCreditTone = slope < 0 ? "negative" : "neutral";
  return {
    id: "curve",
    label: "Curve",
    state,
    tone,
    summary: `2s10s is ${bp(slope)}; its one-month change is ${bp(slopeChange)}.`,
  };
}

function makeCreditSignal(metrics: MarketRatesCreditMetric[]): MarketRatesCreditSignal {
  const investmentGrade = metricById(metrics, "ig_broad");
  const highYield = metricById(metrics, "hy_broad");
  if (!investmentGrade || !highYield) {
    return { id: "credit", label: "Credit", state: "Insufficient data", tone: "neutral", summary: "Broad IG and HY spread observations are incomplete." };
  }

  const hyMove = highYield.change1m ?? 0;
  const igMove = investmentGrade.change1m ?? 0;
  const bb = metricById(metrics, "hy_bb");
  const ccc = metricById(metrics, "hy_ccc");
  const qualityChange = bb?.change1m !== null && bb?.change1m !== undefined && ccc?.change1m !== null && ccc?.change1m !== undefined
    ? ccc.change1m - bb.change1m
    : null;
  const creditMetrics = metrics.filter((metric) => metric.group === "credit_ig" || metric.group === "credit_hy");
  const observedMoves = creditMetrics.flatMap((metric) => metric.change1m === null ? [] : [metric.change1m]);
  const wideningBreadth = observedMoves.length ? observedMoves.filter((change) => change > 0).length / observedMoves.length : null;
  const highAndWide = highYield.percentile !== null && highYield.percentile >= 90 && highYield.value >= 6;
  const qualityStress = qualityChange !== null && qualityChange >= 0.5;
  const context = `HY ${bp(hyMove)}, IG ${bp(igMove)}, breadth ${wideningBreadth === null ? "n/a" : `${Math.round(wideningBreadth * 100)}% widening`}${qualityChange === null ? "" : `, CCC-BB ${bp(qualityChange)}`}.`;
  if (hyMove >= 0.5 || highAndWide || qualityStress) {
    return { id: "credit", label: "Credit", state: "Stressed", tone: "negative", summary: context };
  }
  if (hyMove >= 0.15 || igMove >= 0.08 || (wideningBreadth !== null && wideningBreadth >= 0.75 && hyMove > 0.05)) {
    return { id: "credit", label: "Credit", state: "Deteriorating", tone: "negative", summary: context };
  }
  if (hyMove <= -0.15 || igMove <= -0.08) {
    return { id: "credit", label: "Credit", state: "Improving", tone: "positive", summary: context };
  }
  return { id: "credit", label: "Credit", state: "Stable", tone: "neutral", summary: context };
}

function makeCompositeSignal(signals: MarketRatesCreditSignal[]): MarketRatesCreditSignal {
  const negative = signals.filter((signal) => signal.tone === "negative").length;
  const positive = signals.filter((signal) => signal.tone === "positive").length;
  const credit = signals.find((signal) => signal.id === "credit");
  if (credit?.state === "Stressed") {
    return { id: "composite", label: "Composite", state: "Dislocation risk", tone: "negative", summary: "Credit stress is the dominant condition and warrants closer monitoring." };
  }
  if (negative >= 2) {
    return { id: "composite", label: "Composite", state: "Tightening", tone: "negative", summary: `${negative} of the three observable rates and credit signals are adverse.` };
  }
  if (positive >= 2) {
    return { id: "composite", label: "Composite", state: "Easing", tone: "positive", summary: `${positive} of the three observable rates and credit signals are constructive.` };
  }
  return { id: "composite", label: "Composite", state: "Mixed", tone: "neutral", summary: "Rates, curve, and credit signals are not moving in a single direction." };
}

export function buildRatesCreditSignals(metrics: MarketRatesCreditMetric[]): MarketRatesCreditSignal[] {
  const signals = [makeRatesSignal(metrics), classifyCurve(metrics), makeCreditSignal(metrics)];
  return [...signals, makeCompositeSignal(signals)];
}

function buildDrivers(metrics: MarketRatesCreditMetric[]): MarketRatesCreditDriver[] {
  const candidateIds = new Set(["treasury_2y", "treasury_10y", "real_10y", "ig_broad", "hy_broad", "hy_ccc"]);
  return metrics
    .filter((metric) => candidateIds.has(metric.id) && metric.change1m !== null)
    .sort((left, right) => Math.abs(right.change1m ?? 0) - Math.abs(left.change1m ?? 0))
    .slice(0, 4)
    .map((metric) => {
      const change = metric.change1m ?? 0;
      const isSpread = metric.group === "credit_ig" || metric.group === "credit_hy";
      const verb = change >= 0 ? (isSpread ? "widened" : "rose") : (isSpread ? "tightened" : "fell");
      return {
        label: metric.shortLabel,
        detail: `${metric.label} ${verb} ${Math.abs(change * 100).toFixed(0)} bp over one month.`,
        tone: change === 0 ? "neutral" : change > 0 ? "negative" : "positive",
      } satisfies MarketRatesCreditDriver;
    });
}

async function fetchMetric(
  definition: RatesCreditDefinition,
  apiKey: string,
  historicalPoints: MarketRatesCreditPoint[] = [],
): Promise<MarketRatesCreditMetric> {
  const url = new URL(`${FRED_API_BASE}/series/observations`);
  url.searchParams.set("api_key", apiKey);
  url.searchParams.set("file_type", "json");
  url.searchParams.set("series_id", definition.seriesId);
  url.searchParams.set("sort_order", "desc");
  url.searchParams.set("limit", "800");

  const response = await fetch(url, { next: { revalidate: RATES_CREDIT_CACHE_SECONDS }, signal: AbortSignal.timeout(10_000) });
  if (!response.ok) throw new Error(`FRED ${definition.seriesId} returned HTTP ${response.status}`);
  const payload = await response.json() as FredObservationPayload;
  const merged = new Map(historicalPoints.map((point) => [point.date, point]));
  for (const point of parseObservations(payload)) merged.set(point.date, point);
  return buildRatesCreditMetric(definition, [...merged.values()]);
}

export async function fetchRatesCreditData(
  apiKey = String(process.env.FRED_API_KEY ?? "").trim(),
  includeIceData = String(process.env.RATES_CREDIT_ICE_DATA_ENABLED ?? "").trim().toLowerCase() === "true",
  historyBySeries: Record<string, MarketRatesCreditPoint[]> = {},
): Promise<MarketRatesCreditData> {
  if (!apiKey) throw new Error("FRED_API_KEY is not configured.");

  const activeDefinitions = RATES_CREDIT_DEFINITIONS.filter((definition) => includeIceData || !definition.group.startsWith("credit_"));
  const results = await Promise.allSettled(activeDefinitions.map((definition) => fetchMetric(definition, apiKey, historyBySeries[definition.seriesId] ?? [])));
  const metrics = results.flatMap((result) => result.status === "fulfilled" ? [result.value] : []);
  if (!metrics.length) {
    const failure = results.find((result) => result.status === "rejected");
    throw failure && failure.status === "rejected" ? failure.reason : new Error("FRED returned no rates or credit observations.");
  }

  const failedDefinitions = results.flatMap((result, index) => result.status === "rejected" ? [activeDefinitions[index].shortLabel] : []);
  const warnings = failedDefinitions.length ? [`Unavailable series: ${failedDefinitions.join(", ")}.`] : [];
  if (!includeIceData) warnings.push("ICE BofA credit spreads are disabled pending authorized data rights for this deployment.");
  for (const group of ["treasury", "real_yield", "credit_ig", "credit_hy"] satisfies MarketRatesCreditGroup[]) {
    if (!includeIceData && group.startsWith("credit_")) continue;
    if (!metrics.some((metric) => metric.group === group)) warnings.push(`No ${group.replace("_", " ")} data is currently available.`);
    const dates = metrics.filter((metric) => metric.group === group).map((metric) => metric.observationDate).sort();
    if (dates.length > 1) {
      const oldest = new Date(`${dates[0]}T00:00:00Z`).getTime();
      const newest = new Date(`${dates[dates.length - 1]}T00:00:00Z`).getTime();
      if ((newest - oldest) / 86_400_000 > 4) warnings.push(`${group.replace("_", " ")} observations are not aligned to the same market week.`);
    }
  }

  return {
    treasuryCurve: metrics.filter((metric) => metric.group === "treasury").sort((left, right) => (left.tenorYears ?? 0) - (right.tenorYears ?? 0)),
    realYields: metrics.filter((metric) => metric.group === "real_yield").sort((left, right) => (left.tenorYears ?? 0) - (right.tenorYears ?? 0)),
    investmentGrade: metrics.filter((metric) => metric.group === "credit_ig"),
    highYield: metrics.filter((metric) => metric.group === "credit_hy"),
    signals: buildRatesCreditSignals(metrics),
    drivers: buildDrivers(metrics),
    generatedAt: new Date().toISOString(),
    cacheSeconds: RATES_CREDIT_CACHE_SECONDS,
    source: "FRED",
    creditDataStatus: includeIceData ? "enabled" : "license_required",
    warnings,
  };
}
