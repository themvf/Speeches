import {
  alignAsOf,
  attributeWindow,
  decompose,
  latestDifference,
  leadLag,
  levelFromSpread,
  passThrough,
} from "../rate-transmission.ts";
import { fetchFredSeriesPoints, seriesRefreshSeconds, type FredCadence } from "./fred-macro.ts";
import type {
  MarketRateTransmission,
  MarketRatesCreditData,
  MarketRatesCreditDriver,
  MarketRatesCreditGroup,
  MarketRatesCreditMetric,
  MarketRatesCreditPoint,
  MarketRatesCreditSignal,
  MarketRatesCreditTone,
  RateTransmissionTargetBlock,
} from "./types.ts";

export const RATES_CREDIT_CACHE_SECONDS = 15 * 60;

/** Observations per series. Deep enough that a percentile means something. */
const SERIES_LIMIT = 1_500;

export interface RatesCreditDefinition {
  id: string;
  seriesId: string;
  label: string;
  shortLabel: string;
  group: MarketRatesCreditGroup;
  tenorYears?: number;
  /**
   * How often the source publishes. Everything here is a daily market rate
   * except the mortgage survey, which prints once a week and does not need
   * asking 96 times a day.
   */
  cadence?: FredCadence;
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
  // What borrowers actually pay, plus the policy rate the front end is measured
  // against. These feed the transmission block below; the Treasury legs it needs
  // are already in this list, so nothing is fetched twice.
  { id: "mortgage_30y", seriesId: "MORTGAGE30US", label: "30-Year Fixed Mortgage", shortLabel: "30Y mortgage", group: "borrowing", cadence: "weekly" },
  { id: "baa_spread", seriesId: "BAA10Y", label: "Baa Corporate Spread over 10Y", shortLabel: "Baa spread", group: "borrowing" },
  { id: "fed_funds", seriesId: "DFF", label: "Effective Fed Funds Rate", shortLabel: "Fed funds", group: "borrowing" },
];

/**
 * Lookback windows for the change attribution. Calendar days rather than
 * observation counts, so a window means the same elapsed time for the weekly
 * mortgage series and the daily corporate one.
 */
const ATTRIBUTION_WINDOWS: ReadonlyArray<{ label: string; days: number }> = [
  { label: "1M", days: 30 },
  { label: "3M", days: 91 },
  { label: "6M", days: 182 },
  { label: "12M", days: 365 },
];

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


/**
 * The transmission block: what borrowers pay, split into the Treasury yield
 * underneath and the spread on top.
 *
 * Built from the metrics this route already fetched, so every figure here
 * shares an observation date with the curve above it. Nothing is requested
 * twice and nothing can be an hour out of step with its neighbour.
 */
function buildTransmission(metrics: MarketRatesCreditMetric[]): MarketRateTransmission | null {
  const points = (id: string) => metrics.find((metric) => metric.id === id)?.points ?? [];
  const treasury10y = points("treasury_10y");
  if (!treasury10y.length) return null;

  const mortgage = alignAsOf(treasury10y, points("mortgage_30y"));
  // The Baa yield level is rebuilt from FRED's published spread rather than
  // fetched separately and re-spread here, so the spread shown is the exact
  // number FRED serves and the Baa card renders.
  const baaSpread = points("baa_spread");
  const baaLevel = alignAsOf(treasury10y, levelFromSpread(baaSpread, treasury10y));

  const targets: RateTransmissionTargetBlock[] = [
    {
      id: "mortgage_30y",
      label: "30Y mortgage",
      aligned: mortgage,
      // Weekly survey: a year is 52 observations, and ten weeks either side is
      // a generous window for a series that only prints once a week.
      passThroughWindow: 52,
      passThroughLabel: "weekly changes",
      // The survey covers quotes from earlier in its week, so the Treasury
      // change it responds to is the previous week's.
      passThroughLag: 1,
      passThroughLagNote: "Treasury change taken one week earlier, because the survey covers quotes made before it was published.",
      leadLagMax: 4,
      leadLagPeriod: "weeks",
      // Freddie Mac surveys lenders directly, so this rate is observed
      // independently of the Treasury yield it is compared against.
      independentOfBase: true,
      leadLagNote: null,
      missing: "The mortgage survey or the 10-year Treasury is unavailable.",
    },
    {
      id: "baa_corporate",
      label: "Baa corporate",
      aligned: baaLevel,
      passThroughWindow: 120,
      passThroughLabel: "daily changes",
      passThroughLag: 0,
      passThroughLagNote: null,
      leadLagMax: 10,
      leadLagPeriod: "days",
      // FRED publishes the Baa SPREAD, so the yield level here is rebuilt as
      // spread + 10Y and therefore contains the 10Y by construction. Comparing
      // the two would return a strong positive correlation that is arithmetic,
      // not timing. Answering this properly needs an independently observed Baa
      // yield (DBAA), whose FRED licence tag has not been checked - and this
      // route is public, so it is not added on an assumption.
      independentOfBase: false,
      leadLagNote: "Timing needs a corporate yield observed independently of the Treasury; the published series here is a spread, so comparing it would only restate the arithmetic.",
      missing: "The Baa spread or the 10-year Treasury is unavailable.",
    },
  ].map((target) => {
    const level = decompose(target.aligned);
    // Lead/lag runs on the two yield LEVELS, never on a spread against the
    // yield it was derived from - that comparison carries a mechanical -1.
    const levels = target.aligned.map((point) => ({ date: point.date, value: point.target }));
    return {
      id: target.id,
      label: target.label,
      level,
      attribution: ATTRIBUTION_WINDOWS.map((window) => ({
        window: window.label,
        value: attributeWindow(target.aligned, window.days),
      })),
      passThrough: passThrough(target.aligned, target.passThroughWindow, target.passThroughLabel, target.passThroughLag, target.passThroughLagNote),
      leadLag: target.independentOfBase
        ? leadLag(treasury10y, levels, target.leadLagMax, target.leadLagPeriod, "The 10-year Treasury", target.label)
        : null,
      leadLagNote: target.leadLagNote,
      unavailableReason: level ? null : target.missing,
    } satisfies RateTransmissionTargetBlock;
  });

  return {
    baseLabel: "10Y Treasury",
    targets,
    curve: [
      { id: "short_tail", label: "Short tail", description: "2Y − 3M", reading: latestDifference(points("treasury_3m"), points("treasury_2y")) },
      { id: "belly", label: "Belly", description: "10Y − 2Y", reading: latestDifference(points("treasury_2y"), treasury10y) },
      { id: "long_tail", label: "Long tail", description: "30Y − 10Y", reading: latestDifference(treasury10y, points("treasury_30y")) },
      { id: "policy_gap", label: "Policy gap", description: "2Y − effective fed funds", reading: latestDifference(points("fed_funds"), points("treasury_2y")) },
    ],
    windows: ATTRIBUTION_WINDOWS.map((window) => window.label),
    notes: [
      "The mortgage rate is a weekly survey covering quotes from earlier in its week, so it is paired with the Treasury yield as of the survey date.",
      "Level splits and change attribution are arithmetic on published yields. Pass-through and timing are estimates, shown with the sample they rest on.",
    ],
  };
}

async function fetchMetric(
  definition: RatesCreditDefinition,
  apiKey: string,
  historicalPoints: MarketRatesCreditPoint[] = [],
): Promise<MarketRatesCreditMetric> {
  // Shared FRED client rather than a third copy of the URL, key and error
  // handling. One series id plus one limit means one upstream request, so the
  // Treasury legs this route and the transmission block both need are fetched
  // exactly once.
  const observations = await fetchFredSeriesPoints(
    definition.seriesId,
    { limit: SERIES_LIMIT, revalidate: seriesRefreshSeconds(definition.cadence ?? "daily") },
    apiKey,
  );
  const merged = new Map(historicalPoints.map((point) => [point.date, point]));
  for (const point of observations) merged.set(point.date, point);
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

  // Computed while the observation history is still in hand, then dropped from
  // the response: nothing on the client reads `points`, and shipping 1,500
  // observations per series was ~350KB per visitor for no rendered pixel.
  const transmission = buildTransmission(metrics);
  const shipped = metrics.map((metric) => ({ ...metric, points: [] }));
  const inGroup = (group: MarketRatesCreditGroup) => shipped.filter((metric) => metric.group === group);
  const byTenor = (left: MarketRatesCreditMetric, right: MarketRatesCreditMetric) => (left.tenorYears ?? 0) - (right.tenorYears ?? 0);

  return {
    treasuryCurve: inGroup("treasury").sort(byTenor),
    realYields: inGroup("real_yield").sort(byTenor),
    investmentGrade: inGroup("credit_ig"),
    highYield: inGroup("credit_hy"),
    borrowing: inGroup("borrowing"),
    transmission,
    signals: buildRatesCreditSignals(metrics),
    drivers: buildDrivers(metrics),
    generatedAt: new Date().toISOString(),
    cacheSeconds: RATES_CREDIT_CACHE_SECONDS,
    source: "FRED",
    creditDataStatus: includeIceData ? "enabled" : "license_required",
    warnings,
  };
}
