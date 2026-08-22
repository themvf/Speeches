import type { MarketMacroIndicator, MarketMacroIndicatorId, MarketMacroPoint } from "@/lib/server/types";

/**
 * Reading-level and cross-indicator context for the Macro tab.
 *
 * Two jobs, both descriptive:
 *
 *  1. `percentileContext` says where the current print sits in the history we
 *     actually hold. "Baa spread 1.64" means nothing on its own; "lower than
 *     84% of readings since Mar 2025" does.
 *
 *  2. `assessConditions` combines indicators into a handful of named states,
 *     each carrying the textbook convention it rests on and the exact numbers
 *     that drove it, so every claim on screen is auditable against a value
 *     also on screen.
 *
 * Deliberate limits. These describe CONDITIONS, never forecasts, and never a
 * recommendation. Where indicators disagree the summary says they disagree
 * rather than forcing one narrative - a mixed picture is information, and
 * flattening it would be the dishonest part.
 */

export type ConditionState = "calm" | "watch" | "alert" | "neutral";

export interface ConditionDriver {
  label: string;
  value: string;
}

export interface MacroCondition {
  id: string;
  /** What this condition is about, e.g. "Yield curve". */
  label: string;
  state: ConditionState;
  /** The current state in a few words, e.g. "Positively sloped". */
  headline: string;
  /** The convention this rests on. Stated as convention, not prediction. */
  meaning: string;
  /** The readings behind it, so the claim can be checked on the same page. */
  drivers: ConditionDriver[];
}

export interface PercentileContext {
  /** Share of observations at or below the current value, 0-100. */
  percentile: number;
  /** Observations the percentile was computed over. */
  sampleSize: number;
  /** Plain-language window, e.g. "since Mar 2025". */
  window: string;
  /** Full sentence, e.g. "Higher than 84% of readings since Mar 2025". */
  summary: string;
}

function monthYear(date: string): string {
  const parsed = new Date(`${date}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return date;
  return parsed.toLocaleDateString("en-US", { month: "short", year: "numeric", timeZone: "UTC" });
}

/**
 * Where the latest print sits within the series' own history.
 *
 * The window is whatever that series carries, which varies a lot: daily series
 * hold about 17 months, monthly ones about five years, quarterly GDP just 20
 * prints. So the window is always named. Claiming "a multi-year low" off a
 * series that only goes back to last spring would be the easy mistake here.
 */
export function percentileContext(indicator: MarketMacroIndicator): PercentileContext | null {
  const points: MarketMacroPoint[] = indicator.points ?? [];
  // Below this a percentile is noise dressed as precision.
  if (points.length < 12) return null;

  const values = points.map((point) => point.value);
  const current = indicator.value;
  const atOrBelow = values.filter((value) => value <= current).length;
  const percentile = Math.round((atOrBelow / values.length) * 100);
  const window = `since ${monthYear(points[0].date)}`;

  let summary: string;
  if (percentile >= 98) summary = `Highest reading ${window}`;
  else if (percentile <= 2) summary = `Lowest reading ${window}`;
  else if (percentile >= 50) summary = `Higher than ${percentile}% of readings ${window}`;
  else summary = `Lower than ${100 - percentile}% of readings ${window}`;

  return { percentile, sampleSize: values.length, window, summary };
}

function byId(indicators: MarketMacroIndicator[]): Map<MarketMacroIndicatorId, MarketMacroIndicator> {
  return new Map(indicators.map((indicator) => [indicator.id, indicator]));
}

const pp = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(2)} pp`;
const pct = (value: number) => `${value.toFixed(2)}%`;

/**
 * Named condition reads across the indicator set. Any condition whose inputs
 * are missing is simply omitted rather than guessed at.
 */
export function assessConditions(indicators: MarketMacroIndicator[]): MacroCondition[] {
  const map = byId(indicators);
  const conditions: MacroCondition[] = [];

  // ── Yield curve ──────────────────────────────────────────────────────────
  const curve = map.get("yield_curve_10y2y");
  if (curve) {
    const inverted = curve.value < 0;
    const flat = curve.value >= 0 && curve.value < 0.25;
    conditions.push({
      id: "curve",
      label: "Yield curve",
      state: inverted ? "alert" : flat ? "watch" : "calm",
      headline: inverted ? "Inverted" : flat ? "Nearly flat" : "Positively sloped",
      meaning: inverted
        ? "Short maturities yield more than long ones. Every US recession since the 1970s was preceded by an inversion, typically by several quarters - though not every inversion was followed by one."
        : flat
          ? "The gap between long and short maturities has nearly closed, the shape that precedes an inversion."
          : "Longer maturities yield more than shorter ones, the ordinary shape of the curve.",
      drivers: [{ label: "10Y minus 2Y", value: pp(curve.value) }],
    });
  }

  // ── Credit ───────────────────────────────────────────────────────────────
  const spread = map.get("credit_spread_baa");
  const creditConditions = map.get("credit_conditions");
  if (spread || creditConditions) {
    const spreadPercentile = spread ? percentileContext(spread)?.percentile ?? null : null;
    const tight = (creditConditions?.value ?? 0) > 0;
    const stretched = spreadPercentile !== null && spreadPercentile >= 80;
    conditions.push({
      id: "credit",
      label: "Credit",
      state: stretched || tight ? (stretched && tight ? "alert" : "watch") : "calm",
      headline: stretched && tight ? "Tightening" : stretched ? "Spreads elevated" : tight ? "Conditions tight" : "Calm",
      meaning:
        "What corporates pay over Treasuries, and how freely credit is flowing. Spreads usually widen ahead of equity drawdowns, which is why credit is watched as the earlier signal.",
      drivers: [
        ...(spread ? [{ label: "Baa over 10Y", value: pp(spread.value) }] : []),
        ...(creditConditions ? [{ label: "Chicago Fed credit subindex", value: creditConditions.value.toFixed(2) }] : []),
      ],
    });
  }

  // ── Labor ────────────────────────────────────────────────────────────────
  const sahm = map.get("sahm_rule");
  const payrolls = map.get("nonfarm_payrolls");
  if (sahm || payrolls) {
    const triggered = (sahm?.value ?? 0) >= 0.5;
    const shedding = (payrolls?.value ?? 0) < 0;
    conditions.push({
      id: "labor",
      label: "Labor market",
      state: triggered ? "alert" : shedding ? "watch" : "calm",
      headline: triggered ? "Sahm rule triggered" : shedding ? "Payrolls contracting" : "Steady",
      meaning: triggered
        ? "The Sahm rule reads 0.50 or higher. It is a real-time marker that has historically coincided with the start of a recession rather than predicting one."
        : shedding
          ? "Payrolls fell over the month. A single negative print is noisy and often revised; the Sahm rule below is the smoothed version of the same question."
          : "Payrolls are growing and the Sahm recession marker is untriggered.",
      drivers: [
        ...(payrolls ? [{ label: "Monthly payroll change", value: `${payrolls.value >= 0 ? "+" : ""}${Math.round(payrolls.value)}K` }] : []),
        ...(sahm ? [{ label: "Sahm rule (triggers at 0.50)", value: sahm.value.toFixed(2) }] : []),
      ],
    });
  }

  // ── Policy stance ────────────────────────────────────────────────────────
  const realYield = map.get("real_yield_10y");
  const funds = map.get("effective_fed_funds");
  if (realYield) {
    const restrictive = realYield.value >= 1.5;
    conditions.push({
      id: "policy",
      label: "Policy stance",
      state: restrictive ? "watch" : "neutral",
      headline: restrictive ? "Restrictive" : realYield.value < 0 ? "Negative real rates" : "Near neutral",
      meaning:
        "The inflation-protected 10-year yield is the rate policy actually transmits through. Estimates of the neutral real rate cluster near 0.5-1%; readings well above that are conventionally described as restrictive.",
      drivers: [
        { label: "10Y real yield", value: pct(realYield.value) },
        ...(funds ? [{ label: "Effective fed funds", value: pct(funds.value) }] : []),
      ],
    });
  }

  // ── Inflation vs target ──────────────────────────────────────────────────
  const corePce = map.get("core_pce_inflation");
  if (corePce) {
    const above = corePce.value > 2.5;
    conditions.push({
      id: "inflation",
      label: "Inflation",
      state: above ? "watch" : "calm",
      headline: above ? "Above target" : "Near target",
      meaning:
        "The Federal Reserve targets 2% inflation measured on PCE, and core PCE is the series it weights most heavily in judging progress.",
      drivers: [{ label: "Core PCE, year over year", value: pct(corePce.value) }],
    });
  }

  return conditions;
}

/** "a", "a and b", "a, b and c" - so the summary reads as a sentence. */
function joinList(items: string[]): string {
  if (items.length <= 1) return items[0] ?? "";
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

/**
 * One honest sentence over the set. When conditions disagree it says so -
 * a mixed picture is the finding, not something to resolve into a story.
 */
export function summarizeConditions(conditions: MacroCondition[]): string {
  if (!conditions.length) return "";
  const alerts = conditions.filter((condition) => condition.state === "alert");
  const watches = conditions.filter((condition) => condition.state === "watch");
  const calm = conditions.filter((condition) => condition.state === "calm");

  const name = (list: MacroCondition[]) => joinList(list.map((condition) => condition.label.toLowerCase()));

  if (alerts.length) {
    const rest = watches.length + calm.length;
    return rest > 0
      ? `Signals disagree: ${name(alerts)} at historically notable levels, while ${rest} other condition${rest === 1 ? "" : "s"} read${rest === 1 ? "s" : ""} normally.`
      : `${name(alerts)} at historically notable levels.`;
  }
  if (watches.length && calm.length) {
    return `Signals are mixed: ${name(watches)} worth watching; ${name(calm)} unremarkable.`;
  }
  if (watches.length) return `Worth watching: ${name(watches)}.`;
  return "No tracked condition is at a notable level.";
}
