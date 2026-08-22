import assert from "node:assert/strict";
import test from "node:test";

import {
  assessConditions,
  percentileContext,
  summarizeConditions,
  type MacroCondition,
} from "./macro-context.ts";
import type { MarketMacroIndicator, MarketMacroIndicatorId, MarketMacroPoint } from "./server/types.ts";

function series(values: number[], startYear = 2025): MarketMacroPoint[] {
  return values.map((value, index) => ({
    date: `${startYear + Math.floor(index / 12)}-${String((index % 12) + 1).padStart(2, "0")}-01`,
    value,
  }));
}

function indicator(
  id: MarketMacroIndicatorId,
  value: number,
  points: MarketMacroPoint[] = [],
): MarketMacroIndicator {
  return {
    id, value, points,
    fredSeriesId: "X", label: id, description: "", frequency: "Monthly",
    unit: "percent", group: "financial", priority: 1,
    previousValue: null, change: null, observationDate: "2026-08-20",
    lastUpdated: "", sourceUrl: "",
  };
}

const state = (conditions: MacroCondition[], id: string) => conditions.find((c) => c.id === id)?.state;
const headline = (conditions: MacroCondition[], id: string) => conditions.find((c) => c.id === id)?.headline;

// ── percentile ───────────────────────────────────────────────────────────────

test("places a reading within its own history and names the window", () => {
  const context = percentileContext(indicator("credit_spread_baa", 5, series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])));
  assert.equal(context?.percentile, 42);
  assert.equal(context?.sampleSize, 12);
  assert.equal(context?.window, "since Jan 2025");
  assert.equal(context?.summary, "Lower than 58% of readings since Jan 2025");
});

test("calls out the extremes plainly", () => {
  const high = percentileContext(indicator("credit_spread_baa", 99, series(Array.from({ length: 20 }, (_, i) => i))));
  assert.equal(high?.summary, "Highest reading since Jan 2025");
  const low = percentileContext(indicator("credit_spread_baa", -99, series(Array.from({ length: 20 }, (_, i) => i))));
  assert.equal(low?.summary, "Lowest reading since Jan 2025");
});

test("refuses to compute a percentile from too little history", () => {
  // Quarterly GDP carries 20 prints; anything under a dozen is noise dressed
  // as precision, and a claim about it would overstate what we hold.
  assert.equal(percentileContext(indicator("real_gdp_growth", 2, series([1, 2, 3]))), null);
  assert.equal(percentileContext(indicator("real_gdp_growth", 2, [])), null);
});

// ── conditions ───────────────────────────────────────────────────────────────

test("reads the curve by its sign, not its level", () => {
  assert.equal(state([...assessConditions([indicator("yield_curve_10y2y", -0.4)])], "curve"), "alert");
  assert.equal(headline(assessConditions([indicator("yield_curve_10y2y", -0.4)]), "curve"), "Inverted");
  assert.equal(state(assessConditions([indicator("yield_curve_10y2y", 0.1)]), "curve"), "watch");
  assert.equal(state(assessConditions([indicator("yield_curve_10y2y", 0.5)]), "curve"), "calm");
});

test("the Sahm rule outranks a single negative payroll print", () => {
  const both = assessConditions([indicator("sahm_rule", 0.6), indicator("nonfarm_payrolls", -23)]);
  assert.equal(state(both, "labor"), "alert");
  assert.equal(headline(both, "labor"), "Sahm rule triggered");

  const payrollsOnly = assessConditions([indicator("sahm_rule", -0.03), indicator("nonfarm_payrolls", -23)]);
  assert.equal(state(payrollsOnly, "labor"), "watch");
  assert.equal(headline(payrollsOnly, "labor"), "Payrolls contracting");
});

test("every condition carries the readings that produced it", () => {
  const conditions = assessConditions([
    indicator("sahm_rule", -0.03),
    indicator("nonfarm_payrolls", -23),
  ]);
  const labor = conditions.find((c) => c.id === "labor")!;
  assert.deepEqual(labor.drivers.map((d) => d.value), ["-23K", "-0.03"]);
  assert.match(labor.meaning, /Sahm rule/);
});

test("a condition whose inputs are missing is omitted, not guessed", () => {
  const conditions = assessConditions([indicator("yield_curve_10y2y", 0.5)]);
  assert.deepEqual(conditions.map((c) => c.id), ["curve"]);
});

test("credit needs both a stretched spread and tight conditions to alert", () => {
  const history = series(Array.from({ length: 40 }, (_, i) => i / 20));
  const stretched = indicator("credit_spread_baa", 99, history);
  assert.equal(state(assessConditions([stretched]), "credit"), "watch");
  assert.equal(state(assessConditions([stretched, indicator("credit_conditions", 0.4)]), "credit"), "alert");
  assert.equal(state(assessConditions([indicator("credit_spread_baa", 0.1, history)]), "credit"), "calm");
});

// ── summary honesty ──────────────────────────────────────────────────────────

test("says signals disagree rather than picking a story", () => {
  // Today's real shape: payrolls negative, curve normal, credit calm.
  const summary = summarizeConditions(assessConditions([
    indicator("yield_curve_10y2y", 0.5),
    indicator("nonfarm_payrolls", -23),
    indicator("sahm_rule", -0.03),
    indicator("core_pce_inflation", 3.29),
  ]));
  assert.match(summary, /mixed/i);
  assert.match(summary, /labor market/);
  assert.match(summary, /yield curve/);
});

test("an alert alongside normal reads is reported as a disagreement", () => {
  const summary = summarizeConditions(assessConditions([
    indicator("yield_curve_10y2y", -0.4),
    indicator("core_pce_inflation", 2.0),
  ]));
  assert.match(summary, /disagree/i);
});

test("an all-quiet board says so without inventing a concern", () => {
  const summary = summarizeConditions(assessConditions([
    indicator("yield_curve_10y2y", 0.9),
    indicator("core_pce_inflation", 2.0),
  ]));
  assert.equal(summary, "No tracked condition is at a notable level.");
});

test("no condition text makes a forecast or a recommendation", () => {
  const conditions = assessConditions([
    indicator("yield_curve_10y2y", -0.4),
    indicator("sahm_rule", 0.6),
    indicator("nonfarm_payrolls", -23),
    indicator("real_yield_10y", 2.35),
    indicator("core_pce_inflation", 3.29),
    indicator("credit_spread_baa", 1.64, series(Array.from({ length: 30 }, (_, i) => 1 + i / 30))),
  ]);
  const forbidden = /\b(will|should|buy|sell|expect|forecast|predict|recommend)\b/i;
  for (const condition of conditions) {
    assert.equal(forbidden.test(condition.meaning), false, `"${condition.meaning}" reads as advice`);
    assert.equal(forbidden.test(condition.headline), false, `"${condition.headline}" reads as advice`);
  }
});
