import assert from "node:assert/strict";
import test from "node:test";

import { SIGNALS, signalFor } from "./macro-signals.ts";
import { FRED_MACRO_DEFINITIONS } from "./server/fred-macro.ts";
import type { MarketMacroIndicator, MarketMacroIndicatorId } from "./server/types.ts";

function indicator(overrides: Partial<MarketMacroIndicator> = {}): MarketMacroIndicator {
  return {
    id: "cpi_inflation",
    fredSeriesId: "CPIAUCSL",
    label: "CPI Inflation",
    description: "",
    frequency: "Monthly",
    unit: "percent",
    group: "headline",
    priority: 1,
    value: 2.4,
    previousValue: 2.2,
    change: 0.2,
    observationDate: "2026-08-01",
    lastUpdated: "",
    points: [],
    sourceUrl: "",
    ...overrides,
  };
}

test("every shipped indicator has a signal", () => {
  for (const definition of FRED_MACRO_DEFINITIONS) {
    assert.equal(
      typeof SIGNALS[definition.id],
      "function",
      `${definition.seriesId} (${definition.id}) has no signal`,
    );
  }
});

test("returns the indicator's own signal when the bundle knows the id", () => {
  assert.equal(signalFor(indicator({ id: "cpi_inflation", change: 0.2 })).text, "Inflation heating");
  assert.equal(signalFor(indicator({ id: "cpi_inflation", change: -0.2 })).text, "Inflation cooling");
});

test("an alerting signal still flags", () => {
  assert.equal(signalFor(indicator({ id: "sahm_rule", value: 0.6 })).alert, true);
});

/**
 * The regression this module exists for. A browser holding an older page
 * bundle across a deploy fetches fresh API data into old code; when the deploy
 * added indicators, the old bundle looked one up, got undefined, and called it
 * - taking out the whole Macro tab. An unknown id must degrade to a neutral
 * badge instead.
 */
test("an id this bundle has never seen degrades instead of throwing", () => {
  const unknown = indicator({ id: "some_future_indicator" as MarketMacroIndicatorId });

  assert.doesNotThrow(() => signalFor(unknown));
  assert.equal(signalFor({ ...unknown, change: 0.4 }).text, "Rising");
  assert.equal(signalFor({ ...unknown, change: -0.4 }).text, "Falling");
  assert.equal(signalFor({ ...unknown, change: 0 }).text, "Updated");
  assert.equal(signalFor({ ...unknown, change: null }).text, "Updated");
  assert.equal(signalFor(unknown).alert, undefined);
});

test("the three credit and rate indicators added alongside the curve resolve", () => {
  assert.equal(signalFor(indicator({ id: "credit_spread_baa", change: 0.1 })).text, "Credit spreads widening");
  assert.equal(signalFor(indicator({ id: "credit_conditions", value: 0.3 })).text, "Credit tighter than average");
  assert.equal(signalFor(indicator({ id: "real_yield_10y", value: -0.2 })).text, "Real yield negative");
});
