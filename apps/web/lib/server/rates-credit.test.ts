import assert from "node:assert/strict";
import test from "node:test";

import type { MarketRatesCreditMetric } from "./types.ts";
import {
  buildRatesCreditMetric,
  buildRatesCreditSignals,
  classifyCurve,
  fetchRatesCreditData,
  RATES_CREDIT_DEFINITIONS,
} from "./rates-credit.ts";
import { snapshotRowsFromData } from "./rates-credit-store.ts";

function metric(id: string, value: number, change1m: number, percentile: number | null = 50): MarketRatesCreditMetric {
  return {
    id,
    fredSeriesId: id,
    label: id,
    shortLabel: id,
    group: id.startsWith("hy") ? "credit_hy" : id.startsWith("ig") ? "credit_ig" : "treasury",
    tenorYears: null,
    value,
    change1d: 0,
    change1w: 0,
    change1m,
    percentile,
    zScore: 0,
    observationDate: "2026-08-26",
    points: [],
    sourceUrl: "https://example.com",
  };
}

test("defines the Phase 1 Treasury, real-yield, and credit catalog", () => {
  assert.equal(RATES_CREDIT_DEFINITIONS.filter((definition) => definition.group === "treasury").length, 10);
  assert.equal(RATES_CREDIT_DEFINITIONS.filter((definition) => definition.group === "real_yield").length, 3);
  assert.deepEqual(
    RATES_CREDIT_DEFINITIONS.filter((definition) => definition.group === "credit_ig").map((definition) => definition.shortLabel),
    ["IG", "AAA", "AA", "A", "BBB"],
  );
  assert.deepEqual(
    RATES_CREDIT_DEFINITIONS.filter((definition) => definition.group === "credit_hy").map((definition) => definition.shortLabel),
    ["HY", "BB", "B", "CCC"],
  );
  assert.equal(new Set(RATES_CREDIT_DEFINITIONS.map((definition) => definition.seriesId)).size, RATES_CREDIT_DEFINITIONS.length);
});

test("builds changes and historical statistics without treating missing history as zero", () => {
  const definition = RATES_CREDIT_DEFINITIONS.find((item) => item.id === "treasury_10y");
  assert.ok(definition);
  const points = Array.from({ length: 30 }, (_, index) => ({
    date: `2026-07-${String(index + 1).padStart(2, "0")}`,
    value: 4 + index * 0.01,
  }));
  const result = buildRatesCreditMetric(definition, points);
  assert.ok(Math.abs((result.change1d ?? 0) - 0.01) < 1e-12);
  assert.ok(Math.abs((result.change1w ?? 0) - 0.05) < 1e-12);
  assert.ok(Math.abs((result.change1m ?? 0) - 0.21) < 1e-12);
  assert.equal(result.percentile, 100);
  assert.ok((result.zScore ?? 0) > 1);
});

test("classifies bear steepening from the relative 2Y and 10Y move", () => {
  const result = classifyCurve([
    metric("treasury_2y", 4.1, 0.1),
    metric("treasury_10y", 4.5, 0.25),
  ]);
  assert.equal(result.state, "Bear steepener");
  assert.match(result.summary, /2s10s is \+40 bp/);
});

test("flags broad credit deterioration and rolls it into the composite", () => {
  const signals = buildRatesCreditSignals([
    metric("treasury_2y", 4.1, 0.15),
    metric("treasury_10y", 4.4, 0.18),
    metric("ig_broad", 1.15, 0.1),
    metric("hy_broad", 4.2, 0.3, 75),
  ]);
  assert.equal(signals.find((signal) => signal.id === "rates")?.state, "Yields rising");
  assert.equal(signals.find((signal) => signal.id === "credit")?.state, "Deteriorating");
  assert.equal(signals.find((signal) => signal.id === "composite")?.state, "Tightening");
});

test("keeps ICE credit observations behind the explicit entitlement gate", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({ observations: Array.from({ length: 30 }, (_, index) => ({
    date: `2026-07-${String(index + 1).padStart(2, "0")}`,
    value: String(4 + index * 0.01),
  })) }), { status: 200, headers: { "content-type": "application/json" } });
  try {
    const result = await fetchRatesCreditData("test-key", false, {
      DGS10: [{ date: "2020-01-02", value: 1.88 }],
    });
    assert.equal(result.creditDataStatus, "license_required");
    assert.equal(result.investmentGrade.length, 0);
    assert.equal(result.highYield.length, 0);
    assert.equal(result.treasuryCurve.find((metric) => metric.fredSeriesId === "DGS10")?.points[0].date, "2020-01-02");
    assert.equal(snapshotRowsFromData(result).length, 13);
    assert.match(result.warnings.join(" "), /disabled pending authorized data rights/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
