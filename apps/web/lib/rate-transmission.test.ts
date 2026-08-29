import assert from "node:assert/strict";
import test from "node:test";
import {
  alignAsOf,
  attributeWindow,
  attributeYearToDate,
  buildCreditResearch,
  crossCorrelate,
  decompose,
  RATE_TRANSMISSION_SERIES,
  rollingOls,
  type AlignedRatePoint,
} from "./rate-transmission.ts";

test("alignAsOf uses the last base observation on or before each target date", () => {
  const base = [
    { date: "2025-01-02", value: 4.1 },
    { date: "2025-01-03", value: 4.2 },
    { date: "2025-01-06", value: 4.4 },
  ];
  const target = [
    { date: "2025-01-01", value: 6.5 },
    { date: "2025-01-04", value: 6.6 },
    { date: "2025-01-09", value: 6.8 },
  ];
  assert.deepEqual(alignAsOf(base, target), [
    { date: "2025-01-04", target: 6.6, base: 4.2, baseDate: "2025-01-03", spread: 2.3999999999999995 },
    { date: "2025-01-09", target: 6.8, base: 4.4, baseDate: "2025-01-06", spread: 2.3999999999999995 },
  ]);
});

test("decompose and attribution preserve the accounting identity", () => {
  const points = alignAsOf(
    [{ date: "2024-01-01", value: 4 }, { date: "2025-01-01", value: 4.5 }],
    [{ date: "2024-01-01", value: 6 }, { date: "2025-01-01", value: 7 }],
  );
  const level = decompose(points);
  const attribution = attributeWindow(points, 365);
  assert.ok(level);
  assert.equal(level.rate, level.base + level.spread);
  assert.ok(attribution);
  assert.ok(Math.abs(attribution.totalBp - attribution.baseBp - attribution.spreadBp) < 1e-9);
});

test("every fetched series has an explicit license disposition", () => {
  assert.ok(RATE_TRANSMISSION_SERIES.length > 0);
  for (const series of RATE_TRANSMISSION_SERIES) assert.ok(series.license);
});

test("YTD attribution starts from the final aligned observation of the prior year", () => {
  const points = alignAsOf(
    [{ date: "2025-12-30", value: 4 }, { date: "2026-01-02", value: 4.1 }, { date: "2026-03-01", value: 4.3 }],
    [{ date: "2025-12-31", value: 6 }, { date: "2026-01-03", value: 6.2 }, { date: "2026-03-01", value: 6.5 }],
  );
  const result = attributeYearToDate(points);
  assert.ok(result);
  assert.equal(result.startDate, "2025-12-31");
  assert.ok(Math.abs(result.totalBp - result.baseBp - result.spreadBp) < 1e-9);
});

function alignedFromChanges(xChanges: number[], yChanges: number[]): AlignedRatePoint[] {
  let base = 4;
  let target = 6;
  const points: AlignedRatePoint[] = [{ date: "2025-01-01", base, target, baseDate: "2025-01-01", spread: target - base }];
  for (let index = 0; index < xChanges.length; index += 1) {
    base += xChanges[index];
    target += yChanges[index];
    const date = new Date(Date.UTC(2025, 0, 8 + index * 7)).toISOString().slice(0, 10);
    points.push({ date, base, target, baseDate: date, spread: target - base });
  }
  return points;
}

test("rollingOls reports beta, uncertainty, fit, and suppresses small samples", () => {
  const x = Array.from({ length: 40 }, (_, index) => ((index % 5) - 2) / 10);
  const y = x.map((value) => 0.05 + 1.5 * value);
  const result = rollingOls(alignedFromChanges(x, y), 52, 30);
  assert.ok(result);
  assert.ok(Math.abs(result.beta - 1.5) < 1e-10);
  assert.ok(result.standardError < 1e-10);
  assert.ok(Math.abs(result.rSquared - 1) < 1e-10);
  assert.equal(result.n, 40);
  assert.equal(rollingOls(alignedFromChanges(x.slice(0, 20), y.slice(0, 20)), 52, 30), null);
});

test("crossCorrelate recovers an injected Treasury lead and avoids naming contemporaneous correlation a lead", () => {
  let seed = 17;
  const x = Array.from({ length: 70 }, () => {
    seed = (seed * 48271) % 2147483647;
    return seed / 2147483647 - 0.5;
  });
  const delayedY = x.map((_, index) => index >= 2 ? x[index - 2] : 0);
  const delayed = crossCorrelate(alignedFromChanges(x, delayedY), 4, 30);
  assert.ok(delayed);
  assert.equal(delayed.bestLag, 2);
  assert.equal(delayed.verdict, "treasury_leads");

  const contemporaneous = crossCorrelate(alignedFromChanges(x, x), 4, 30);
  assert.ok(contemporaneous);
  assert.equal(contemporaneous.bestLag, 0);
  assert.equal(contemporaneous.verdict, "no_clear_lead");
});

test("buildCreditResearch separates default risk from sentiment and classifies the cross-asset regime", () => {
  const result = buildCreditResearch([
    { date: "2025-10-01", corporateSpread: 1.1, excessBondPremium: -0.1, defaultRiskComponent: 1.2, recessionProbability: 0.1 },
    { date: "2026-01-01", corporateSpread: 1.5, excessBondPremium: 0.2, defaultRiskComponent: 1.3, recessionProbability: 0.2 },
  ], [
    { date: "2025-10-01", value: 4 },
    { date: "2026-01-01", value: 4.3 },
  ]);
  assert.ok(result);
  assert.equal(result.regime, "restrictive_financing");
  assert.ok(Math.abs((result.ebpChange3mBp ?? 0) - 30) < 1e-10);
  assert.ok(Math.abs((result.treasuryChange3mBp ?? 0) - 30) < 1e-10);
  assert.equal(result.defaultRiskComponent, 1.3);
});
