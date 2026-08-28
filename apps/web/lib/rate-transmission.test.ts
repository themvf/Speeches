import assert from "node:assert/strict";
import test from "node:test";
import { alignAsOf, attributeWindow, decompose, RATE_TRANSMISSION_SERIES } from "./rate-transmission.ts";

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
