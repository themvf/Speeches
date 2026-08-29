import assert from "node:assert/strict";
import test from "node:test";
import { alignAsOf, attributeWindow, decompose, MAX_BASE_STALENESS_DAYS, RATE_TRANSMISSION_SERIES } from "./rate-transmission.ts";

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

/** Business days from a start date, so fixtures carry real weekend holes. */
function businessDays(start: string, count: number): string[] {
  const out: string[] = [];
  const cursor = new Date(`${start}T00:00:00Z`);
  while (out.length < count) {
    const day = cursor.getUTCDay();
    if (day !== 0 && day !== 6) out.push(cursor.toISOString().slice(0, 10));
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }
  return out;
}

test("a base observation beyond the staleness window is dropped, not carried forward", () => {
  const base = [{ date: "2026-01-05", value: 4.2 }];
  const target = [
    { date: "2026-01-08", value: 6.4 },
    { date: "2026-06-04", value: 6.9 },
  ];
  const aligned = alignAsOf(base, target);
  assert.deepEqual(aligned.map((point) => point.date), ["2026-01-08"],
    "the June point has no base within a week and must not pair with January's yield");

  // The boundary itself is inclusive.
  const edge = new Date(`2026-01-05T00:00:00Z`);
  edge.setUTCDate(edge.getUTCDate() + MAX_BASE_STALENESS_DAYS);
  assert.equal(alignAsOf(base, [{ date: edge.toISOString().slice(0, 10), value: 6.4 }]).length, 1);
});

/**
 * The bug the as-of join exists to prevent, on a fixture built so that walking
 * the two arrays in step looks entirely reasonable: naive pairing lines three
 * weekly points up against the first three daily ones, all from one week, and
 * reports a spread that widens when the truth is flat.
 */
test("index-zipping the same series disagrees with the as-of join", () => {
  const daily = [
    { date: "2026-01-05", value: 4.0 }, { date: "2026-01-06", value: 4.0 },
    { date: "2026-01-07", value: 4.0 }, { date: "2026-01-08", value: 4.0 },
    { date: "2026-01-12", value: 4.5 }, { date: "2026-01-15", value: 4.5 },
    { date: "2026-01-20", value: 5.0 }, { date: "2026-01-22", value: 5.0 },
  ];
  const weekly = [
    { date: "2026-01-08", value: 6.0 },
    { date: "2026-01-15", value: 6.5 },
    { date: "2026-01-22", value: 7.0 },
  ];
  const aligned = alignAsOf(daily, weekly);
  assert.deepEqual(aligned.map((point) => Number(point.spread.toFixed(2))), [2, 2, 2]);

  const naive = weekly.map((point, index) => Number((point.value - daily[index].value).toFixed(2)));
  assert.deepEqual(naive, [2, 2.5, 3], "index-zipping invents a trend");
});

test("attribution legs still sum to the total after rounding to whole basis points", () => {
  // Chosen so independent rounding of each leg would break the identity:
  // 12.5 -> 13, 8.4 -> 8, 4.1 -> 4, and 8 + 4 is not 13.
  const points = alignAsOf(
    [{ date: "2025-01-01", value: 4.0 }, { date: "2026-01-01", value: 4.084 }],
    [{ date: "2025-01-01", value: 6.0 }, { date: "2026-01-01", value: 6.125 }],
  );
  const attribution = attributeWindow(points, 365);
  assert.ok(attribution);
  assert.equal(attribution.totalBp, attribution.baseBp + attribution.spreadBp);
  assert.ok(Number.isInteger(attribution.baseBp) && Number.isInteger(attribution.spreadBp));
});

test("the spread percentile names the window it was computed over", () => {
  const dates = businessDays("2026-01-01", 40);
  const base = dates.map((date) => ({ date, value: 4.0 }));
  const target = dates.map((date, index) => ({ date, value: 5.0 + index * 0.01 }));
  const level = decompose(alignAsOf(base, target));
  assert.ok(level?.spreadContext);
  assert.match(level.spreadContext.summary, /since /,
    "a percentile without its window is how a 40-observation sample becomes 'a record low'");
  assert.equal(level.spreadContext.percentile, 100);
});

test("the Baa spread is sourced from FRED's published series, not recomputed", () => {
  const baa = RATE_TRANSMISSION_SERIES.find((series) => series.id === "BAA10Y");
  assert.ok(baa, "the corporate half of the panel needs BAA10Y");
  assert.equal(baa.license, "citation-required",
    "BAA10Y is citation required - the pre-approval tag that blocks a public route is ICE BofA's, not Moody's");
});
