import assert from "node:assert/strict";
import test from "node:test";
import {
  alignAsOf,
  type AlignedRatePoint,
  attributeWindow,
  decompose,
  leadLag,
  levelFromSpread,
  MAX_BASE_STALENESS_DAYS,
  MIN_PASS_THROUGH_OBSERVATIONS,
  passThrough,
  RATE_TRANSMISSION_SERIES,
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

// ── Pass-through (stage 4) ───────────────────────────────────────────────────

/** Aligned points whose target moves by `beta` for every unit the base moves. */
function alignedWithBeta(count: number, beta: number, noise = 0): AlignedRatePoint[] {
  const dates = businessDays("2025-01-01", count);
  let base = 4;
  let target = 6;
  return dates.map((date, index) => {
    const step = ((index % 7) - 3) / 100;
    base += step;
    target += beta * step + (noise ? ((index % 5) - 2) * noise : 0);
    return { date, target, base, baseDate: date, spread: target - base };
  });
}

test("pass-through recovers the share of a Treasury move that reaches the borrower", () => {
  const result = passThrough(alignedWithBeta(120, 0.8), 100, "daily changes");
  assert.ok(result);
  assert.ok(Math.abs(result.beta - 0.8) < 1e-9, `expected 0.8, got ${result.beta}`);
  assert.ok(result.rSquared > 0.999, "a clean relationship should explain nearly all variation");
  assert.equal(result.observations, 100);
  assert.match(result.windowLabel, /100 daily changes/);
});

test("pass-through reports a wider error band once the relationship is noisy", () => {
  const clean = passThrough(alignedWithBeta(200, 0.8), 150, "daily changes");
  const noisy = passThrough(alignedWithBeta(200, 0.8, 0.02), 150, "daily changes");
  assert.ok(clean && noisy);
  assert.ok(noisy.stdError > clean.stdError, "noise must widen the band, not hide in the estimate");
  assert.ok(noisy.rSquared < clean.rSquared);
});

test("pass-through shows nothing rather than something thin", () => {
  assert.equal(passThrough(alignedWithBeta(20, 0.8), 52, "weekly changes"), null,
    `below ${MIN_PASS_THROUGH_OBSERVATIONS} observations there is no honest number`);
});

test("pass-through declines when the Treasury barely moved", () => {
  const dates = businessDays("2025-01-01", 120);
  const flat = dates.map((date, index) => ({ date, target: 6 + index * 0.01, base: 4, baseDate: date, spread: 2 }));
  assert.equal(passThrough(flat, 100, "daily changes"), null, "no variation to explain means no slope");
});

// ── Lead/lag (stage 5) ───────────────────────────────────────────────────────

test("lead/lag recovers an injected lead", () => {
  const dates = businessDays("2025-01-01", 200);
  const leader = dates.map((date, index) => ({ date, value: 4 + Math.sin(index * 0.6) }));
  // The follower repeats the leader's path three observations later.
  const follower = dates.map((date, index) => ({ date, value: 6 + Math.sin((index - 3) * 0.6) }));
  const result = leadLag(leader, follower, 8, "days", "Leader", "Follower");
  assert.ok(result);
  assert.equal(result.bestLagPeriods, 3);
  assert.match(result.verdict, /Leader has moved 3 days ahead of Follower/);

  // Singular when the lead is one period - "1 weeks" is a tell that nobody read the output.
  const oneAhead = dates.map((date, index) => ({ date, value: 6 + Math.sin((index - 1) * 0.6) }));
  const single = leadLag(leader, oneAhead, 8, "weeks", "Leader", "Follower");
  assert.ok(single);
  assert.match(single.verdict, /1 week ahead/);
  assert.doesNotMatch(single.verdict, /1 weeks/);
});

test("lead/lag refuses to name a leader when nothing stands out", () => {
  const dates = businessDays("2025-01-01", 200);
  let a = 4;
  let b = 6;
  const first = dates.map((date, index) => { a += (((index * 37) % 17) - 8) / 1000; return { date, value: a }; });
  const second = dates.map((date, index) => { b += (((index * 91) % 23) - 11) / 1000; return { date, value: b }; });
  const result = leadLag(first, second, 8, "days", "First", "Second");
  assert.ok(result);
  assert.equal(result.bestLagPeriods, 0);
  assert.match(result.verdict, /No clear timing relationship|move together/);
});

/**
 * The trap, made executable.
 *
 * A yield level rebuilt as `published spread + base` contains the base, so it
 * correlates with it almost perfectly at lag zero no matter what the spread
 * does. Any "finding" from that comparison is arithmetic. This is why the route
 * gates lead/lag on a series being observed independently of its base.
 */
test("a level rebuilt from a spread is circular against its own base", () => {
  const dates = businessDays("2025-01-01", 200);
  const base = dates.map((date, index) => ({ date, value: 4 + Math.sin(index * 0.5) }));
  const spread = dates.map((date, index) => ({ date, value: 1.6 + Math.cos(index * 0.11) / 40 }));
  const rebuilt = levelFromSpread(spread, base);

  const circular = leadLag(base, rebuilt, 8, "days", "Base", "Rebuilt level");
  assert.ok(circular);
  assert.ok(Math.abs(circular.correlation) > 0.9,
    "the rebuilt level moves with its own base by construction, not by economics");
  assert.equal(circular.bestLagPeriods, 0);
});

/**
 * The survey-timing correction, which changed the answer by more than double on
 * real data: 25% (R-squared 0.08) same-week against 56% (R-squared 0.44) with
 * the Treasury change lagged one week. Without it the panel would report that
 * Treasury moves barely reach mortgage borrowers, which is an artefact of when
 * Freddie Mac takes its survey, not a fact about lenders.
 */
test("pass-through can lag the Treasury change for a rate measured later", () => {
  const dates = businessDays("2025-01-01", 160);
  let base = 4;
  const bases: number[] = [];
  for (let i = 0; i < dates.length; i += 1) {
    base += (((i * 29) % 13) - 6) / 100;
    bases.push(base);
  }
  // The target responds to LAST period's base move, at 70%.
  const aligned: AlignedRatePoint[] = dates.map((date, index) => {
    const previousMove = index >= 2 ? bases[index - 1] - bases[index - 2] : 0;
    const target = 6 + 0.7 * (bases[Math.max(index - 1, 0)] - bases[0]) + previousMove * 0;
    return { date, target, base: bases[index], baseDate: date, spread: target - bases[index] };
  });

  const sameWeek = passThrough(aligned, 100, "weekly changes");
  const lagged = passThrough(aligned, 100, "weekly changes", 1, "Treasury change taken one week earlier.");
  assert.ok(sameWeek && lagged);
  assert.ok(lagged.rSquared > sameWeek.rSquared,
    `lagging should fit better here: ${lagged.rSquared.toFixed(2)} vs ${sameWeek.rSquared.toFixed(2)}`);
  assert.ok(Math.abs(lagged.beta - 0.7) < 0.01, `expected ~0.7, got ${lagged.beta}`);
  assert.equal(lagged.lagPeriods, 1);
  assert.match(lagged.lagNote ?? "", /one week earlier/);
  assert.equal(sameWeek.lagPeriods, 0);
  assert.equal(sameWeek.lagNote, null, "an unlagged estimate must not carry a lag caveat");
});
