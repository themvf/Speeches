import assert from "node:assert/strict";
import test from "node:test";

import {
  buildBreadthPair,
  EVEN_THRESHOLD_PP,
  summarizeBreadth,
  type BreadthPair,
} from "./market-breadth.ts";

function pair(capPct: number, equalPct: number, id = "sp500"): BreadthPair {
  const built = buildBreadthPair({
    id,
    label: "S&P 500",
    capSymbol: "SPY",
    capPct,
    equalSymbol: "RSP",
    equalPct,
  });
  assert.ok(built, "expected a pair");
  return built;
}

test("a narrow tape is the index beating the average stock", () => {
  const narrow = pair(1.2, 0.3);
  assert.equal(narrow.tone, "narrow");
  assert.equal(Number(narrow.spreadPp.toFixed(2)), -0.9);
});

test("a broad tape is the average stock beating the index", () => {
  const broad = pair(0.3, 1.2);
  assert.equal(broad.tone, "broad");
  assert.equal(Number(broad.spreadPp.toFixed(2)), 0.9);
});

test("small differences read as even rather than as signal", () => {
  assert.equal(pair(1.0, 1.0).tone, "even");
  assert.equal(pair(1.0, 1.0 + EVEN_THRESHOLD_PP / 2).tone, "even");
  assert.equal(pair(1.0, 1.0 - EVEN_THRESHOLD_PP / 2).tone, "even");
  // Just past the threshold it becomes a reading.
  assert.equal(pair(1.0, 1.0 + EVEN_THRESHOLD_PP + 0.01).tone, "broad");
});

test("calls out a sign disagreement, the case a cap-weighted number hides", () => {
  // 2026-08-22 live: SPY -1.19%, RSP +0.52%. The index fell while the average
  // S&P constituent rose - the whole reason this measure exists.
  const real = pair(-1.19, 0.52);
  assert.equal(real.tone, "broad");
  assert.equal(Number(real.spreadPp.toFixed(2)), 1.71);
  assert.match(real.reading, /Index fell while the average stock rose/);
});

test("calls out the inverse disagreement too", () => {
  const carried = pair(0.8, -0.3);
  assert.equal(carried.tone, "narrow");
  assert.match(carried.reading, /Index rose while the average stock fell/);
});

test("a pair with a missing or unusable leg is dropped, not half-built", () => {
  const base = { id: "sp500", label: "S&P 500", capSymbol: "SPY", equalSymbol: "RSP" };
  assert.equal(buildBreadthPair({ ...base, capPct: 1, equalPct: undefined }), null);
  assert.equal(buildBreadthPair({ ...base, capPct: null, equalPct: 1 }), null);
  assert.equal(buildBreadthPair({ ...base, capPct: 1, equalPct: Number.NaN }), null);
  assert.equal(buildBreadthPair({ ...base, capPct: Number.POSITIVE_INFINITY, equalPct: 1 }), null);
  // Zero is a real reading, not a missing one.
  assert.ok(buildBreadthPair({ ...base, capPct: 0, equalPct: 0 }));
});

test("summarizes agreement across both indices", () => {
  assert.match(summarizeBreadth([pair(1.2, 0.3, "sp500"), pair(1.0, 0.2, "nasdaq100")]), /^Narrow tape/);
  assert.match(summarizeBreadth([pair(0.3, 1.2, "sp500"), pair(0.2, 1.0, "nasdaq100")]), /^Broad tape/);
});

test("says mixed when the two indices disagree, rather than picking one", () => {
  const mixed = summarizeBreadth([pair(1.2, 0.3, "sp500"), pair(0.2, 1.0, "nasdaq100")]);
  assert.match(mixed, /Mixed/);
});

test("hedges when only one of the two leans", () => {
  assert.match(summarizeBreadth([pair(1.2, 0.3, "sp500"), pair(1.0, 1.0, "nasdaq100")]), /Leaning narrow/);
  assert.match(summarizeBreadth([pair(0.3, 1.2, "sp500"), pair(1.0, 1.0, "nasdaq100")]), /Leaning broad/);
});

test("an even board says so without inventing a concern", () => {
  assert.match(summarizeBreadth([pair(1.0, 1.0, "sp500"), pair(0.5, 0.5, "nasdaq100")]), /even/);
  assert.equal(summarizeBreadth([]), "");
});

test("no reading forecasts or advises", () => {
  const forbidden = /\b(will|should|buy|sell|expect|forecast|predict|recommend)\b/i;
  const cases = [pair(-1.19, 0.52), pair(1.2, 0.3), pair(0.8, -0.3), pair(1.0, 1.0)];
  for (const built of cases) {
    assert.equal(forbidden.test(built.reading), false, `"${built.reading}" reads as advice`);
  }
  assert.equal(forbidden.test(summarizeBreadth(cases)), false);
});
