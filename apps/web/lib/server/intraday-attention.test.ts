// Tests for the Hot Right Now (intraday) view's pure computation (SEC-22).
// Run with: npm run test:intraday-attention (node --test; no DB, no Next).
import assert from "node:assert/strict";
import test from "node:test";

import {
  computeIntradayRows,
  computeMovers,
  moodPlurality,
  type IntradayMentionInput,
} from "./intraday-attention.ts";

const HOUR = 60 * 60 * 1000;
const NOW = Date.parse("2026-07-14T12:00:00.000Z");

function mention(ticker: string, author: string, hoursAgo: number, mood?: string): IntradayMentionInput {
  return { ticker, author, created_utc: new Date(NOW - hoursAgo * HOUR).toISOString(), mood };
}

test("computeIntradayRows dedupes by distinct author and decays by recency", () => {
  const mentions: IntradayMentionInput[] = [
    mention("NVDA", "a", 0),
    mention("NVDA", "b", 0),
    mention("NVDA", "a", 0.5), // same author again - dedup keeps latest, doesn't double count
    mention("NVDA", "c", 9),   // very old - decays to near-zero weight
    mention("TSLA", "d", 3),   // exactly one half-life
  ];
  const rows = computeIntradayRows(mentions, NOW, 30);
  const byTicker = new Map(rows.map((r) => [r.ticker, r]));

  assert.equal(byTicker.get("NVDA")?.rawMentionCount, 3);
  assert.ok(byTicker.get("NVDA")!.decayedMentionCount > 1.9); // ~2 fresh + a sliver from the old one
  assert.equal(byTicker.get("TSLA")?.rawMentionCount, 1);
  // 3h at a 3h half-life -> weight ~= e^-1 ~= 0.368
  assert.ok(Math.abs(byTicker.get("TSLA")!.decayedMentionCount - 0.37) < 0.02);

  // freshnessRatio = decayed/raw; TSLA's single old-ish mention has a lower
  // ratio than NVDA's mostly-fresh cluster.
  assert.ok(byTicker.get("TSLA")!.freshnessRatio < byTicker.get("NVDA")!.freshnessRatio);

  // Ranked and capped.
  assert.deepEqual(rows.map((r) => r.rank), rows.map((_, i) => i + 1));
});

test("moodPlurality follows the daily rollup's rules", () => {
  assert.equal(moodPlurality([]), "neutral");
  assert.equal(moodPlurality(["neutral", "neutral"]), "neutral");
  assert.equal(moodPlurality(["bullish", "neutral"]), "bullish");
  assert.equal(moodPlurality(["bearish"]), "bearish");
  assert.equal(moodPlurality(["bullish", "bearish"]), "mixed");
  assert.equal(moodPlurality(["bullish", "bullish", "bearish"]), "bullish");
  assert.equal(moodPlurality(["bearish", "bearish", "bullish", "neutral"]), "bearish");
});

test("computeIntradayRows aggregates mood per ticker from each author's latest mention", () => {
  const mentions: IntradayMentionInput[] = [
    // Author a flipped bearish -> bullish; only the latest (bullish) counts.
    mention("NVDA", "a", 5, "bearish"),
    mention("NVDA", "a", 1, "bullish"),
    mention("NVDA", "b", 1, "bullish"),
    mention("NVDA", "c", 1, "bearish"),
    // TSLA: one bull, one bear -> mixed. GME: no directional votes -> neutral.
    mention("TSLA", "d", 1, "bullish"),
    mention("TSLA", "e", 1, "bearish"),
    mention("GME", "f", 1, "neutral"),
    mention("GME", "g", 1), // absent mood treated as neutral
  ];
  const byTicker = new Map(computeIntradayRows(mentions, NOW, 30).map((r) => [r.ticker, r.mood]));
  assert.equal(byTicker.get("NVDA"), "bullish"); // 2 bull vs 1 bear; a's stale bearish vote superseded
  assert.equal(byTicker.get("TSLA"), "mixed");
  assert.equal(byTicker.get("GME"), "neutral");
});

test("computeIntradayRows respects topN and sorts by decayed count desc", () => {
  const mentions: IntradayMentionInput[] = [
    mention("A", "u1", 0),
    mention("B", "u1", 0),
    mention("B", "u2", 0),
    mention("C", "u1", 0),
    mention("C", "u2", 0),
    mention("C", "u3", 0),
  ];
  const rows = computeIntradayRows(mentions, NOW, 2);
  assert.equal(rows.length, 2);
  assert.deepEqual(rows.map((r) => r.ticker), ["C", "B"]);
});

test("computeMovers is gated off below the required window", () => {
  const mentions: IntradayMentionInput[] = [mention("NVDA", "a", 1)];
  const result = computeMovers(mentions, NOW, 5); // needs 6h (3+3)
  assert.deepEqual(result, { heatingUp: [], coolingOff: [] });
});

test("computeMovers classifies heating, cooling, flat, and brand-new tickers", () => {
  const mentions: IntradayMentionInput[] = [
    // NVDA: 1 -> 3 distinct authors (heating, quantified +200%)
    mention("NVDA", "a", 4.5),
    mention("NVDA", "x", 1), mention("NVDA", "y", 1), mention("NVDA", "z", 1),
    // TSLA: 4 -> 1 (cooling, -75%)
    mention("TSLA", "a", 5), mention("TSLA", "b", 5), mention("TSLA", "c", 5), mention("TSLA", "d", 4.9),
    mention("TSLA", "e", 1),
    // GME: 2 -> 2 (flat, excluded entirely)
    mention("GME", "a", 5), mention("GME", "b", 4.9),
    mention("GME", "a", 1), mention("GME", "b", 1),
    // RIVN: 0 -> 3 (brand new - null changePct, ranks above quantified heating)
    mention("RIVN", "p", 1), mention("RIVN", "q", 1), mention("RIVN", "r", 1),
    // AMC: 1 -> 1 total mentions, below the min-combined-mentions floor - excluded as noise
    mention("AMC", "a", 5), mention("AMC", "a", 1),
  ];
  const { heatingUp, coolingOff } = computeMovers(mentions, NOW, 24);

  const heatingTickers = heatingUp.map((r) => r.ticker);
  const coolingTickers = coolingOff.map((r) => r.ticker);
  assert.ok(!heatingTickers.includes("GME") && !coolingTickers.includes("GME"), "flat ticker excluded");
  assert.ok(!heatingTickers.includes("AMC") && !coolingTickers.includes("AMC"), "below min-combined-mentions floor excluded");

  // Brand-new (null changePct) ranks above a quantified increase.
  assert.deepEqual(heatingTickers, ["RIVN", "NVDA"]);
  assert.equal(heatingUp[0]!.changePct, null);
  assert.equal(heatingUp[1]!.ticker, "NVDA");
  assert.equal(heatingUp[1]!.recentCount, 3);
  assert.equal(heatingUp[1]!.priorCount, 1);
  assert.equal(heatingUp[1]!.changePct, 200);

  assert.deepEqual(coolingTickers, ["TSLA"]);
  assert.equal(coolingOff[0]!.recentCount, 1);
  assert.equal(coolingOff[0]!.priorCount, 4);
  assert.equal(coolingOff[0]!.changePct, -75);
});

test("computeMovers caps each list at MOVER_TOP_N", () => {
  // 12 tickers, each going 1 -> 2 distinct authors (heating).
  const mentions: IntradayMentionInput[] = [];
  for (let i = 0; i < 12; i++) {
    const ticker = `T${i}`;
    mentions.push(mention(ticker, "old1", 5));
    mentions.push(mention(ticker, "new1", 1));
    mentions.push(mention(ticker, "new2", 1));
  }
  const { heatingUp } = computeMovers(mentions, NOW, 24);
  assert.equal(heatingUp.length, 8); // MOVER_TOP_N
});
