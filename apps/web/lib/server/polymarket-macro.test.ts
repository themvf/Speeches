import assert from "node:assert/strict";
import test from "node:test";
import { buildMacroPredictionEvents, type GammaEvent } from "./polymarket-macro.ts";

function event(id: string, title: string, endDate: string, volume: number, label = "Yes"): GammaEvent {
  return {
    id,
    slug: id,
    title,
    endDate,
    volume,
    markets: [{
      id: `${id}-market`,
      conditionId: `${id}-condition`,
      groupItemTitle: label,
      outcomes: JSON.stringify(["Yes", "No"]),
      outcomePrices: JSON.stringify(["0.64", "0.36"]),
      volumeNum: volume,
      liquidityNum: 2500,
      oneDayPriceChange: 0.03,
      closed: false,
    }],
  };
}

test("maps supported macro events and selects the nearest upcoming Fed decision", () => {
  const events = buildMacroPredictionEvents([
    event("fed-september", "Fed Decision in September?", "2026-09-16T12:00:00Z", 900),
    event("fed-july", "Fed Decision in July?", "2026-07-29T12:00:00Z", 300),
    event("unemployment", "How high will US unemployment go in 2026?", "2026-12-31T12:00:00Z", 800, "Above 5%"),
    event("recession", "US recession by end of 2026?", "2026-12-31T12:00:00Z", 700),
    event("mortgage", "Will the 30-year Mortgage Rate hit 7% in 2026?", "2026-12-31T12:00:00Z", 600),
  ], new Date("2026-07-16T12:00:00Z"));

  assert.equal(events.find((item) => item.mappingKey === "fed_next_decision")?.eventId, "fed-july");
  assert.equal(events.find((item) => item.mappingKey === "unemployment_annual_high")?.matchKind, "exact_series");
  assert.equal(events.find((item) => item.mappingKey === "recession_year")?.matchKind, "related_signal");
  assert.equal(events.find((item) => item.mappingKey === "mortgage_annual_range")?.indicatorIds[0], "mortgage_rate_30y");
});

test("deduplicates tag results, parses probabilities, and ignores malformed markets", () => {
  const valid = event("gdp", "GDP growth in 2026", "2026-12-31T12:00:00Z", 1000, "2% to 3%");
  valid.markets?.push({ id: "bad", outcomes: "not json", outcomePrices: "[]" });
  const events = buildMacroPredictionEvents([valid, valid]);
  const result = events.find((item) => item.mappingKey === "gdp_full_year");

  assert.ok(result);
  assert.equal(result.outcomes.length, 1);
  assert.equal(result.leadingOutcome?.label, "2% to 3%");
  assert.equal(result.leadingOutcome?.probability, 0.64);
  assert.equal(result.leadingOutcome?.oneDayChange, 0.03);
  assert.match(result.matchNote, /full-year/i);
  assert.equal(result.url, "https://polymarket.com/event/gdp");
});

test("prefers the highest-volume annual contract and clamps invalid probabilities", () => {
  const low = event("cpi-low", "How high will inflation get in 2026?", "2026-12-31T12:00:00Z", 50);
  const high = event("cpi-high", "How high will inflation get in 2026?", "2026-12-31T12:00:00Z", 500);
  if (high.markets) high.markets[0].outcomePrices = JSON.stringify(["1.4", "-0.4"]);
  const result = buildMacroPredictionEvents([low, high]).find((item) => item.mappingKey === "inflation_annual_tail");

  assert.equal(result?.eventId, "cpi-high");
  assert.equal(result?.leadingOutcome?.probability, 1);
});
