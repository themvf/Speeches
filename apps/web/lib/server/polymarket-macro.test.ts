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
    event("jobs", "How many jobs added in July?", "2026-08-07T12:30:00Z", 800, "100k to 150k"),
    event("unemployment", "July Unemployment Rate", "2026-08-07T12:30:00Z", 700, "4.1%"),
    event("gdp", "US GDP growth in Q2 2026?", "2026-07-30T12:30:00Z", 600, "2% to 3%"),
  ], new Date("2026-07-16T12:00:00Z"));

  assert.equal(events.find((item) => item.mappingKey === "fed_next_decision")?.eventId, "fed-july");
  assert.equal(events.find((item) => item.mappingKey === "nonfarm_payrolls_next")?.indicatorIds[0], "nonfarm_payrolls");
  assert.equal(events.find((item) => item.mappingKey === "unemployment_next")?.matchKind, "exact_series");
  assert.equal(events.find((item) => item.mappingKey === "us_gdp_next")?.indicatorIds[0], "real_gdp_growth");
});

test("deduplicates tag results, parses probabilities, and ignores malformed markets", () => {
  const valid = event("gdp", "US GDP growth in Q2 2026?", "2026-07-30T12:30:00Z", 1000, "2% to 3%");
  valid.markets?.push({ id: "bad", outcomes: "not json", outcomePrices: "[]" });
  const events = buildMacroPredictionEvents([valid, valid]);
  const result = events.find((item) => item.mappingKey === "us_gdp_next");

  assert.ok(result);
  assert.equal(result.outcomes.length, 1);
  assert.equal(result.leadingOutcome?.label, "2% to 3%");
  assert.equal(result.leadingOutcome?.probability, 0.64);
  assert.equal(result.leadingOutcome?.oneDayChange, 0.03);
  assert.match(result.matchNote, /quarterly/i);
  assert.equal(result.url, "https://polymarket.com/event/gdp");
});

test("selects the nearest recurring CPI release and clamps invalid probabilities", () => {
  const july = event("cpi-july", "July Inflation US - Annual", "2026-08-12T12:30:00Z", 50);
  const august = event("cpi-august", "August Inflation US - Annual", "2026-09-11T12:30:00Z", 500);
  if (july.markets) july.markets[0].outcomePrices = JSON.stringify(["1.4", "-0.4"]);
  const result = buildMacroPredictionEvents([august, july], new Date("2026-07-16T12:00:00Z")).find((item) => item.mappingKey === "headline_cpi_annual_next");

  assert.equal(result?.eventId, "cpi-july");
  assert.equal(result?.leadingOutcome?.probability, 1);
});
