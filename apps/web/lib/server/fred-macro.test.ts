import assert from "node:assert/strict";
import test from "node:test";

import {
  buildMacroIndicator,
  FRED_MACRO_DEFINITIONS,
  parseFredObservations,
  seriesRefreshSeconds,
} from "./fred-macro.ts";

test("defines the complete grouped market macro catalog", () => {
  assert.deepEqual(FRED_MACRO_DEFINITIONS.map((definition) => definition.seriesId), [
    "A191RL1Q225SBEA",
    "CPIAUCSL",
    "PAYEMS",
    "UNRATE",
    "DFF",
    "T10Y2Y",
    "RSAFS",
    "INDPRO",
    "PCEPILFE",
    "T10YIE",
    "PPIFIS",
    "ICSA",
    "CES0500000003",
    "CIVPART",
    "JTSJOL",
    "SAHMREALTIME",
    "NFCI",
    "STLFSI4",
    "WALCL",
    "M2SL",
    "SOFR",
    "BAA10Y",
    "NFCICREDIT",
    "DFII10",
    "DTWEXBGS",
    "HOUST",
    "PERMIT",
    "MORTGAGE30US",
  ]);
  assert.equal(new Set(FRED_MACRO_DEFINITIONS.map((definition) => definition.id)).size, 28);
  assert.deepEqual(new Set(FRED_MACRO_DEFINITIONS.map((definition) => definition.group)), new Set([
    "headline", "activity", "inflation", "labor", "financial", "housing",
  ]));
});

test("scales source units into dashboard display units", () => {
  const definition = FRED_MACRO_DEFINITIONS.find((item) => item.seriesId === "WALCL");
  assert.ok(definition);
  const indicator = buildMacroIndicator(definition, [
    { date: "2026-06-03", value: 6_600_000 },
    { date: "2026-06-10", value: 6_625_000 },
  ], { frequency: "Weekly" });
  assert.equal(indicator.value, 6.625);
  assert.equal(indicator.previousValue, 6.6);
  assert.ok(Math.abs((indicator.change ?? 0) - 0.025) < 1e-12);
  assert.equal(indicator.unit, "trillions");
  assert.equal(indicator.group, "financial");
});

test("parses observations chronologically and discards missing values", () => {
  assert.deepEqual(parseFredObservations({ observations: [
    { date: "2026-02-01", value: "4.2" },
    { date: "2026-03-01", value: "." },
    { date: "2026-01-01", value: "4.3" },
  ] }), [
    { date: "2026-01-01", value: 4.3 },
    { date: "2026-02-01", value: 4.2 },
  ]);
});

test("builds current, previous, and change values from the latest observations", () => {
  const definition = FRED_MACRO_DEFINITIONS.find((item) => item.seriesId === "UNRATE");
  assert.ok(definition);
  const indicator = buildMacroIndicator(definition, [
    { date: "2026-05-01", value: 4.3 },
    { date: "2026-06-01", value: 4.2 },
  ], { frequency: "Monthly", lastUpdated: "2026-07-02 08:31:00-05" });
  assert.equal(indicator.value, 4.2);
  assert.equal(indicator.previousValue, 4.3);
  assert.ok(Math.abs((indicator.change ?? 0) + 0.1) < 1e-12);
  assert.equal(indicator.observationDate, "2026-06-01");
  assert.equal(indicator.frequency, "Monthly");
});

/**
 * Cadence drives how long an observation response is cached, so a wrong pin
 * silently makes a card stale rather than throwing. These are the frequencies
 * FRED reported for each series in a live production response on 2026-08-29 -
 * re-check against `/api/market/macro` if a definition is added or changed.
 */
const EXPECTED_CADENCE: Readonly<Record<string, string>> = {
  DFF: "daily", T10Y2Y: "daily", T10YIE: "daily", SOFR: "daily",
  DFII10: "daily", DTWEXBGS: "daily", BAA10Y: "daily",
  ICSA: "weekly", NFCI: "weekly", STLFSI4: "weekly", WALCL: "weekly",
  NFCICREDIT: "weekly", MORTGAGE30US: "weekly",
  CPIAUCSL: "monthly", PAYEMS: "monthly", UNRATE: "monthly", RSAFS: "monthly",
  INDPRO: "monthly", PCEPILFE: "monthly", PPIFIS: "monthly",
  CES0500000003: "monthly", CIVPART: "monthly", JTSJOL: "monthly",
  SAHMREALTIME: "monthly", M2SL: "monthly", HOUST: "monthly", PERMIT: "monthly",
  A191RL1Q225SBEA: "quarterly",
};

test("every macro series pins the cadence FRED actually publishes at", () => {
  for (const definition of FRED_MACRO_DEFINITIONS) {
    const expected = EXPECTED_CADENCE[definition.seriesId];
    assert.ok(expected, `${definition.seriesId} has no recorded publication frequency`);
    assert.equal(definition.cadence, expected,
      `${definition.seriesId} is pinned ${definition.cadence} but FRED publishes it ${expected}`);
  }
});

test("a daily series refreshes faster than a periodic one", () => {
  assert.ok(seriesRefreshSeconds("daily") < seriesRefreshSeconds("weekly"),
    "market rates move intraday; a monthly print does not");
  assert.ok(seriesRefreshSeconds("monthly") <= 60 * 60,
    "worst-case staleness for a new print should stay within an hour");
});
