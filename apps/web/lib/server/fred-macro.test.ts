import assert from "node:assert/strict";
import test from "node:test";

import {
  buildMacroIndicator,
  FRED_MACRO_DEFINITIONS,
  parseFredObservations,
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
    "DTWEXBGS",
    "HOUST",
    "PERMIT",
    "MORTGAGE30US",
  ]);
  assert.equal(new Set(FRED_MACRO_DEFINITIONS.map((definition) => definition.id)).size, 25);
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
