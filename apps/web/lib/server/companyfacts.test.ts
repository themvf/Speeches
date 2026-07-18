// Tests for the companyfacts quarterly extraction (SEC-54).
// Run with: npm run test:companyfacts (node --test).
import assert from "node:assert/strict";
import test from "node:test";

import { extractQuarterlySeries, type CompanyFactsJson } from "./companyfacts.ts";

const FACTS: CompanyFactsJson = {
  entityName: "TestCo",
  facts: {
    "us-gaap": {
      Revenues: {
        units: {
          USD: [
            // Quarterly frames (canonical) - including a duplicate end date
            // from a later filing that must overwrite, not double-count.
            { start: "2025-10-01", end: "2025-12-31", val: 100, frame: "CY2025Q4" },
            { start: "2026-01-01", end: "2026-03-31", val: 110, frame: "CY2026Q1" },
            { start: "2026-01-01", end: "2026-03-31", val: 111, frame: "CY2026Q1" },
            // Cumulative 6-month fact - NO quarterly frame -> excluded (the
            // cumulative-vs-quarterly trap).
            { start: "2025-10-01", end: "2026-03-31", val: 210 },
            // Annual frame -> excluded.
            { start: "2025-01-01", end: "2025-12-31", val: 400, frame: "CY2025" },
            // Instant-style frame -> excluded by the flow regex.
            { end: "2026-03-31", val: 999, frame: "CY2026Q1I" },
          ],
        },
      },
      RevenueFromContractWithCustomerExcludingAssessedTax: { units: { USD: [] } },
    },
  },
};

test("extractQuarterlySeries keeps only quarterly frames, deduped by period end", () => {
  const series = extractQuarterlySeries(FACTS, ["Revenues"], "USD");
  assert.deepEqual(series, [
    { end: "2025-12-31", value: 100 },
    { end: "2026-03-31", value: 111 }, // later filing wins the duplicate end
  ]);
});

test("extractQuarterlySeries walks the concept fallback chain", () => {
  // Primary concept has no rows -> falls through to Revenues.
  const series = extractQuarterlySeries(FACTS, ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"], "USD");
  assert.equal(series.length, 2);
});

test("extractQuarterlySeries prefers the FRESHEST concept, not the first non-empty", () => {
  // The CAT case: NetIncomeLoss carries only decade-old frames while
  // ProfitLoss has the current ones - recency must win over chain order.
  const facts: CompanyFactsJson = {
    facts: {
      "us-gaap": {
        NetIncomeLoss: { units: { USD: [
          { start: "2011-07-01", end: "2011-09-30", val: 1, frame: "CY2011Q3" },
        ] } },
        ProfitLoss: { units: { USD: [
          { start: "2026-01-01", end: "2026-03-31", val: 42, frame: "CY2026Q1" },
        ] } },
      },
    },
  };
  const series = extractQuarterlySeries(facts, ["NetIncomeLoss", "ProfitLoss"], "USD");
  assert.deepEqual(series, [{ end: "2026-03-31", value: 42 }]);
});

test("extractQuarterlySeries caps at the requested quarter count", () => {
  const many: CompanyFactsJson = {
    facts: {
      "us-gaap": {
        Revenues: {
          units: {
            USD: Array.from({ length: 12 }, (_, i) => ({
              start: `20${20 + Math.floor(i / 4)}-01-01`,
              end: `202${Math.floor(i / 4)}-0${(i % 4) * 3 + 1}-30`,
              val: i,
              frame: `CY202${Math.floor(i / 4)}Q${(i % 4) + 1}`,
            })),
          },
        },
      },
    },
  };
  assert.equal(extractQuarterlySeries(many, ["Revenues"], "USD", 8).length, 8);
});

test("extractQuarterlySeries returns empty for unknown concepts/units", () => {
  assert.deepEqual(extractQuarterlySeries(FACTS, ["NoSuchConcept"], "USD"), []);
  assert.deepEqual(extractQuarterlySeries(FACTS, ["Revenues"], "EUR"), []);
});
