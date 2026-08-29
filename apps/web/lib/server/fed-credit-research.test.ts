import assert from "node:assert/strict";
import test from "node:test";
import { parseFedCreditResearchCsv } from "./fed-credit-research.ts";

test("parses Fed credit research and derives the default-risk component", () => {
  const points = parseFedCreditResearchCsv([
    "date,gz_spread,ebp,est_prob",
    "2/1/2026,1.2,0.2,0.25",
    "1/1/2026,1.0,-0.1,0.15",
    "bad,row",
  ].join("\n"));
  assert.deepEqual(points, [
    { date: "2026-01-01", corporateSpread: 1, excessBondPremium: -0.1, defaultRiskComponent: 1.1, recessionProbability: 0.15 },
    { date: "2026-02-01", corporateSpread: 1.2, excessBondPremium: 0.2, defaultRiskComponent: 1, recessionProbability: 0.25 },
  ]);
});
