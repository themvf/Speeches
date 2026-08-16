import assert from "node:assert/strict";
import test from "node:test";

import {
  CAPITAL_FORMATION_BROAD_TERMS,
  CAPITAL_FORMATION_FOCUS_AREAS,
  CAPITAL_FORMATION_KEYWORDS,
  CAPITAL_FORMATION_PATTERN_ALIASES,
  CAPITAL_FORMATION_SORT_ORDER,
  CAPITAL_FORMATION_TOPIC_KEY,
  capitalFormationKeywordRegex,
  capitalFormationQueryTerms,
} from "./capital-formation-vocabulary.ts";

// Fixtures mirrored in tests/test_capital_formation_vocabulary.py. Change one,
// change the other - that pairing is what keeps the TS and Python readers from
// drifting the way the seven hand-maintained copies did.
const MIRRORED_SORT_ORDER = 12;
const MIRRORED_TOPIC_KEY = "CAPITAL_FORMATION";
const MIRRORED_FOCUS_AREA_IDS = [
  "capital_public_offerings",
  "capital_private_capital",
  "capital_direct_participation",
  "capital_debt_financing",
  "capital_strategic_transactions",
  "capital_access_policy",
];
const MIRRORED_REQUIRED_KEYWORDS = [
  "capital formation",
  "Rule 506(b)",
  "Rule 506(c)",
  "Reg CF",
  "non-traded REIT",
  "direct participation program",
  "Delaware statutory trust",
  "unregistered broker",
  "blue sky preemption",
  "no-action letter",
];

test("exposes the shared vocabulary identity", () => {
  assert.equal(CAPITAL_FORMATION_TOPIC_KEY, MIRRORED_TOPIC_KEY);
  assert.equal(CAPITAL_FORMATION_SORT_ORDER, MIRRORED_SORT_ORDER);
  assert.deepEqual(CAPITAL_FORMATION_FOCUS_AREAS.map((area) => area.id), MIRRORED_FOCUS_AREA_IDS);
  for (const keyword of MIRRORED_REQUIRED_KEYWORDS) {
    assert.ok(CAPITAL_FORMATION_KEYWORDS.includes(keyword), `missing keyword: ${keyword}`);
  }
});

test("keeps the keyword list free of duplicates and of its own broad terms", () => {
  const normalized = CAPITAL_FORMATION_KEYWORDS.map((k) => k.toLowerCase().trim());
  assert.equal(new Set(normalized).size, normalized.length, "duplicate keyword in vocabulary");

  // broadTerms drives an "Avoid As Standalone Signals" warning in the admin
  // panel that fires whenever a broad term is also an active keyword. Any
  // overlap would flag this rule permanently and train people to ignore it.
  for (const term of CAPITAL_FORMATION_BROAD_TERMS) {
    assert.ok(!normalized.includes(term.toLowerCase()), `broad term is also a keyword: ${term}`);
  }
});

test("omits form types that only appear in filing metadata", () => {
  // Reg A / Reg CF ongoing-reporting forms are handled by filing ingestion, not
  // keyword matching; they essentially never appear in prose. See the note in
  // the JSON config.
  const normalized = CAPITAL_FORMATION_KEYWORDS.map((k) => k.toLowerCase());
  for (const form of ["form 1-k", "form 1-sa", "form 1-u", "form c-ar", "form c-u"]) {
    assert.ok(!normalized.includes(form), `filing-only form type leaked into keywords: ${form}`);
  }
});

test("every focus area carries patterns and visible-text query terms", () => {
  for (const area of CAPITAL_FORMATION_FOCUS_AREAS) {
    assert.ok(area.label.length > 0, `${area.id} has no label`);
    assert.ok(area.weight > 0, `${area.id} has no weight`);
    assert.ok(area.rawPatterns.length > 0, `${area.id} has no raw patterns`);
    // A focus area with no query terms silently falls back to matching
    // UPPER_SNAKE raw patterns against visible text, which never matches.
    assert.ok(area.queryTerms.length > 0, `${area.id} has no query terms`);
  }

  const queryTerms = capitalFormationQueryTerms();
  assert.deepEqual(Object.keys(queryTerms).sort(), [...MIRRORED_FOCUS_AREA_IDS].sort());
});

test("ranks policy and offering focus areas above plain deal coverage", () => {
  const byId = new Map(CAPITAL_FORMATION_FOCUS_AREAS.map((area) => [area.id, area]));
  const mergers = byId.get("capital_strategic_transactions")?.weight ?? 0;
  for (const id of ["capital_access_policy", "capital_public_offerings", "capital_direct_participation"]) {
    assert.ok((byId.get(id)?.weight ?? 0) > mergers, `${id} should outweigh M&A coverage`);
  }
});

test("keyword regex prefers the most specific term and rejects near-misses", () => {
  const re = capitalFormationKeywordRegex();
  assert.ok(re.test("the issuer completed a rule 506(b) private stock offering"));
  assert.ok(re.test("sponsor launches a non-traded reit"));
  assert.ok(re.test("sec staff issued a no-action letter"));
  // Longest-first alternation: the specific rule wins over bare "rule 506".
  assert.equal("closed under rule 506(c) yesterday".match(re)?.[0], "rule 506(c)");
  // Rule 144 and Rule 144A are different exemptions and must not collide.
  assert.equal("resold under rule 144a to qibs".match(re)?.[0], "rule 144a");
  assert.equal(re.test("the pipeline burst overnight"), false);
  assert.equal(re.test("quarterly earnings beat expectations"), false);
});

test("pattern aliases resolve both directions for the paired product terms", () => {
  assert.ok(CAPITAL_FORMATION_PATTERN_ALIASES.BDC?.includes("BUSINESS_DEVELOPMENT_COMPANY"));
  assert.ok(CAPITAL_FORMATION_PATTERN_ALIASES.BUSINESS_DEVELOPMENT_COMPANY?.includes("BDC"));
  assert.ok(CAPITAL_FORMATION_PATTERN_ALIASES.NON_TRADED_REIT?.includes("NONTRADED_REIT"));
  assert.ok(CAPITAL_FORMATION_PATTERN_ALIASES.DELAWARE_STATUTORY_TRUST?.includes("DST"));
});
