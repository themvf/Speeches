// Tests for the entity alias map's TS side (see CLAUDE.md "Entity
// normalization / alias map"). The normalization fixture pairs here are
// mirrored in tests/test_entity_aliases.py - the two suites pin the TS and
// Python implementations to identical behavior. If you change a pair here,
// change it there too.
//
// Run with: npm run test:entity-aliases (node --test; no DB, no Next).
import assert from "node:assert/strict";
import test from "node:test";

import {
  canonicalEntityLabel,
  canonicalNormalizedEntityValue,
  entityAliasPairs,
  normalizeMentionValue,
} from "./entity-aliases.ts";
import entityAliasConfig from "./entity-aliases.json" with { type: "json" };

const NORMALIZATION_FIXTURES: Array<[string, string]> = [
  ["SEC", "sec"],
  ["  The U.S. Securities & Exchange Commission  ", "the u s securities exchange commission"],
  ["O'Brien-Smith", "obrien smith"],
  ["“Smart” quotes aren’t kept", "smart quotes arent kept"],
  ["J.P. Morgan", "j p morgan"],
  ["", ""],
];

test("normalizeMentionValue matches the Python fixture pairs", () => {
  for (const [raw, expected] of NORMALIZATION_FIXTURES) {
    assert.equal(normalizeMentionValue(raw), expected, raw);
  }
});

test("known alias pairs collapse to the same canonical label and normalized value", () => {
  const variants = [
    "SEC",
    "sec",
    "Securities and Exchange Commission",
    "U.S. Securities and Exchange Commission",
    "Securities & Exchange Commission",
    "S.E.C.",
    "the Commission",
  ];
  for (const variant of variants) {
    assert.equal(canonicalEntityLabel(variant), "SEC", variant);
    assert.equal(canonicalNormalizedEntityValue(variant), "sec", variant);
  }
});

test("unknown entities pass through unchanged", () => {
  assert.equal(canonicalEntityLabel("Acme Widgets LLC"), "Acme Widgets LLC");
  assert.equal(canonicalNormalizedEntityValue("Acme Widgets LLC"), "acme widgets llc");
  assert.equal(canonicalEntityLabel(""), "");
});

test("alias config has no duplicate normalized keys across entities", () => {
  // Two canonical entities claiming the same normalized alias would make
  // resolution order-dependent (first entry silently wins on both sides).
  const seen = new Map<string, string>();
  for (const entry of entityAliasConfig.entities) {
    for (const alias of [entry.canonical, ...entry.aliases]) {
      const key = normalizeMentionValue(alias);
      assert.ok(key, `alias ${JSON.stringify(alias)} normalizes to empty`);
      const owner = seen.get(key);
      assert.ok(
        owner === undefined || owner === entry.canonical,
        `normalized alias ${JSON.stringify(key)} claimed by both ${JSON.stringify(owner)} and ${JSON.stringify(entry.canonical)}`
      );
      seen.set(key, entry.canonical);
    }
  }
});

test("entityAliasPairs excludes identity mappings and stays label-consistent", () => {
  const pairs = entityAliasPairs();
  assert.ok(pairs.length > 0, "expected at least one non-identity alias pair");
  for (const pair of pairs) {
    assert.notEqual(pair.aliasNormalized, pair.canonicalNormalized);
    assert.equal(normalizeMentionValue(pair.canonicalLabel), pair.canonicalNormalized);
  }
  assert.ok(pairs.every((pair) => pair.aliasNormalized !== "sec"));
});
