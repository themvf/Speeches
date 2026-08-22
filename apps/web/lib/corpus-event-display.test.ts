import assert from "node:assert/strict";
import test from "node:test";

import {
  buildCorpusChips,
  MAX_CHIPS_PER_TICKER,
  sourceLabel,
  type CorpusEventInput,
} from "./corpus-event-display.ts";

function row(overrides: Partial<CorpusEventInput> = {}): CorpusEventInput {
  return {
    ticker: "NVDA",
    document_id: "doc-1",
    title: "A document title",
    source_kind: "sec_speech",
    published_date: "2026-08-20",
    url: "https://example.test/doc-1",
    confidence: 1.0,
    ...overrides,
  };
}

test("names known source kinds the way a reader would", () => {
  assert.equal(sourceLabel("sec_enforcement_litigation"), "SEC enforcement");
  assert.equal(sourceLabel("finra_awc"), "FINRA action");
  assert.equal(sourceLabel("cfpb_newsroom"), "CFPB release");
});

test("falls back to a readable form rather than inventing a label", () => {
  assert.equal(sourceLabel("some_new_source"), "Some New Source");
});

test("carries the document's own title through verbatim", () => {
  const title = "SEC Charges Firm With Misleading Investors About AI Capabilities";
  const chips = buildCorpusChips([row({ title })]);
  assert.equal(chips.get("NVDA")?.[0].title, title);
});

test("a title match is marked as the subject; a body match is not", () => {
  assert.equal(buildCorpusChips([row({ confidence: 1.0 })]).get("NVDA")?.[0].subject, true);
  assert.equal(buildCorpusChips([row({ confidence: 0.6 })]).get("NVDA")?.[0].subject, false);
});

test("an accusatory source kind is suppressed on a body-only match", () => {
  // The company was named somewhere in an enforcement action's body. That is
  // not enough to put an "SEC enforcement" chip against its ticker.
  const chips = buildCorpusChips([
    row({ source_kind: "sec_enforcement_litigation", confidence: 0.6 }),
  ]);
  assert.equal(chips.size, 0);
});

test("an accusatory source kind survives a title match", () => {
  const chips = buildCorpusChips([
    row({ source_kind: "sec_enforcement_litigation", confidence: 1.0 }),
  ]);
  assert.equal(chips.get("NVDA")?.[0].sourceLabel, "SEC enforcement");
  assert.equal(chips.get("NVDA")?.[0].subject, true);
});

test("a non-accusatory kind is allowed on a body-only match", () => {
  const chips = buildCorpusChips([row({ source_kind: "sec_speech", confidence: 0.6 })]);
  assert.equal(chips.get("NVDA")?.length, 1);
});

test("drops rows with nothing to link to or nothing to say", () => {
  assert.equal(buildCorpusChips([row({ url: "" })]).size, 0);
  assert.equal(buildCorpusChips([row({ title: "" })]).size, 0);
  assert.equal(buildCorpusChips([row({ ticker: "" })]).size, 0);
});

test("caps chips per ticker, keeping the earliest rows given", () => {
  const rows = [1, 2, 3, 4].map((n) =>
    row({ document_id: `doc-${n}`, title: `Title ${n}`, url: `https://example.test/${n}` }),
  );
  const chips = buildCorpusChips(rows);
  assert.equal(chips.get("NVDA")?.length, MAX_CHIPS_PER_TICKER);
  // Rows arrive newest-first, so the cap must keep the front of the list.
  assert.deepEqual(chips.get("NVDA")?.map((c) => c.title), ["Title 1", "Title 2"]);
});

test("groups independently per ticker", () => {
  const chips = buildCorpusChips([
    row({ ticker: "NVDA", document_id: "a", url: "https://example.test/a" }),
    row({ ticker: "MSFT", document_id: "b", url: "https://example.test/b" }),
  ]);
  assert.deepEqual([...chips.keys()].sort(), ["MSFT", "NVDA"]);
});

test("an empty result is an empty map, not a throw", () => {
  assert.equal(buildCorpusChips([]).size, 0);
});
