import assert from "node:assert/strict";
import test from "node:test";

import {
  ACCUSATORY_SOURCE_KINDS,
  buildCorpusChips,
  normalizePublishedDate,
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

test("gates the corpus's real enforcement kinds, including the biggest one", () => {
  // The first version of this set was guessed and missed doj_usao_press_release,
  // which is the largest enforcement source in the corpus.
  for (const kind of ["doj_usao_press_release", "sec_enforcement_litigation", "finra_awc"]) {
    assert.equal(ACCUSATORY_SOURCE_KINDS.has(kind), true, `${kind} must be gated`);
    assert.equal(buildCorpusChips([row({ source_kind: kind, confidence: 0.6 })]).size, 0);
    assert.equal(buildCorpusChips([row({ source_kind: kind, confidence: 1.0 })]).size, 1);
  }
});

test("names known source kinds the way a reader would", () => {
  assert.equal(sourceLabel("sec_enforcement_litigation"), "SEC enforcement");
  assert.equal(sourceLabel("finra_awc"), "FINRA action");
  assert.equal(sourceLabel("doj_usao_press_release"), "DOJ charge");
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


// ── published_date is TEXT and holds two shapes ──────────────────────────────

test("normalizes both date shapes the corpus actually contains", () => {
  assert.equal(normalizePublishedDate("2026-08-19T02:07:24Z"), "2026-08-19");
  assert.equal(normalizePublishedDate("August 18, 2026"), "2026-08-18");
  assert.equal(normalizePublishedDate("2026-08-19"), "2026-08-19");
  assert.equal(normalizePublishedDate("December 1, 2025"), "2025-12-01");
  assert.equal(normalizePublishedDate("  March 7, 2024  "), "2024-03-07");
});

test("returns null for anything it does not recognize", () => {
  assert.equal(normalizePublishedDate(""), null);
  assert.equal(normalizePublishedDate("   "), null);
  assert.equal(normalizePublishedDate("Smarch 40, 2026"), null);
  assert.equal(normalizePublishedDate("last Tuesday"), null);
});

test("the written form is windowed correctly, not silently dropped", () => {
  // The bug this replaced: "August 18, 2026" >= "2026-07-22" is true only
  // because "A" sorts above "2", so string comparison passed every such row
  // through a lower bound and failed every upper bound - dropping the whole
  // newsapi half of the corpus.
  const window = { since: "2026-07-22", until: "2026-08-22" };
  const inWindow = buildCorpusChips([row({ published_date: "August 18, 2026" })], window);
  assert.equal(inWindow.get("NVDA")?.[0].publishedDate, "2026-08-18");

  const oldWrittenDate = buildCorpusChips([row({ published_date: "August 18, 2019" })], window);
  assert.equal(oldWrittenDate.size, 0, "an old written-form date must not slip through");
});

test("windows ISO dates at both ends", () => {
  const window = { since: "2026-07-22", until: "2026-08-22" };
  assert.equal(buildCorpusChips([row({ published_date: "2026-08-19T02:07:24Z" })], window).size, 1);
  assert.equal(buildCorpusChips([row({ published_date: "2026-07-01T00:00:00Z" })], window).size, 0);
  assert.equal(buildCorpusChips([row({ published_date: "2026-09-01T00:00:00Z" })], window).size, 0);
});

test("orders newest first across mixed formats, not by raw string", () => {
  const rows = [
    row({ document_id: "a", url: "https://e.test/a", published_date: "August 1, 2026", title: "older written" }),
    row({ document_id: "b", url: "https://e.test/b", published_date: "2026-08-20T00:00:00Z", title: "newest iso" }),
    row({ document_id: "c", url: "https://e.test/c", published_date: "August 15, 2026", title: "middle written" }),
  ];
  const chips = buildCorpusChips(rows, { since: "2026-07-01", until: "2026-08-31" });
  assert.deepEqual(chips.get("NVDA")?.map((c) => c.title), ["newest iso", "middle written"]);
});

test("an unparseable date is dropped rather than guessed at", () => {
  assert.equal(buildCorpusChips([row({ published_date: "last Tuesday" })]).size, 0);
});
