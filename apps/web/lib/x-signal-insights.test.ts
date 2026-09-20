import assert from "node:assert/strict";
import test from "node:test";

import { buildXSignalInsights, isXSignalPost, suggestedXSignalSubjects } from "./x-signal-insights.ts";
import type { XSignalPost, XSignalWindow } from "./x-signal-insights.ts";

const NOW = Date.parse("2026-09-20T12:00:00Z");

function post(overrides: Partial<XSignalPost> = {}): XSignalPost {
  return {
    id: 1,
    guid: "x:backpack:12345678901",
    feed_key: "x_public_timeline_backpack",
    title: "Backpack announces a product update",
    description: "Backpack says its new product is available today. More details will follow.",
    author: "Backpack (@Backpack)",
    url: "https://x.com/Backpack/status/12345678901",
    published_at: "2026-09-20T10:00:00Z",
    fetched_at: "2026-09-20T11:00:00Z",
    ...overrides,
  };
}

function analysis(overrides: Record<string, unknown> = {}) {
  return {
    status: "enriched", fallback: false,
    thesis: "Backpack says it launched a new product.",
    why_it_matters: ["Availability may expand access for users."],
    follow_up_questions: ["Which regions are eligible?"],
    entities: ["Backpack"],
    ...overrides,
  };
}

function insights(posts: XSignalPost[], query = "", window: XSignalWindow = "7d") {
  return buildXSignalInsights(posts, { query, window, now: NOW });
}

test("identifies X by actual URL host or timeline key, never by mentions in news", () => {
  assert.equal(isXSignalPost(post({ feed_key: "other", url: "https://MOBILE.TWITTER.COM/Backpack/status/1" })), true);
  assert.equal(isXSignalPost(post({ url: "https://example.com" })), true);
  assert.equal(isXSignalPost(post({ feed_key: "news", url: "https://x.com/Backpack" })), false);
  for (const url of ["https://example.com/x.com/Backpack", "https://x.com.example.org/a", "https://x.com@example.org/a", "https://example.org/?link=https://x.com/a", "javascript:alert('x.com')"]) {
    assert.equal(isXSignalPost(post({ feed_key: "news", url })), false, url);
  }
});

test("deduplicates global tweet IDs across host, account case, tracking parameters and feeds", () => {
  const original = post({ analysis: analysis() });
  const duplicate = post({ id: 2, guid: "other-guid", feed_key: "rss-import", author: "@BACKPACK", url: "https://TWITTER.com/BACKPACK/status/12345678901?s=20" });
  const other = post({ id: 3, guid: "different", url: "https://x.com/backpack/status/22222222222", description: "Backpack says a second release is planned.", published_at: "2026-09-20T11:00:00Z" });
  const input = [duplicate, original, other];
  const result = insights(input, "Backpack");
  assert.deepEqual(result.posts, [other, original]);
  assert.equal(result.posts[1], original);
  assert.equal(result.accountCount, 1);
  assert.equal(result.latestPublishedAt, other.published_at);
  assert.deepEqual(input, [duplicate, original, other]);
});

test("publication windows exclude future and unknown dates and never use ingestion time", () => {
  const recent = post();
  const yesterday = post({ id: 2, guid: "two", url: "https://x.com/b/status/2", published_at: "2026-09-19T12:00:00Z" });
  const older = post({ id: 3, guid: "three", url: "https://x.com/c/status/3", published_at: "2026-09-14T12:00:00Z" });
  const old = post({ id: 4, guid: "four", url: "https://x.com/d/status/4", published_at: "2026-08-01T12:00:00Z" });
  const undated = post({ id: 5, guid: "five", url: "https://x.com/e/status/5", published_at: null });
  const invalid = post({ id: 6, guid: "six", url: "https://x.com/f/status/6", published_at: "not a date" });
  const future = post({ id: 7, guid: "seven", url: "https://x.com/g/status/7", published_at: "2026-09-20T13:00:00Z" });
  const input = [undated, old, invalid, future, older, yesterday, recent];
  assert.deepEqual(insights(input, "", "24h").posts.map(({ id }) => id), [1, 2]);
  assert.deepEqual(insights(input).posts.map(({ id }) => id), [1, 2, 3]);
  const all = insights(input, "", "all");
  assert.deepEqual(all.posts.map(({ id }) => id), [1, 2, 3, 4, 5, 6]);
  assert.equal(all.undatedCount, 2);
  assert.equal(all.latestPublishedAt, recent.published_at);
  assert.equal(insights([undated], "", "all").latestPublishedAt, null);
});

test("stale, failed, pending, fallback and malformed analyses cannot supply insights or entity matches", () => {
  for (const untrusted of [analysis({ status: "stale" }), analysis({ status: "failed" }), analysis({ status: "pending" }), analysis({ fallback: true }), analysis({ fallback: undefined }), [analysis()], "analysis"]) {
    const item = post({ analysis: untrusted, title: "Product news", description: "The account posted a product update.", author: "Other", feed_key: "other", url: "https://x.com/other/status/1" });
    assert.equal(insights([item], "Backpack").posts.length, 0);
    const takeaway = insights([item]).takeaways[0];
    assert.equal(takeaway.kind, "excerpt");
    assert.equal(takeaway.text, item.description);
    assert.equal(takeaway.whyItMatters, null);
    assert.equal(takeaway.watchNext, null);
  }
});

test("matches complete words, normalized phrases, canonical handles and valid entities", () => {
  const base = post({ author: "Someone", title: "A product update", description: "A product update", feed_key: "other", url: "https://x.com/someone/status/1", guid: "one" });
  const byText = { ...base, description: "BACKPACK has an update." };
  const byHandle = { ...base, id: 2, url: "https://x.com/Backpack/status/2" };
  const byEntity = { ...base, id: 3, url: "https://x.com/someone/status/3", analysis: analysis() };
  const substring = { ...base, id: 4, url: "https://x.com/hiking/status/4", description: "We went backpacking with BackpackPro." };
  assert.deepEqual(insights([byText, byHandle, byEntity, substring], " Backpack ").posts.map(({ id }) => id).sort(), [1, 2, 3]);
  assert.equal(insights([byHandle], "@BACKPACK").posts.length, 1);
  assert.equal(insights([{ ...base, url: "https://x.com/Backpack/status/1/photo/1" }], "@Backpack").posts.length, 1);
  assert.equal(insights([{ ...base, description: "The Backpack   Exchange launched." }], "backpack exchange").posts.length, 1);
  assert.equal(insights([{ ...base, description: "The Backpack Exchangeable voucher." }], "backpack exchange").posts.length, 0);
  assert.equal(insights([{ ...base, description: "Backpack (beta) opened." }], "backpack (beta)").posts.length, 1);
});

test("returns at most three newest distinct takeaways with source posts and attributed analysis fields", () => {
  const analyzed = post({ analysis: analysis(), published_at: "2026-09-20T11:30:00Z" });
  const copied = post({ id: 2, guid: "two", url: "https://x.com/other/status/2", description: ` ${analyzed.description.toUpperCase()} https://t.co/copy`, published_at: "2026-09-20T11:00:00Z" });
  const extras = [3, 4, 5].map((id) => post({ id, guid: `${id}`, url: `https://x.com/account${id}/status/${id}`, description: `Independent update number ${id}.`, published_at: `2026-09-20T0${9 - id}:00:00Z` }));
  const result = insights([extras[2], copied, extras[0], analyzed, extras[1]]);
  assert.deepEqual(result.takeaways.map(({ post: item }) => item.id), [1, 3, 4]);
  assert.equal(result.posts.length, 5);
  assert.equal(result.takeaways[0].post, analyzed);
  assert.equal(result.takeaways[0].kind, "analysis");
  assert.equal(result.takeaways[0].whyItMatters, "Availability may expand access for users.");
  assert.equal(result.takeaways[0].watchNext, "Which regions are eligible?");
});

test("short excerpts respect sentences or words while preserving full original content", () => {
  const sentence = "Backpack says its latest product update is available to users in eligible regions.";
  const source = `${sentence} ${"Further details are expected from the account next week ".repeat(8)}`;
  const item = post({ description: source });
  const result = insights([item]);
  assert.equal(result.takeaways[0].text, sentence);
  assert.equal(result.posts[0].description, source);
  const longSentence = insights([post({ description: "Backpack update ".repeat(30) })]).takeaways[0].text;
  assert.ok(longSentence.length <= 260);
  assert.ok(longSentence.endsWith("…"));
});

test("source excerpts surface a matching subject even late in a long post and decode entities safely", () => {
  const source = `${"Other market updates are discussed here. ".repeat(10)}Backpack says its new feature is available. More details soon.`;
  const result = insights([post({ description: source })], "Backpack");
  assert.ok(result.takeaways[0].text.includes("Backpack says its new feature is available."));
  assert.ok(result.takeaways[0].text.length <= 260);
  const encoded = post({ title: "Update", description: "&#66;ackpack &amp; Solana &#x1f680; &#99999999;" });
  assert.equal(insights([encoded], "backpack").takeaways[0].text, "Backpack & Solana 🚀 �");
});

test("suggestions use only validated X entities and fall back to real account handles", () => {
  const items = [
    post({ analysis: analysis({ entities: ["Backpack", "Solana", "Backpack", 42] }) }),
    post({ id: 2, guid: "two", url: "https://x.com/another/status/2", analysis: analysis({ entities: ["backpack", "Bitcoin", "Ethereum", "USDC", "Jito"] }) }),
    post({ id: 3, guid: "three", url: "https://x.com/third/status/3", analysis: analysis({ status: "stale", entities: ["Stale Entity"] }) }),
    post({ id: 4, feed_key: "news", url: "https://example.com/a", analysis: analysis({ entities: ["News Only"] }) }),
  ];
  const suggestions = suggestedXSignalSubjects(items);
  assert.equal(suggestions[0], "Backpack");
  assert.equal(suggestions.length, 5);
  assert.ok(!suggestions.includes("Stale Entity"));
  assert.ok(!suggestions.includes("News Only"));
  assert.deepEqual(suggestedXSignalSubjects([post(), post({ analysis: analysis() })]), ["Backpack"]);
  assert.deepEqual(suggestedXSignalSubjects([post(), post({ id: 2, url: "https://twitter.com/BACKPACK/status/2" }), post({ id: 3, url: "https://x.com/armaniferrante/status/3" })]), ["@backpack", "@armaniferrante"]);
  assert.deepEqual(insights([]), { posts: [], takeaways: [], accountCount: 0, latestPublishedAt: null, undatedCount: 0 });
});
