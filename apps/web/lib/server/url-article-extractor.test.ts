// Tests for the URL article extractor (SEC-35).
// Run with: npm run test:url-extractor (node --test).
import assert from "node:assert/strict";
import test from "node:test";

import { extractArticle } from "./url-article-extractor.ts";

const SAMPLE = `<!doctype html>
<html><head>
  <title>Fallback Title</title>
  <meta property="og:title" content="OG Article Title" />
  <meta name="description" content="A concise summary of the article." />
  <meta property="article:published_time" content="2026-07-16T09:00:00Z" />
  <script type="application/ld+json">
  {"@type":"NewsArticle","headline":"JSON-LD Headline","author":{"name":"Jane Reporter"},"datePublished":"2026-07-15T12:00:00Z"}
  </script>
  <style>.x{color:red}</style>
</head>
<body>
  <nav>Home About <span class="sr-only">skip to content</span></nav>
  <article>
    <h1>The Real Headline On Page</h1>
    <p>This is the first substantial paragraph of the article body with enough length.</p>
    <p style="display:none">Hidden paywall bait text that should not count as visible.</p>
    <p>Second real paragraph, also well over the minimum length threshold here.</p>
    <button>Subscribe</button>
    <ul><li>A list item that is long enough to be kept in the output.</li></ul>
  </article>
  <footer>Copyright 2026 do not include this footer text please</footer>
</body></html>`;

test("extractArticle prefers JSON-LD headline, byline, and date", () => {
  const a = extractArticle(SAMPLE, "https://example.com/x");
  assert.equal(a.title, "JSON-LD Headline");
  assert.equal(a.author, "Jane Reporter");
  assert.equal(a.publishedAt, new Date("2026-07-15T12:00:00Z").toISOString());
  assert.equal(a.jsonLdCount, 1);
});

test("extractArticle pulls body text and drops junk/hidden/nav/footer", () => {
  const a = extractArticle(SAMPLE, "https://example.com/x");
  assert.match(a.readableText, /first substantial paragraph/);
  assert.match(a.readableText, /Second real paragraph/);
  assert.match(a.readableText, /list item that is long enough/);
  // Removed content:
  assert.doesNotMatch(a.readableText, /Subscribe/);
  assert.doesNotMatch(a.readableText, /Copyright 2026/);
  assert.doesNotMatch(a.readableText, /Hidden paywall bait/);
  assert.ok(a.wordCount > 15);
});

test("extractArticle detects hidden nodes and computes visibility", () => {
  const a = extractArticle(SAMPLE, "https://example.com/x");
  // The display:none paragraph and the sr-only span.
  assert.ok(a.hiddenNodes.some((n) => n.hiddenBy.includes("display:none")));
  assert.ok(a.hiddenNodes.some((n) => n.hiddenBy.includes("sr-only")));
  assert.ok(a.textVisibility.hiddenWords > 0);
  assert.ok(a.textVisibility.hiddenRatio > 0 && a.textVisibility.hiddenRatio < 1);
});

test("extractArticle uses meta description as the feed snippet", () => {
  const a = extractArticle(SAMPLE, "https://example.com/x");
  assert.equal(a.description, "A concise summary of the article.");
});

test("extractArticle falls back to <title> and body lead without metadata", () => {
  const bare = `<html><head><title>Bare Title</title></head><body><main>
    <p>Only paragraph here, long enough to survive the minimum length filter.</p>
  </main></body></html>`;
  const a = extractArticle(bare, "https://example.com/bare");
  assert.equal(a.title, "Bare Title");
  assert.equal(a.author, "");
  assert.equal(a.publishedAt, null);
  assert.match(a.description, /Only paragraph here/);
});
