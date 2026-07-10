import assert from "node:assert/strict";
import test from "node:test";

import {
  compileKeywords,
  getMatchingTopics,
  getTopicMatches,
  matchingKeywordsForArticle,
  normalizeTopicRules,
} from "./intel-topic-matching.ts";

test("word-boundary matching does not over-match short acronyms inside longer words", () => {
  const matchers = compileKeywords(["ai", "sec"]);

  assert.deepEqual(
    matchingKeywordsForArticle(matchers, { title: "Quarterly email newsletter", description: "" }),
    []
  );
  assert.deepEqual(
    matchingKeywordsForArticle(matchers, { title: "Second quarter earnings", description: "" }),
    []
  );
  assert.deepEqual(
    matchingKeywordsForArticle(matchers, { title: "New AI governance rules", description: "" }),
    ["ai"]
  );
  assert.deepEqual(
    matchingKeywordsForArticle(matchers, { title: "The SEC proposed a rule", description: "" }),
    ["sec"]
  );
});

test("multi-word keywords match across flexible separators", () => {
  const matchers = compileKeywords(["money laundering", "market structure"]);

  assert.deepEqual(
    matchingKeywordsForArticle(matchers, { title: "Anti money-laundering program", description: "" }),
    ["money laundering"]
  );
  assert.deepEqual(
    matchingKeywordsForArticle(matchers, { title: "", description: "Equity market_structure reform" }),
    ["market structure"]
  );
});

test("matchingKeywordsForArticle returns every matched keyword, not just the best", () => {
  const matchers = compileKeywords(["crypto", "bitcoin", "fraud"]);
  const matched = matchingKeywordsForArticle(matchers, {
    title: "Bitcoin fraud scheme tied to crypto exchange",
    description: "",
  });

  assert.deepEqual(new Set(matched), new Set(["crypto", "bitcoin", "fraud"]));
});

test("compileKeywords + matchingKeywordsForArticle agrees with getTopicMatches for a single rule", () => {
  const rules = normalizeTopicRules([
    { topic_key: "AI_TECH", label: "AI & Tech", keywords: "ai, machine learning", active: true, sort_order: 50 },
  ]);
  const article = { title: "New AI governance framework proposed", description: "" };

  const topicMatches = getTopicMatches(article, rules);
  const matchers = compileKeywords(["ai", "machine learning"]);
  const keywordMatches = matchingKeywordsForArticle(matchers, article);

  assert.equal(topicMatches.length > 0, keywordMatches.length > 0);
  assert.deepEqual(getMatchingTopics(article, rules).map((r) => r.topic_key), ["AI_TECH"]);
});
