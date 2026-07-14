import assert from "node:assert/strict";
import test from "node:test";

import {
  compileKeywords,
  filterTopicMappedArticles,
  filterCanonicalTopicMappedDocuments,
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

test("feed topic gate keeps current matches and rejects unmapped or model-only articles", () => {
  const rules = normalizeTopicRules([
    { topic_key: "CREDIT_MARKETS", label: "Credit Markets", keywords: "treasury yield, bond market", active: true, sort_order: 10 },
    { topic_key: "ECONOMIC_GROWTH", label: "Economic Growth", keywords: "fiscal policy, inflation", active: true, sort_order: 20 },
    { topic_key: "RETIRED", label: "Retired", keywords: "currencies", active: false, sort_order: 30 },
  ]);
  const articles = [
    { id: 1, title: "Treasury yield falls after inflation report", description: "" },
    { id: 2, title: "Policy briefing", description: "The fiscal-policy outlook changed." },
    { id: 3, title: "Asian currencies consolidate", description: "Risk-off sentiment", topics: ["Credit Markets"] },
    { id: 4, title: "Second-quarter update", description: "", analysis: { topics: ["Economic Growth"] } },
  ];

  const mapped = filterTopicMappedArticles(articles, rules);

  assert.deepEqual(mapped.map((article) => article.id), [1, 2]);
  assert.deepEqual(mapped[0].topics, ["Credit Markets", "Economic Growth"]);
  assert.deepEqual(mapped[1].topics, ["Economic Growth"]);
});

test("feed topic gate fails closed when no active taxonomy is available", () => {
  assert.deepEqual(
    filterTopicMappedArticles([{ title: "SEC rulemaking", description: "" }], []),
    []
  );
});

test("document feed gate accepts only deterministic or active canonical assignments", () => {
  const rules = normalizeTopicRules([
    { topic_key: "CREDIT_MARKETS", label: "Credit Markets", keywords: "treasury yield", active: true, sort_order: 10 },
    { topic_key: "AI_TECH", label: "AI & Tech", keywords: "artificial intelligence", active: true, sort_order: 20 },
  ]);
  const documents = [
    { title: "Treasury yield rises", description: "", topics: [] },
    { title: "Quarterly letter", description: "", topics: ["AI_TECH"] },
    { title: "Company picnic", description: "", topics: ["Lifestyle"] },
  ];

  const mapped = filterCanonicalTopicMappedDocuments(documents, rules);

  assert.deepEqual(mapped.map((item) => item.title), ["Treasury yield rises", "Quarterly letter"]);
  assert.deepEqual(mapped[0].topics, ["Credit Markets"]);
  assert.deepEqual(mapped[1].topics, ["AI & Tech"]);
  assert.deepEqual(filterCanonicalTopicMappedDocuments(documents, []), []);
});
