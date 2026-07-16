import assert from "node:assert/strict";
import test from "node:test";

import {
  buildCompanyNewsRssUrl,
  classifyCompanyNewsCatalyst,
  companyNewsResultWindow,
  manualRefreshDecision,
  normalizeCompanyNewsArticles,
} from "./market-company-news.ts";
import { findMarketSectorCompany, MARKET_SECTOR_COMPANIES } from "./market-sector-companies.ts";
import type { RssArticle } from "./rss-fetcher.ts";

const NOW = new Date("2026-07-16T12:00:00.000Z");

function article(overrides: Partial<RssArticle> = {}): RssArticle {
  return {
    guid: "article-1",
    title: "Apple announces new U.S. investment - Reuters",
    url: "https://news.google.com/rss/articles/apple-investment",
    description: "Apple said the company will expand its U.S. operations.",
    author: "",
    publishedAt: new Date("2026-07-16T10:00:00.000Z"),
    ...overrides,
  };
}

test("keeps exactly ten configured companies in every market sector", () => {
  assert.equal(Object.keys(MARKET_SECTOR_COMPANIES).length, 11);
  for (const companies of Object.values(MARKET_SECTOR_COMPANIES)) {
    assert.equal(companies.length, 10);
  }
  assert.equal(findMarketSectorCompany("googl")?.name, "Alphabet");
  assert.equal(findMarketSectorCompany("UNKNOWN"), null);
});

test("builds a free Google News RSS query for the U.S. English edition", () => {
  const company = findMarketSectorCompany("GOOGL");
  assert.ok(company);
  const url = new URL(buildCompanyNewsRssUrl(company, 7));

  assert.equal(url.origin + url.pathname, "https://news.google.com/rss/search");
  assert.equal(url.searchParams.get("hl"), "en-US");
  assert.equal(url.searchParams.get("gl"), "US");
  assert.equal(url.searchParams.get("ceid"), "US:en");
  assert.match(url.searchParams.get("q") ?? "", /"Alphabet" OR "Google"/);
  assert.match(url.searchParams.get("q") ?? "", /when:7d/);
});

test("normalizes, sorts, and deduplicates relevant company news", () => {
  const company = findMarketSectorCompany("AAPL");
  assert.ok(company);
  const results = normalizeCompanyNewsArticles([
    article(),
    article({
      guid: "article-2",
      title: "Apple earnings lift shares - CNBC",
      url: "https://news.google.com/rss/articles/apple-earnings",
      description: "Apple earnings beat expectations.",
      publishedAt: new Date("2026-07-16T11:00:00.000Z"),
    }),
    article({
      guid: "duplicate-title",
      url: "https://news.google.com/rss/articles/apple-investment-copy",
    }),
    article({
      guid: "irrelevant",
      title: "Microsoft announces new cloud region - Reuters",
      url: "https://news.google.com/rss/articles/microsoft-cloud",
      description: "Microsoft expanded Azure capacity.",
    }),
    article({
      guid: "old",
      title: "Apple opens an older store - AP",
      url: "https://news.google.com/rss/articles/apple-old",
      publishedAt: new Date("2026-05-01T10:00:00.000Z"),
    }),
  ], company, NOW);

  assert.equal(results.length, 2);
  assert.equal(results[0].title, "Apple earnings lift shares");
  assert.equal(results[0].publisher, "CNBC");
  assert.equal(results[1].publisher, "Reuters");
});

test("accepts canonical aliases and filters non-English Google News results", () => {
  const company = findMarketSectorCompany("PG");
  assert.ok(company);
  const results = normalizeCompanyNewsArticles([
    article({
      guid: "alias",
      title: "Procter & Gamble raises its outlook - Bloomberg",
      url: "https://news.google.com/rss/articles/pg-outlook",
      description: "Procter & Gamble increased its annual forecast.",
    }),
    article({
      guid: "spanish",
      title: "Procter & Gamble anuncia nuevas inversiones - Ejemplo",
      url: "https://news.google.com/rss/articles/pg-spanish",
      description: "La empresa anuncia nuevas inversiones para los consumidores y los mercados.",
    }),
  ], company, NOW);

  assert.equal(results.length, 1);
  assert.equal(results[0].title, "Procter & Gamble raises its outlook");
});

test("ranks a strong company headline above a newer incidental mention", () => {
  const company = findMarketSectorCompany("AAPL");
  assert.ok(company);
  const results = normalizeCompanyNewsArticles([
    article({
      guid: "incidental",
      title: "Alibaba shares rise on AI partnership - CNBC",
      url: "https://news.google.com/rss/articles/alibaba-partnership",
      description: "The agreement will integrate models into Apple Intelligence.",
      publishedAt: new Date("2026-07-16T11:30:00.000Z"),
    }),
    article({
      guid: "direct",
      title: "Apple expands U.S. manufacturing investment - Reuters",
      url: "https://news.google.com/rss/articles/apple-manufacturing",
      description: "Apple announced a major expansion for the company.",
      publishedAt: new Date("2026-07-14T11:30:00.000Z"),
    }),
  ], company, NOW);

  assert.equal(results[0].title, "Apple expands U.S. manufacturing investment");
  assert.ok(results[0].relevanceScore > results[1].relevanceScore);
});

test("assigns deterministic catalyst labels without model inference", () => {
  assert.equal(classifyCompanyNewsCatalyst("Company raises quarterly earnings guidance"), "Earnings");
  assert.equal(classifyCompanyNewsCatalyst("Company agrees to acquisition in $4 billion deal"), "M&A");
  assert.equal(classifyCompanyNewsCatalyst("Analyst upgrades shares and lifts price target"), "Analyst Rating");
  assert.equal(classifyCompanyNewsCatalyst("Company settles shareholder lawsuit"), "Litigation");
  assert.equal(classifyCompanyNewsCatalyst("FTC opens antitrust investigation"), "Regulation");
  assert.equal(classifyCompanyNewsCatalyst("Board appoints a new CEO"), "Management");
  assert.equal(classifyCompanyNewsCatalyst("Company launches new software platform"), "Product");
  assert.equal(classifyCompanyNewsCatalyst("Company hosts annual community event"), null);
});

test("limits cached results to five or ten without another provider fetch", () => {
  const articles = Array.from({ length: 10 }, (_, index) => ({
    title: `Apple article ${index}`,
    publisher: "Reuters",
    url: `https://example.com/${index}`,
    snippet: "Apple update",
    publishedAt: new Date(NOW.getTime() - index * 60_000).toISOString(),
    relevanceScore: 90 - index,
    catalyst: null,
  }));

  const initial = companyNewsResultWindow(articles, 5);
  const expanded = companyNewsResultWindow(articles, 10);
  assert.equal(initial.articles.length, 5);
  assert.equal(initial.availableArticleCount, 10);
  assert.equal(initial.hasMore, true);
  assert.equal(expanded.articles.length, 10);
  assert.equal(expanded.hasMore, false);
});

test("allows one manual refresh per cooldown window", () => {
  assert.deepEqual(manualRefreshDecision(undefined, 120_000), { allowed: true, remainingSeconds: 60 });
  assert.deepEqual(manualRefreshDecision(100_000, 120_000), { allowed: false, remainingSeconds: 40 });
  assert.deepEqual(manualRefreshDecision(100_000, 160_000), { allowed: true, remainingSeconds: 60 });
});
