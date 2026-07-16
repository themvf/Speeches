import assert from "node:assert/strict";
import test from "node:test";

import {
  dominantCatalyst,
  filterCompanyNewsArticles,
  isPossiblePriceCatalyst,
  isSessionNewsCacheFresh,
  sortAndFilterSectorStocks,
  summarizeLoadedSectorNews,
} from "./market-news-signals.ts";
import type { CompanyNewsArticle, MarketCompanyNewsData, SectorStock } from "./server/types.ts";

const NOW = Date.parse("2026-07-16T12:00:00.000Z");

function article(overrides: Partial<CompanyNewsArticle> = {}): CompanyNewsArticle {
  return {
    title: "Apple raises earnings guidance",
    publisher: "Reuters",
    url: "https://example.com/apple",
    snippet: "Apple update",
    publishedAt: "2026-07-16T10:00:00.000Z",
    relevanceScore: 90,
    catalyst: "Earnings",
    sourceTier: "Premium",
    isLikelyPaywalled: false,
    isPressRelease: false,
    clusterSize: 1,
    ...overrides,
  };
}

function companyNews(symbol: string, articles: CompanyNewsArticle[], availableArticleCount = articles.length): MarketCompanyNewsData {
  return {
    symbol,
    companyName: symbol,
    articles,
    provider: "Google News RSS",
    searchedDays: 7,
    generatedAt: "2026-07-16T12:00:00.000Z",
    availableArticleCount,
    hasMore: false,
  };
}

const stocks: SectorStock[] = [
  { symbol: "AAPL", name: "Apple", price: 210, pct: 2.4, change: 4.9, up: true },
  { symbol: "MSFT", name: "Microsoft", price: 500, pct: -1.2, change: -6, up: false },
  { symbol: "NVDA", name: "Nvidia", price: 170, pct: 0.2, change: 0.3, up: true },
];

test("summarizes only company news already loaded in the session", () => {
  const news = {
    AAPL: companyNews("AAPL", [article(), article({ catalyst: "Product" })], 5),
    MSFT: companyNews("MSFT", [article({ catalyst: "Earnings" })], 3),
  };
  assert.deepEqual(summarizeLoadedSectorNews(stocks, news), {
    loadedCompanies: 2,
    totalArticles: 8,
    dominantCatalyst: "Earnings",
    priceReaction: "Mixed",
    gainers: 1,
    losers: 1,
  });
  assert.equal(dominantCatalyst(news.AAPL), "Earnings");
});

test("filters article quality without changing cached results", () => {
  const articles = [
    article(),
    article({ publisher: "MarketWatch", sourceTier: "Established", url: "https://example.com/two" }),
    article({ publisher: "Business Wire", sourceTier: "Other", isPressRelease: true, url: "https://example.com/three" }),
  ];
  const filtered = filterCompanyNewsArticles(articles, { hidePressReleases: true, minSourceTier: "Established" });
  assert.deepEqual(filtered.map((item) => item.publisher), ["Reuters", "MarketWatch"]);
  assert.equal(articles.length, 3);
});

test("sorts and filters companies using loaded deterministic signals", () => {
  const news = {
    AAPL: companyNews("AAPL", [article({ relevanceScore: 75 })]),
    MSFT: companyNews("MSFT", [article({ catalyst: "Regulation", relevanceScore: 96, publishedAt: "2026-07-16T11:00:00.000Z" })]),
  };
  assert.deepEqual(sortAndFilterSectorStocks(stocks, news, {
    sort: "relevance", price: "all", catalyst: "all", loadedOnly: true,
  }).map((stock) => stock.symbol), ["MSFT", "AAPL"]);
  assert.deepEqual(sortAndFilterSectorStocks(stocks, news, {
    sort: "move", price: "losers", catalyst: "Regulation", loadedOnly: false,
  }).map((stock) => stock.symbol), ["MSFT"]);
});

test("labels only recent categorized news beside a material price move as a possible catalyst", () => {
  assert.equal(isPossiblePriceCatalyst(article(), 2.4, NOW), true);
  assert.equal(isPossiblePriceCatalyst(article(), 0.5, NOW), false);
  assert.equal(isPossiblePriceCatalyst(article({ catalyst: null }), 2.4, NOW), false);
  assert.equal(isPossiblePriceCatalyst(article({ publishedAt: "2026-07-12T10:00:00.000Z" }), 2.4, NOW), false);
});

test("expires the browser-session cache after fifteen minutes", () => {
  assert.equal(isSessionNewsCacheFresh(NOW - 14 * 60_000, NOW), true);
  assert.equal(isSessionNewsCacheFresh(NOW - 15 * 60_000, NOW), false);
  assert.equal(isSessionNewsCacheFresh(NOW + 1, NOW), false);
});
