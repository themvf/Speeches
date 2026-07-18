import assert from "node:assert/strict";
import test from "node:test";

import { isSameStoredRssArticle, rssArticleIdentity } from "./rss-article-identity.ts";
import {
  EXISTING_RSS_SOURCE_PROMOTION_KEYS,
  EXISTING_RSS_SOURCE_PROMOTIONS,
  MARKET_COMMENTARY_RSS_SOURCE_KEYS,
  MARKET_COMMENTARY_RSS_SOURCES,
  RETIRED_RSS_FEED_KEYS,
} from "./rss-source-catalog.ts";
import {
  DEFAULT_RSS_FEEDS,
  MAX_RSS_FEED_BYTES,
  rssDiscoveryFallbackUrl,
} from "./server/rss-fetcher.ts";

const EXPECTED_EXISTING_KEYS = [
  "harvard_corp_gov_forum",
  "cls_blue_sky_blog",
  "the_corporate_counsel_net",
  "rss_nytimes_com_services_xml_rss_nyt_economy_xml",
  "google_news_senate_banking_committee",
  "google_news_senate_finance_committee",
  "google_news_senate_agriculture_committee",
  "google_news_senate_judiciary_committee",
  "google_news_senate_hsgac",
  "google_news_senate_commerce_committee",
  "american_banker",
  "search_cnbc_com_rs_search_combinedcms_view_xml",
  "rss_nytimes_com_services_xml_rss_nyt_business_xml",
  "rss_nytimes_com_services_xml_rss_nyt_dealbook_xml",
  "www_centralbanking_com_feeds_rss_category_central_banks_fina",
] as const;

test("promotes exactly fifteen existing source keys into maintained configuration", () => {
  assert.equal(EXISTING_RSS_SOURCE_PROMOTION_KEYS.length, 15);
  assert.deepEqual([...EXISTING_RSS_SOURCE_PROMOTION_KEYS], EXPECTED_EXISTING_KEYS);
  assert.equal(new Set(EXISTING_RSS_SOURCE_PROMOTION_KEYS).size, 15);
});

test("uses valid unique feed URLs and cost-conscious refresh cadences", () => {
  const definitions = Object.values(EXISTING_RSS_SOURCE_PROMOTIONS);
  assert.equal(new Set(definitions.map((feed) => feed.feedUrl)).size, definitions.length);
  for (const feed of definitions) {
    assert.doesNotThrow(() => new URL(feed.feedUrl));
    assert.ok(feed.refreshIntervalMinutes >= 30);
  }
});

test("preserves the same upstream GUID when different sources cover it", () => {
  const firstPublisher = rssArticleIdentity("publisher_one", "shared-story-guid");
  const secondPublisher = rssArticleIdentity("publisher_two", "shared-story-guid");
  assert.equal(isSameStoredRssArticle(firstPublisher, secondPublisher), false);
  assert.equal(isSameStoredRssArticle(firstPublisher, rssArticleIdentity("publisher_one", "shared-story-guid")), true);
});

const EXPECTED_MARKET_COMMENTARY_KEYS = [
  "fox_business_markets",
  "bbc_business",
  "seeking_alpha_all_news",
  "investing_com_news",
  "investing_com_stock_markets",
  "investing_com_market_overview",
  "abnormal_returns",
  "the_bear_cave",
  "klement_on_investing",
  "angry_bear_blog",
  "zerohedge",
] as const;

test("registers the SEC-55 market commentary sources without duplicating MarketWatch", () => {
  assert.deepEqual([...MARKET_COMMENTARY_RSS_SOURCE_KEYS], EXPECTED_MARKET_COMMENTARY_KEYS);
  assert.equal(new Set(MARKET_COMMENTARY_RSS_SOURCE_KEYS).size, EXPECTED_MARKET_COMMENTARY_KEYS.length);
  assert.equal("mw_top_stories" in MARKET_COMMENTARY_RSS_SOURCES, false);
  assert.equal(
    Object.values(MARKET_COMMENTARY_RSS_SOURCES).some((feed) => feed.feedUrl.includes("capitalflowresearch.com")),
    false,
  );
  assert.equal(MARKET_COMMENTARY_RSS_SOURCES.angry_bear_blog.proxyFallback, "webshare");
});

test("uses unique valid URLs and bounded refresh cadences for market commentary", () => {
  const definitions = Object.values(MARKET_COMMENTARY_RSS_SOURCES);
  assert.equal(new Set(definitions.map((feed) => feed.feedUrl)).size, definitions.length);
  for (const feed of definitions) {
    assert.doesNotThrow(() => new URL(feed.feedUrl));
    assert.ok(feed.refreshIntervalMinutes >= 30);
    assert.ok(feed.refreshIntervalMinutes <= 180);
  }
});

test("keeps only PR Newswire Financial Services active", () => {
  const activePrNewswireKeys = Object.keys(DEFAULT_RSS_FEEDS).filter((key) => key.startsWith("prnewswire_"));
  assert.deepEqual(activePrNewswireKeys, ["prnewswire_financial_services"]);
  assert.deepEqual([...RETIRED_RSS_FEED_KEYS], [
    "prnewswire_all",
    "prnewswire_consumer_technology",
    "prnewswire_policy_public_interest",
  ]);
});

test("uses bounded discovery fallbacks for publishers that block server RSS fetches", () => {
  const fallbackSources = [
    "https://www.spglobal.com/spdji/en/rss/rss-details/?rssFeedName=corporate-news",
    "https://ir.thomsonreuters.com/rss/news-releases.xml?items=15",
    "https://www.tripwire.com/state-of-security/feed",
    "https://www.akamai.com/blog/rss.xml",
  ];
  for (const source of fallbackSources) {
    const fallback = new URL(rssDiscoveryFallbackUrl(source));
    assert.equal(fallback.hostname, "news.google.com");
    assert.equal(fallback.pathname, "/rss/search");
  }
  assert.equal(rssDiscoveryFallbackUrl("https://example.com/feed"), "");
  assert.ok(MAX_RSS_FEED_BYTES >= 3_000_000);
  assert.ok(MAX_RSS_FEED_BYTES <= 5_000_000);
});
