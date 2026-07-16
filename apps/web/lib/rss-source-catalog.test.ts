import assert from "node:assert/strict";
import test from "node:test";

import { isSameStoredRssArticle, rssArticleIdentity } from "./rss-article-identity.ts";
import {
  EXISTING_RSS_SOURCE_PROMOTION_KEYS,
  EXISTING_RSS_SOURCE_PROMOTIONS,
} from "./rss-source-catalog.ts";

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
