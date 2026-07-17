import {
  EXISTING_RSS_SOURCE_PROMOTIONS,
  MARKET_COMMENTARY_RSS_SOURCES,
} from "../lib/rss-source-catalog.ts";
import { fetchRssFeed } from "../lib/server/rss-fetcher.ts";

const maintainedSources = {
  ...EXISTING_RSS_SOURCE_PROMOTIONS,
  ...MARKET_COMMENTARY_RSS_SOURCES,
};
const hasWebshareProxy = Boolean(
  String(process.env.WEBSHARE_PROXY_USERNAME || "").trim()
  && String(process.env.WEBSHARE_PROXY_PASSWORD || "").trim(),
);

const results = await Promise.all(Object.entries(maintainedSources).map(async ([feedKey, feed]) => {
  const proxyFallback = "proxyFallback" in feed ? feed.proxyFallback : undefined;
  try {
    const articles = await fetchRssFeed(feed.feedUrl, 3, 20_000);
    return { feedKey, label: feed.label, articleCount: articles.length, error: "", proxyFallback };
  } catch (error) {
    return { feedKey, label: feed.label, articleCount: 0, error: error instanceof Error ? error.message : String(error), proxyFallback };
  }
}));

for (const result of results) {
  const skippedProxyCheck = result.proxyFallback === "webshare" && !hasWebshareProxy && result.articleCount === 0;
  console.log(`${result.feedKey}: ${skippedProxyCheck ? "proxy verification skipped" : `${result.articleCount} parsed`}${result.error ? ` (${result.error})` : ""}`);
}

const failures = results.filter((result) =>
  result.articleCount === 0 && (result.proxyFallback !== "webshare" || hasWebshareProxy)
);
if (failures.length > 0) {
  throw new Error(`Promoted RSS validation failed for: ${failures.map((result) => result.label).join(", ")}`);
}
