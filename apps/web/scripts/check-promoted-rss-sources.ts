import { EXISTING_RSS_SOURCE_PROMOTIONS } from "../lib/rss-source-catalog.ts";
import { fetchRssFeed } from "../lib/server/rss-fetcher.ts";

const results = await Promise.all(Object.entries(EXISTING_RSS_SOURCE_PROMOTIONS).map(async ([feedKey, feed]) => {
  try {
    const articles = await fetchRssFeed(feed.feedUrl, 3, 20_000);
    return { feedKey, label: feed.label, articleCount: articles.length, error: "" };
  } catch (error) {
    return { feedKey, label: feed.label, articleCount: 0, error: error instanceof Error ? error.message : String(error) };
  }
}));

for (const result of results) {
  console.log(`${result.feedKey}: ${result.articleCount} parsed${result.error ? ` (${result.error})` : ""}`);
}

const failures = results.filter((result) => result.articleCount === 0);
if (failures.length > 0) {
  throw new Error(`Promoted RSS validation failed for: ${failures.map((result) => result.label).join(", ")}`);
}
