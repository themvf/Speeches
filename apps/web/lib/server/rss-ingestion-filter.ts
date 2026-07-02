import { getMatchingTopics, normalizeTopicRules, type TopicRuleInput } from "@/lib/intel-topic-matching";
import type { RssArticle } from "@/lib/server/rss-fetcher";

const KEYWORD_FILTERED_FEED_PREFIXES = ["prnewswire_"];

export type RssIngestionFilterResult = {
  articles: RssArticle[];
  fetched: number;
  matched: number;
  filtered: number;
};

export function shouldKeywordFilterFeed(feedKey: string): boolean {
  const key = String(feedKey || "").trim().toLowerCase();
  return KEYWORD_FILTERED_FEED_PREFIXES.some((prefix) => key.startsWith(prefix));
}

export function rssFetchLimitForFeed(feedKey: string): number {
  return shouldKeywordFilterFeed(feedKey) ? 100 : 50;
}

export function filterRssArticlesForIngestion(
  feedKey: string,
  articles: RssArticle[],
  topicRules: TopicRuleInput[]
): RssIngestionFilterResult {
  if (!shouldKeywordFilterFeed(feedKey)) {
    return {
      articles,
      fetched: articles.length,
      matched: articles.length,
      filtered: 0,
    };
  }

  const rules = normalizeTopicRules(topicRules);
  if (rules.length === 0) {
    return {
      articles: [],
      fetched: articles.length,
      matched: 0,
      filtered: articles.length,
    };
  }

  const filteredArticles = articles.filter((article) => getMatchingTopics(article, rules).length > 0);
  return {
    articles: filteredArticles,
    fetched: articles.length,
    matched: filteredArticles.length,
    filtered: articles.length - filteredArticles.length,
  };
}
