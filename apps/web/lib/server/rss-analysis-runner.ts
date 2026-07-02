import { getMatchingTopics, normalizeTopicRules } from "@/lib/intel-topic-matching";
import { normalizeText } from "@/lib/server/api-utils";
import { generateFeedAnalysis, type FeedAnalysisInput } from "@/lib/server/feed-analysis";
import {
  getRssArticlesNeedingAnalysis,
  getTopicRules,
  saveRssArticleAnalysis,
  saveRssArticleAnalysisFailure,
  type StoredRssArticle,
} from "@/lib/server/neon";

function articleDate(article: StoredRssArticle): string {
  return normalizeText(article.published_at || article.fetched_at);
}

function inputForArticle(article: StoredRssArticle, topics: string[]): FeedAnalysisInput {
  return {
    title: normalizeText(article.title).slice(0, 400),
    description: normalizeText(article.description).slice(0, 8000),
    url: normalizeText(article.url).slice(0, 1000),
    source: normalizeText(article.feed_key).slice(0, 200),
    author: normalizeText(article.author).slice(0, 200),
    published_at: articleDate(article).slice(0, 80),
    tone_label: normalizeText(article.tone_label).slice(0, 40),
    topics,
    item_type: "article",
  };
}

export async function analyzeMissingRssArticles(limit = 5): Promise<{
  selected_count: number;
  saved_count: number;
  failed_count: number;
  failed: Array<{ article_id: number; title: string; error: string }>;
}> {
  const selected = await getRssArticlesNeedingAnalysis(Math.max(1, Math.min(50, limit)));
  if (selected.length === 0) {
    return { selected_count: 0, saved_count: 0, failed_count: 0, failed: [] };
  }

  const rules = normalizeTopicRules(await getTopicRules(true));
  const results = await Promise.all(selected.map(async (article) => {
    const topics = getMatchingTopics(article, rules).map((topic) => topic.label);
    try {
      const generated = await generateFeedAnalysis(inputForArticle(article, topics));
      await saveRssArticleAnalysis(article, generated, topics);
      return { saved: true as const };
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      await saveRssArticleAnalysisFailure(article, message);
      return { saved: false as const, failed: { article_id: article.id, title: article.title, error: message } };
    }
  }));

  const savedCount = results.filter((result) => result.saved).length;
  const failed = results
    .filter((result): result is { saved: false; failed: { article_id: number; title: string; error: string } } => !result.saved)
    .map((result) => result.failed);

  return {
    selected_count: selected.length,
    saved_count: savedCount,
    failed_count: failed.length,
    failed,
  };
}
