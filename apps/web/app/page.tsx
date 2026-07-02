import type { Metadata } from "next";
import { NewsFeedWorkspace } from "@/components/news-feed-workspace";
import { compactFeedArticles } from "@/lib/server/feed-payload";
import { getRecentArticles, getTopicRules } from "@/lib/server/neon";
import type { StoredRssArticle, StoredRssTopicRule } from "@/lib/server/neon";
import { isAllowedRssArticleForIngestion } from "@/lib/server/rss-ingestion-filter";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "News Feed | Policy Research Hub",
  description: "Live regulatory news stream filtered by topic.",
};

const INITIAL_FEED_ARTICLE_LIMIT = 500;

export default async function HomePage() {
  let initialArticles: StoredRssArticle[] = [];
  let initialTopicRules: StoredRssTopicRule[] = [];
  try {
    const [articles, topicRules] = await Promise.all([
      getRecentArticles({ limit: INITIAL_FEED_ARTICLE_LIMIT }),
      getTopicRules(true),
    ]);
    initialArticles = compactFeedArticles(articles.filter((article) => isAllowedRssArticleForIngestion(article.feed_key, {
      guid: article.guid,
      title: article.title,
      url: article.url,
      description: article.description,
      author: article.author,
      publishedAt: article.published_at ? new Date(article.published_at) : null,
    }, topicRules)));
    initialTopicRules = topicRules;
  } catch {
    // DB not yet configured or schema not created; start with empty feed.
  }

  return (
    <main className="mx-auto w-full max-w-[1800px] px-3 py-4 md:px-5">
      <NewsFeedWorkspace
        initialArticles={initialArticles}
        initialTopicRules={initialTopicRules}
        initialDocuments={[]}
      />
    </main>
  );
}
