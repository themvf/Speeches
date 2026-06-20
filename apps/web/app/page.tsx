import type { Metadata } from "next";
import { NewsFeedWorkspace } from "@/components/news-feed-workspace";
import { buildDocumentListItems, loadCorpusDocuments, loadEnrichmentState, selectNewsFeedDocuments } from "@/lib/server/data-store";
import { compactFeedArticles } from "@/lib/server/feed-payload";
import { getRecentArticles, getTopicRules } from "@/lib/server/neon";
import type { StoredRssArticle, StoredRssTopicRule } from "@/lib/server/neon";
import type { DocumentListItem } from "@/lib/server/types";

export const revalidate = 60;

export const metadata: Metadata = {
  title: "News Feed | Policy Research Hub",
  description: "Live regulatory news stream filtered by topic.",
};

const INITIAL_FEED_ARTICLE_LIMIT = 100;
const INITIAL_FEED_DOCUMENT_LIMIT = 80;

export default async function HomePage() {
  let initialArticles: StoredRssArticle[] = [];
  let initialTopicRules: StoredRssTopicRule[] = [];
  let initialDocuments: DocumentListItem[] = [];
  try {
    const [articles, topicRules, corpusDocs, enrichment] = await Promise.all([
      getRecentArticles({ limit: INITIAL_FEED_ARTICLE_LIMIT }),
      getTopicRules(true),
      loadCorpusDocuments(),
      loadEnrichmentState(),
    ]);
    initialArticles = compactFeedArticles(articles);
    initialTopicRules = topicRules;
    initialDocuments = selectNewsFeedDocuments(buildDocumentListItems(corpusDocs, enrichment), {
      limit: INITIAL_FEED_DOCUMENT_LIMIT,
    });
  } catch {
    // DB not yet configured or schema not created; start with empty feed.
  }

  return (
    <main className="mx-auto w-full max-w-[1800px] px-3 py-4 md:px-5">
      <NewsFeedWorkspace
        initialArticles={initialArticles}
        initialTopicRules={initialTopicRules}
        initialDocuments={initialDocuments}
      />
    </main>
  );
}
