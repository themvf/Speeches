import type { Metadata } from "next";
import { NewsFeedWorkspace } from "@/components/news-feed-workspace";
import { buildDocumentListItems, loadCorpusDocuments, loadEnrichmentState, parseComparableDate } from "@/lib/server/data-store";
import { getRecentArticles, getTopicRules } from "@/lib/server/neon";
import type { StoredRssArticle, StoredRssTopicRule } from "@/lib/server/neon";
import type { DocumentListItem } from "@/lib/server/types";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "News Feed | Policy Research Hub",
  description: "Live regulatory news stream filtered by topic.",
};

export default async function HomePage() {
  let initialArticles: StoredRssArticle[] = [];
  let initialTopicRules: StoredRssTopicRule[] = [];
  let initialDocuments: DocumentListItem[] = [];
  try {
    const [articles, topicRules, corpusDocs, enrichment] = await Promise.all([
      getRecentArticles({ limit: 400 }),
      getTopicRules(true),
      loadCorpusDocuments(),
      loadEnrichmentState(),
    ]);
    initialArticles = articles;
    initialTopicRules = topicRules;
    initialDocuments = buildDocumentListItems(corpusDocs, enrichment)
      .filter((item) => parseComparableDate(item.published_at || item.date) > 0)
      .sort((a, b) => parseComparableDate(b.published_at || b.date) - parseComparableDate(a.published_at || a.date))
      .slice(0, 250);
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
