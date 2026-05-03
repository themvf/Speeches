import type { Metadata } from "next";
import { IntelBetaDashboard } from "@/components/intelbeta-dashboard";
import { getRecentArticles, getTopicRules } from "@/lib/server/neon";
import type { StoredRssArticle, StoredRssTopicRule } from "@/lib/server/neon";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "News Feed | Policy Research Hub",
  description: "Live regulatory news stream filtered by topic.",
};

export default async function HomePage() {
  let initialArticles: StoredRssArticle[] = [];
  let initialTopicRules: StoredRssTopicRule[] = [];
  try {
    [initialArticles, initialTopicRules] = await Promise.all([
      getRecentArticles({ limit: 400 }),
      getTopicRules(true),
    ]);
  } catch {
    // DB not yet configured or schema not created; start with empty feed.
  }

  return (
    <main className="mx-auto w-full max-w-[1800px] px-3 py-4 md:px-5">
      <IntelBetaDashboard initialArticles={initialArticles} initialTopicRules={initialTopicRules} />
    </main>
  );
}
