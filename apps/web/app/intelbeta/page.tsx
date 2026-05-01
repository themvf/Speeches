import type { Metadata } from "next";
import { IntelBetaDashboard } from "@/components/intelbeta-dashboard";
import { getRecentArticles, getTopicRules } from "@/lib/server/neon";
import type { StoredRssArticle, StoredRssTopicRule } from "@/lib/server/neon";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Intelligence Feed | Policy Research Hub",
  description: "Live WSJ & MarketWatch news stream filtered by regulatory topic.",
};

export default async function IntelBetaPage() {
  let initialArticles: StoredRssArticle[] = [];
  let initialTopicRules: StoredRssTopicRule[] = [];
  try {
    [initialArticles, initialTopicRules] = await Promise.all([
      getRecentArticles({ limit: 200 }),
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
