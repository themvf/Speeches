import type { Metadata } from "next";
import { IntelBetaDashboard } from "@/components/intelbeta-dashboard";
import { getRecentArticles } from "@/lib/server/neon";
import type { StoredRssArticle } from "@/lib/server/neon";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Intel Feed | Policy Research Hub",
  description: "Live WSJ & MarketWatch news stream filtered by regulatory topic.",
};

export default async function IntelBetaPage() {
  let initialArticles: StoredRssArticle[] = [];
  try {
    initialArticles = await getRecentArticles({ limit: 100 });
  } catch {
    // DB not yet configured or schema not created — start with empty feed
  }

  return (
    <main className="mx-auto w-full max-w-[1800px] px-3 py-4 md:px-5">
      <IntelBetaDashboard initialArticles={initialArticles} />
    </main>
  );
}
