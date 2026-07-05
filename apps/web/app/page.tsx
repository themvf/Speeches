import type { Metadata } from "next";
import { NewsFeedWorkspace } from "@/components/news-feed-workspace";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "News Feed | Policy Research Hub",
  description: "Live regulatory news stream filtered by topic.",
};

export default function HomePage() {
  return (
    <main className="mx-auto w-full max-w-[1800px] px-3 py-4 md:px-5">
      <NewsFeedWorkspace
        initialArticles={[]}
        initialTopicRules={[]}
        initialDocuments={[]}
      />
    </main>
  );
}
