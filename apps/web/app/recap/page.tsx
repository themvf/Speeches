import type { Metadata } from "next";
import { getRecapSettings, getTopicRules, getTodaysRecap } from "@/lib/server/neon";
import { RecapDashboard } from "@/components/recap-dashboard";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Daily Recap | Policy Research Hub",
  description: "LLM-generated daily summary of regulatory news by topic.",
};

export default async function RecapPage() {
  let selectedKeys: string[] = [];
  let topicRules: Awaited<ReturnType<typeof getTopicRules>> = [];
  let recap: Awaited<ReturnType<typeof getTodaysRecap>> = [];

  try {
    [selectedKeys, topicRules, recap] = await Promise.all([
      getRecapSettings(),
      getTopicRules(true),
      getTodaysRecap(),
    ]);
  } catch {
    // DB not yet configured; render empty state
  }

  return (
    <main className="mx-auto w-full max-w-3xl px-4 py-6 md:px-8">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-[color:var(--ink)]">Daily Recap</h1>
        <p className="mt-1 text-sm text-[color:var(--ink-faint)]">
          AI-generated summary of the past 24 hours of regulatory news for your chosen topics.
        </p>
      </div>
      <RecapDashboard
        initialTopicRules={topicRules}
        initialSelectedKeys={selectedKeys}
        initialRecap={recap}
      />
    </main>
  );
}
