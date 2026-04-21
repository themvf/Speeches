import type { Metadata } from "next";
import { IntelBetaDashboard } from "@/components/intelbeta-dashboard";
import { buildIntelligenceSignalsData } from "@/lib/server/intelligence-signals";
import { loadTrendsData } from "@/lib/server/data-store";
import type { TrendsPayload } from "@/lib/server/types";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Intel Beta | Policy Research Hub",
  description: "Beta workspace combining global trend detection with evidence-backed intelligence signals."
};

async function loadInitialTrends(): Promise<TrendsPayload> {
  try {
    return await loadTrendsData();
  } catch {
    return { version: 1, generated_at: "", trend_count: 0, trends: [] };
  }
}

export default async function IntelBetaPage() {
  const [initialSignals, initialTrends] = await Promise.all([
    Promise.resolve(buildIntelligenceSignalsData()),
    loadInitialTrends()
  ]);

  return (
    <main className="mx-auto w-full max-w-[1800px] px-3 py-4 md:px-5">
      <IntelBetaDashboard initialSignals={initialSignals} initialTrends={initialTrends} />
    </main>
  );
}
