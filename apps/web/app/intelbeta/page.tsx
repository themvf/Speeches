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
    <main className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8">
      <div className="mb-6">
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center rounded-full border border-[color:rgba(79,213,255,0.22)] bg-[color:rgba(79,213,255,0.1)] px-2 py-0.5 text-[10px] font-semibold uppercase text-[color:var(--accent)]">
            Beta
          </span>
          <span className="inline-flex items-center rounded-full border border-[color:rgba(242,171,67,0.26)] bg-[color:rgba(242,171,67,0.1)] px-2 py-0.5 text-[10px] font-semibold uppercase text-[color:var(--warn)]">
            Trends + Intelligence
          </span>
        </div>
        <h1 className="mt-3 text-2xl font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
          Intel Beta
        </h1>
        <p className="mt-1.5 max-w-3xl text-sm text-[color:var(--ink-faint)]">
          Explore a combined workflow where system-level trend detection selects the signal, then the
          evidence layer explains drivers, coverage, articles, and market read-through.
        </p>
      </div>

      <IntelBetaDashboard initialSignals={initialSignals} initialTrends={initialTrends} />
    </main>
  );
}
