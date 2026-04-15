import type { Metadata } from "next";
import { TrendsDashboard } from "@/components/trends-dashboard";

export const metadata: Metadata = {
  title: "Emerging Trends | Policy Research Hub",
  description: "Track emerging regulatory and financial themes across all ingested documents."
};

export default function TrendsPage() {
  return (
    <main className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8">
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-[color:var(--ink)]">Emerging Trends</h1>
        <p className="mt-1 text-sm text-[color:var(--ink-faint)]">
          Tag-based trends aggregated daily across SEC speeches, regulatory notices, news, and more.
          Growth is measured against the prior 30-day baseline.
        </p>
      </div>
      <TrendsDashboard />
    </main>
  );
}
