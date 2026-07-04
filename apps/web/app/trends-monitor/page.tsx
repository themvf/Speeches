import type { Metadata } from "next";
import { TrendsMonitorDashboard } from "@/components/trends-monitor-dashboard";

export const metadata: Metadata = {
  title: "Trend Monitor | Policy Research Hub",
  description: "Market-monitor style view for regulatory and financial trend surveillance."
};

export default function TrendsMonitorPage() {
  return (
    <main className="mx-auto w-full max-w-[1500px] px-4 py-6 md:px-8">
      <div className="mb-5 flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <h1 className="text-xl font-semibold text-[color:var(--ink)]">Trend Monitor</h1>
          <p className="mt-1 max-w-3xl text-sm leading-6 text-[color:var(--ink-faint)]">
            Dense monitoring view for rising, cooling, high-volume, and newly emerging policy themes.
          </p>
        </div>
      </div>
      <TrendsMonitorDashboard />
    </main>
  );
}
