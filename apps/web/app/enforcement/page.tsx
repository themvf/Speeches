import type { Metadata } from "next";
import { EnforcementDashboard } from "@/components/enforcement-dashboard";

export const metadata: Metadata = {
  title: "Enforcement Trends | Policy Research Hub",
  description:
    "Unified FINRA and SEC enforcement heatmap — rule violation trends across AWC, OHO, NAC, and Litigation Releases.",
};

export default function EnforcementPage() {
  return (
    <main className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8">
      <div className="mb-6">
        <div className="flex items-center gap-2">
          <span
            className="inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.1em]"
            style={{
              background: "rgba(79,213,255,0.08)",
              color: "var(--accent)",
              border: "1px solid rgba(79,213,255,0.18)",
            }}
          >
            FINRA
          </span>
          <span
            className="inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.1em]"
            style={{
              background: "rgba(255,80,80,0.08)",
              color: "#ff5050",
              border: "1px solid rgba(255,80,80,0.18)",
            }}
          >
            SEC
          </span>
          <h1 className="text-xl font-semibold text-[color:var(--ink)]">
            Enforcement Trends
          </h1>
        </div>
        <p className="mt-1.5 text-sm text-[color:var(--ink-faint)]">
          Rule violation heatmaps across FINRA disciplinary actions and SEC litigation releases
        </p>
      </div>

      <EnforcementDashboard />
    </main>
  );
}
