import type { Metadata } from "next";
import { FINRADashboard } from "@/components/finra-dashboard";

export const metadata: Metadata = {
  title: "FINRA Enforcement | Policy Research Hub",
  description:
    "Heatmap of FINRA rule violations across disciplinary actions — AWC, OHO, and NAC decisions.",
};

export default function FINRAPage() {
  return (
    <main className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8">
      <div className="mb-6">
        <div className="flex items-center gap-2">
          <span
            className="inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.1em]"
            style={{
              background: "rgba(79,213,255,0.1)",
              color: "var(--accent)",
              border: "1px solid rgba(79,213,255,0.2)",
            }}
          >
            FINRA
          </span>
          <h1 className="text-xl font-semibold text-[color:var(--ink)]">
            Enforcement Intelligence
          </h1>
        </div>
        <p className="mt-1.5 text-sm text-[color:var(--ink-faint)]">
          Rule violation heatmap across AWC, OHO, and NAC disciplinary actions
        </p>
      </div>

      <FINRADashboard />
    </main>
  );
}
