import type { Metadata } from "next";
import { EnforcementBetaDashboard } from "@/components/enforcement-beta-dashboard";

export const metadata: Metadata = {
  title: "Enforcement | Policy Research Hub",
  description: "Combined SEC and FINRA enforcement research view with citation heatmaps and a filterable action feed."
};

export default function EnforcementPage() {
  return (
    <main className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8">
      <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center rounded-full border border-[color:rgba(255,107,127,0.24)] bg-[color:rgba(255,107,127,0.1)] px-2 py-0.5 text-[10px] font-semibold uppercase text-[color:#ff6b7f]">
              SEC
            </span>
            <span className="inline-flex items-center rounded-full border border-[color:rgba(79,213,255,0.22)] bg-[color:rgba(79,213,255,0.1)] px-2 py-0.5 text-[10px] font-semibold uppercase text-[color:var(--accent)]">
              FINRA
            </span>
          </div>
          <h1 className="mt-3 text-2xl font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
            Enforcement
          </h1>
          <p className="mt-1.5 max-w-3xl text-sm text-[color:var(--ink-faint)]">
            Combined enforcement research view with agency modes, citation coverage, heatmaps, and a filterable action feed.
          </p>
        </div>
      </div>

      <EnforcementBetaDashboard />
    </main>
  );
}
