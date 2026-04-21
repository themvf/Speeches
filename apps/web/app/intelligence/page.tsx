import type { Metadata } from "next";
import { ThemeIntelligenceLab } from "@/components/theme-intelligence-lab";
import { buildIntelligenceSignalsData } from "@/lib/server/intelligence-signals";

export const metadata: Metadata = {
  title: "Intelligence | Policy Research Hub",
  description: "Evidence-backed macro narratives, trend changes, and market impact for intelligence workflows."
};

export default function IntelligencePage() {
  const initialData = buildIntelligenceSignalsData();

  return (
    <main className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8">
      <div className="mb-6">
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center rounded-full border border-[color:rgba(79,213,255,0.22)] bg-[color:rgba(79,213,255,0.1)] px-2 py-0.5 text-[10px] font-semibold uppercase text-[color:var(--accent)]">
            GDELT
          </span>
          <span className="inline-flex items-center rounded-full border border-[color:rgba(242,171,67,0.26)] bg-[color:rgba(242,171,67,0.1)] px-2 py-0.5 text-[10px] font-semibold uppercase text-[color:var(--warn)]">
            Macro Signals
          </span>
        </div>
        <h1 className="mt-3 text-2xl font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
          Intelligence
        </h1>
        <p className="mt-1.5 max-w-3xl text-sm text-[color:var(--ink-faint)]">
          Track what changed, why it matters, and which evidence supports the market read-through.
        </p>
      </div>

      <ThemeIntelligenceLab initialData={initialData} />
    </main>
  );
}
