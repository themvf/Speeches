import type { Metadata } from "next";
import { MarketDashboard } from "@/components/market-dashboard";

export const metadata: Metadata = {
  title: "Market | Policy Research Hub",
  description: "Live market overview — indices, sectors, movers, crypto, and global exchanges.",
};

export default function MarketPage() {
  return (
    <main className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8">
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-[color:var(--ink)]">Market</h1>
        <p className="mt-1 text-sm text-[color:var(--ink-faint)]">
          Live indices, sector performance, top movers, crypto markets, and global exchange status.
        </p>
      </div>
      <MarketDashboard />
    </main>
  );
}
