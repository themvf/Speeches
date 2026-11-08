import type { Metadata } from "next";
import { PortfolioWorkspace } from "@/components/portfolio-workspace";

export const metadata: Metadata = {
  title: "Portfolios | Policy Research Hub",
  description: "Create and review saved investment portfolios with live market-value estimates.",
};

export default function PortfoliosPage() {
  return <main className="mx-auto w-full max-w-[1500px] px-4 py-8 md:px-8">
    <div className="mb-6"><span className="kicker">Portfolio workspace</span><h1 className="mt-3 text-2xl font-semibold text-[color:var(--ink)]">Track ideas from entry price to today</h1><p className="mt-1 max-w-3xl text-sm text-[color:var(--ink-faint)]">Build separate portfolios, save shares and trade prices, and return later to review estimated performance and concentration.</p></div>
    <PortfolioWorkspace />
  </main>;
}
