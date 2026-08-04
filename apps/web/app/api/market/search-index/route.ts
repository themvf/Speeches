import { ok } from "@/lib/server/api-utils";
import type { MarketSearchIndexData, TickerSearchEntry } from "@/lib/server/types";
import industryConfig from "@/lib/server/industry-config.json" with { type: "json" };

export const runtime = "nodejs";
// The committed industry-config.json only changes when build_industry_config.py
// is rerun (see CLAUDE.md, SEC-53/57) - a static response lets the client
// fetch this once per deploy instead of on every keystroke.
export const dynamic = "force-static";

// Market-page global search (ticker or company name): a stripped-down index
// over the same tracked universe the Industries tab and fundamentals lookup
// already use, with financial fields dropped since the client only needs
// ticker/name/industry to build the autocomplete list.
export async function GET() {
  const entries: TickerSearchEntry[] = industryConfig.industries.flatMap((industry) =>
    industry.tickers.map((t) => ({ ticker: t.ticker, name: t.name, industry: industry.label }))
  );
  entries.sort((a, b) => a.ticker.localeCompare(b.ticker));

  const data: MarketSearchIndexData = {
    generatedAt: industryConfig.generatedAt,
    entries,
  };
  return ok(data);
}
