import { type NextRequest } from "next/server";
import { createRequestId, ok } from "@/lib/server/api-utils";
import type { IndustryPeerRow, IndustrySummary, MarketIndustriesData } from "@/lib/server/types";
import {
  getDailyStockAttention,
  getLatestStockAttentionDate,
  getPolymarketOpenMarkets,
} from "@/lib/server/neon";
import { fetchYahooQuote } from "@/lib/server/yahoo";
import industryConfig from "@/lib/server/industry-config.json" with { type: "json" };

export const runtime = "nodejs";
// Dynamic for the DB joins (attention counts, report dates) and live quotes;
// the industry grouping itself is the committed SIC config (SEC-53).
export const dynamic = "force-dynamic";

type JoinData = {
  mentionsByTicker: Map<string, number>;
  reportByTicker: Map<string, string>;
};

// Both joins are enrichment, not structure - a missing table or unreachable
// DB must never blank the Industries list (the config alone renders fine).
async function loadJoins(): Promise<JoinData & { warning?: string }> {
  const mentionsByTicker = new Map<string, number>();
  const reportByTicker = new Map<string, string>();
  let warning: string | undefined;
  try {
    const date = await getLatestStockAttentionDate();
    if (date) {
      for (const row of await getDailyStockAttention(date, 500)) {
        mentionsByTicker.set(row.ticker, row.total_mention_count);
      }
    }
  } catch {
    warning = "Attention counts unavailable (DB unreachable) - showing industry structure only.";
  }
  try {
    for (const market of await getPolymarketOpenMarkets()) {
      if (market.ticker && market.report_date) reportByTicker.set(market.ticker, market.report_date);
    }
  } catch {
    // Same fail-soft stance; attention warning (if any) already covers it.
  }
  return { mentionsByTicker, reportByTicker, warning };
}

export async function GET(req: NextRequest) {
  const requestId = createRequestId();
  const industryLabel = (req.nextUrl.searchParams.get("industry") ?? "").trim();
  const joins = await loadJoins();

  const industries: IndustrySummary[] = industryConfig.industries.map((industry) => {
    const symbols = industry.tickers.map((t) => t.ticker);
    return {
      sic: industry.sic,
      label: industry.label,
      tickers: symbols,
      attentionTotal: symbols.reduce((sum, s) => sum + (joins.mentionsByTicker.get(s) ?? 0), 0),
      reportingSoon: symbols
        .filter((s) => joins.reportByTicker.has(s))
        .map((s) => ({ ticker: s, reportDate: joins.reportByTicker.get(s)! })),
    };
  });

  const data: MarketIndustriesData = {
    generatedAt: new Date().toISOString(),
    industries,
    ...(joins.warning ? { warning: joins.warning } : {}),
  };

  // Peer drill-down: live quotes ONLY for the one expanded industry (bounded
  // to its member count; never an all-universe quote fetch).
  if (industryLabel) {
    const industry = industryConfig.industries.find((entry) => entry.label === industryLabel);
    if (industry) {
      const quotes = await Promise.allSettled(
        industry.tickers.map((t) => fetchYahooQuote(t.ticker, 300))
      );
      const rows: IndustryPeerRow[] = industry.tickers.map((t, i) => {
        const settled = quotes[i];
        const quote = settled && settled.status === "fulfilled" ? settled.value : null;
        const entry = t as typeof t & {
          revenue?: number; expenses?: number; profit?: number;
          sharesOutstanding?: number; periodEnd?: string;
        };
        const price = quote?.price ?? null;
        const shares = entry.sharesOutstanding ?? null;
        return {
          ticker: t.ticker,
          name: t.name,
          price,
          pricePct: quote?.pct ?? null,
          marketCap: price != null && shares != null ? price * shares : null,
          revenue: entry.revenue ?? null,
          expenses: entry.expenses ?? null,
          profit: entry.profit ?? null,
          periodEnd: entry.periodEnd ?? null,
          mentions: joins.mentionsByTicker.get(t.ticker) ?? 0,
          reportDate: joins.reportByTicker.get(t.ticker) ?? null,
        };
      });
      // Biggest first - the natural read for a peer table.
      rows.sort((a, b) => (b.marketCap ?? -Infinity) - (a.marketCap ?? -Infinity));
      data.peers = { label: industry.label, rows };
    }
  }

  return ok(data, requestId);
}
