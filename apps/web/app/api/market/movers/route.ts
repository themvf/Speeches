import { createRequestId, ok } from "@/lib/server/api-utils";
import type { MarketMoversData, MoverQuote } from "@/lib/server/types";
import { fetchYahooQuote } from "@/lib/server/yahoo";

export const runtime = "nodejs";
export const revalidate = 120;

const WATCHLIST: { symbol: string; name: string }[] = [
  { symbol: "AAPL", name: "Apple Inc." },
  { symbol: "MSFT", name: "Microsoft Corp." },
  { symbol: "NVDA", name: "Nvidia Corp." },
  { symbol: "AMD", name: "Advanced Micro Devices" },
  { symbol: "TSLA", name: "Tesla Inc." },
  { symbol: "META", name: "Meta Platforms" },
  { symbol: "GOOGL", name: "Alphabet Inc." },
  { symbol: "AMZN", name: "Amazon.com" },
  { symbol: "NFLX", name: "Netflix Inc." },
  { symbol: "PLTR", name: "Palantir Technologies" },
  { symbol: "COIN", name: "Coinbase Global" },
  { symbol: "MSTR", name: "Strategy Inc." },
  { symbol: "SMCI", name: "Super Micro Computer" },
  { symbol: "SOFI", name: "SoFi Technologies" },
  { symbol: "RIVN", name: "Rivian Automotive" },
  { symbol: "GME", name: "GameStop Corp." },
  { symbol: "JPM", name: "JPMorgan Chase" },
  { symbol: "BAC", name: "Bank of America" },
  { symbol: "GS", name: "Goldman Sachs" },
  { symbol: "XOM", name: "Exxon Mobil" },
  { symbol: "CVX", name: "Chevron Corp." },
  { symbol: "PFE", name: "Pfizer Inc." },
  { symbol: "MRNA", name: "Moderna Inc." },
  { symbol: "BA", name: "Boeing Co." },
  { symbol: "DIS", name: "Walt Disney Co." },
  { symbol: "UBER", name: "Uber Technologies" },
  { symbol: "SNOW", name: "Snowflake Inc." },
  { symbol: "SHOP", name: "Shopify Inc." },
  { symbol: "PYPL", name: "PayPal Holdings" },
  { symbol: "INTC", name: "Intel Corp." },
  { symbol: "MU", name: "Micron Technology" },
  { symbol: "IONQ", name: "IonQ Inc." },
  { symbol: "HOOD", name: "Robinhood Markets" },
  { symbol: "SQ", name: "Block Inc." },
  { symbol: "QCOM", name: "Qualcomm Inc." },
];

export async function GET() {
  const requestId = createRequestId();

  const settled = await Promise.allSettled(
    WATCHLIST.map(({ symbol }) => fetchYahooQuote(symbol, 120))
  );

  const quoted = settled
    .map((r, i) => ({
      ...WATCHLIST[i],
      q: r.status === "fulfilled" ? r.value : null,
    }))
    .filter((item) => item.q != null)
    .map((item) => ({
      symbol: item.symbol,
      name: item.name,
      price: item.q!.price,
      pct: item.q!.pct,
      change: item.q!.change,
      up: item.q!.change >= 0,
    }));

  const byPct = [...quoted].sort((a, b) => b.pct - a.pct);

  const gainers: MoverQuote[] = byPct
    .filter((q) => q.pct > 0)
    .slice(0, 10)
    .map((q, i) => ({ rank: i + 1, ...q }));

  const losers: MoverQuote[] = [...byPct]
    .reverse()
    .filter((q) => q.pct < 0)
    .slice(0, 10)
    .map((q, i) => ({ rank: i + 1, ...q }));

  const data: MarketMoversData = { gainers, losers, generatedAt: new Date().toISOString() };
  return ok(data, requestId);
}
