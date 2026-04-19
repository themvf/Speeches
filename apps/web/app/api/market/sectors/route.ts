import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import type { MarketSectorsData, SectorData, SectorPcts, SectorStock } from "@/lib/server/types";

export const runtime = "nodejs";
export const revalidate = 300;

type FinnhubQuote = { c: number | null; d: number | null; dp: number | null };
type FinnhubSectorPerf = {
  sector: string;
  changesPercentage: string;
  d1?: string;
  m1?: string;
  m3?: string;
  ytd?: string;
};

// Finnhub sector names → display names
const SECTOR_NAME_MAP: Record<string, string> = {
  "Technology": "Technology",
  "Communication Services": "Communication Services",
  "Consumer Cyclical": "Consumer Discretionary",
  "Consumer Defensive": "Consumer Staples",
  "Energy": "Energy",
  "Financial Services": "Financials",
  "Healthcare": "Healthcare",
  "Industrials": "Industrials",
  "Basic Materials": "Materials",
  "Real Estate": "Real Estate",
  "Utilities": "Utilities",
};

const SECTOR_STOCKS: Record<string, { symbol: string; name: string }[]> = {
  "Technology": [
    { symbol: "AAPL", name: "Apple Inc." },
    { symbol: "MSFT", name: "Microsoft Corp." },
    { symbol: "NVDA", name: "Nvidia Corp." },
    { symbol: "AMD", name: "Advanced Micro Devices" },
    { symbol: "INTC", name: "Intel Corp." },
  ],
  "Communication Services": [
    { symbol: "GOOGL", name: "Alphabet Inc." },
    { symbol: "META", name: "Meta Platforms" },
    { symbol: "NFLX", name: "Netflix Inc." },
    { symbol: "DIS", name: "Walt Disney Co." },
    { symbol: "T", name: "AT&T Inc." },
  ],
  "Consumer Cyclical": [
    { symbol: "AMZN", name: "Amazon.com" },
    { symbol: "TSLA", name: "Tesla Inc." },
    { symbol: "HD", name: "Home Depot" },
    { symbol: "MCD", name: "McDonald's Corp." },
    { symbol: "NKE", name: "Nike Inc." },
  ],
  "Consumer Defensive": [
    { symbol: "WMT", name: "Walmart Inc." },
    { symbol: "PG", name: "Procter & Gamble" },
    { symbol: "KO", name: "Coca-Cola Co." },
    { symbol: "PEP", name: "PepsiCo Inc." },
    { symbol: "SBUX", name: "Starbucks Corp." },
  ],
  "Energy": [
    { symbol: "XOM", name: "Exxon Mobil" },
    { symbol: "CVX", name: "Chevron Corp." },
    { symbol: "COP", name: "ConocoPhillips" },
    { symbol: "SLB", name: "SLB (Schlumberger)" },
    { symbol: "OXY", name: "Occidental Petroleum" },
  ],
  "Financial Services": [
    { symbol: "JPM", name: "JPMorgan Chase" },
    { symbol: "BAC", name: "Bank of America" },
    { symbol: "GS", name: "Goldman Sachs" },
    { symbol: "MS", name: "Morgan Stanley" },
    { symbol: "WFC", name: "Wells Fargo" },
  ],
  "Healthcare": [
    { symbol: "JNJ", name: "Johnson & Johnson" },
    { symbol: "PFE", name: "Pfizer Inc." },
    { symbol: "MRNA", name: "Moderna Inc." },
    { symbol: "UNH", name: "UnitedHealth Group" },
    { symbol: "ABBV", name: "AbbVie Inc." },
  ],
  "Industrials": [
    { symbol: "BA", name: "Boeing Co." },
    { symbol: "LMT", name: "Lockheed Martin" },
    { symbol: "UPS", name: "United Parcel Service" },
    { symbol: "CAT", name: "Caterpillar Inc." },
    { symbol: "RTX", name: "RTX Corp." },
  ],
  "Basic Materials": [
    { symbol: "LIN", name: "Linde plc" },
    { symbol: "APD", name: "Air Products" },
    { symbol: "NEM", name: "Newmont Corp." },
    { symbol: "FCX", name: "Freeport-McMoRan" },
    { symbol: "ALB", name: "Albemarle Corp." },
  ],
  "Real Estate": [
    { symbol: "PLD", name: "Prologis Inc." },
    { symbol: "AMT", name: "American Tower" },
    { symbol: "EQIX", name: "Equinix Inc." },
    { symbol: "SPG", name: "Simon Property Group" },
    { symbol: "O", name: "Realty Income Corp." },
  ],
  "Utilities": [
    { symbol: "NEE", name: "NextEra Energy" },
    { symbol: "DUK", name: "Duke Energy" },
    { symbol: "SO", name: "Southern Co." },
    { symbol: "D", name: "Dominion Energy" },
    { symbol: "AEP", name: "American Electric Power" },
  ],
};

function parsePct(s: string): number {
  return parseFloat(s.replace(/[^0-9.\-]/g, "")) || 0;
}

async function fetchQuote(symbol: string, apiKey: string): Promise<FinnhubQuote | null> {
  try {
    const res = await fetch(
      `https://finnhub.io/api/v1/quote?symbol=${encodeURIComponent(symbol)}&token=${apiKey}`,
      { next: { revalidate: 300 } }
    );
    if (!res.ok) return null;
    const data: FinnhubQuote = await res.json();
    return data.c != null ? data : null;
  } catch {
    return null;
  }
}

export async function GET() {
  const requestId = createRequestId();
  const apiKey = process.env.FINNHUB_API_KEY;
  if (!apiKey) return fail("FINNHUB_API_KEY not set", "NO_API_KEY", 500, requestId);

  // Fetch sector-level performance
  let sectorPerf: FinnhubSectorPerf[] = [];
  try {
    const res = await fetch(
      `https://finnhub.io/api/v1/stock/sector-performance?token=${apiKey}`,
      { next: { revalidate: 300 } }
    );
    if (res.ok) sectorPerf = await res.json();
  } catch { /* fall through — stocks will show pct 0 for sector header */ }

  const perfMap = new Map(
    sectorPerf.map((s) => [
      s.sector,
      {
        d1:  parsePct(s.d1  ?? s.changesPercentage),
        m1:  parsePct(s.m1  ?? "0"),
        m3:  parsePct(s.m3  ?? "0"),
        ytd: parsePct(s.ytd ?? "0"),
      } satisfies SectorPcts,
    ])
  );

  // Gather all stocks to quote in one parallel batch
  const allStocks: { finnhubKey: string; symbol: string; name: string }[] = [];
  for (const [finnhubKey, stocks] of Object.entries(SECTOR_STOCKS)) {
    for (const s of stocks) {
      allStocks.push({ finnhubKey, ...s });
    }
  }

  const settled = await Promise.allSettled(
    allStocks.map(({ symbol }) => fetchQuote(symbol, apiKey))
  );

  const stockQuotes = settled.map((r, i) => ({
    ...allStocks[i],
    q: r.status === "fulfilled" ? r.value : null,
  }));

  // Build sectors
  const defaultPcts: SectorPcts = { d1: 0, m1: 0, m3: 0, ytd: 0 };

  const sectors: SectorData[] = Object.entries(SECTOR_STOCKS).map(([finnhubKey, stockDefs]) => {
    const displayName = SECTOR_NAME_MAP[finnhubKey] ?? finnhubKey;
    const pcts = perfMap.get(finnhubKey) ?? defaultPcts;

    const stocks: SectorStock[] = stockDefs
      .map((def) => {
        const found = stockQuotes.find((sq) => sq.symbol === def.symbol && sq.finnhubKey === finnhubKey);
        if (!found?.q) return null;
        return {
          symbol: def.symbol,
          name: def.name,
          price: found.q.c ?? 0,
          pct: found.q.dp ?? 0,
          change: found.q.d ?? 0,
          up: (found.q.d ?? 0) >= 0,
        };
      })
      .filter((s): s is SectorStock => s !== null);

    return { name: displayName, pcts, stocks };
  });

  sectors.sort((a, b) => b.pcts.d1 - a.pcts.d1);

  const data: MarketSectorsData = { sectors, generatedAt: new Date().toISOString() };
  return ok(data, requestId);
}
