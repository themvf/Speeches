import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import type { MarketSectorsData, SectorData, SectorPcts, SectorStock } from "@/lib/server/types";

export const runtime = "nodejs";
export const revalidate = 300;

type FinnhubQuote = { c: number | null; d: number | null; dp: number | null };
type FinnhubCandle = { c: number[]; t: number[]; s: string };

// Sector ETFs for accurate multi-period returns (candle data)
const SECTOR_ETFS: Record<string, string> = {
  "Technology":             "XLK",
  "Communication Services": "XLC",
  "Consumer Cyclical":      "XLY",
  "Consumer Defensive":     "XLP",
  "Energy":                 "XLE",
  "Financial Services":     "XLF",
  "Healthcare":             "XLV",
  "Industrials":            "XLI",
  "Basic Materials":        "XLB",
  "Real Estate":            "XLRE",
  "Utilities":              "XLU",
};

const SECTOR_NAME_MAP: Record<string, string> = {
  "Technology":             "Technology",
  "Communication Services": "Communication Services",
  "Consumer Cyclical":      "Consumer Discretionary",
  "Consumer Defensive":     "Consumer Staples",
  "Energy":                 "Energy",
  "Financial Services":     "Financials",
  "Healthcare":             "Healthcare",
  "Industrials":            "Industrials",
  "Basic Materials":        "Materials",
  "Real Estate":            "Real Estate",
  "Utilities":              "Utilities",
};

const SECTOR_STOCKS: Record<string, { symbol: string; name: string }[]> = {
  "Technology":             [{ symbol: "AAPL", name: "Apple Inc." }, { symbol: "MSFT", name: "Microsoft" }, { symbol: "NVDA", name: "Nvidia" }],
  "Communication Services": [{ symbol: "GOOGL", name: "Alphabet" }, { symbol: "META", name: "Meta Platforms" }, { symbol: "NFLX", name: "Netflix" }],
  "Consumer Cyclical":      [{ symbol: "AMZN", name: "Amazon" }, { symbol: "TSLA", name: "Tesla" }, { symbol: "HD", name: "Home Depot" }],
  "Consumer Defensive":     [{ symbol: "WMT", name: "Walmart" }, { symbol: "PG", name: "Procter & Gamble" }, { symbol: "KO", name: "Coca-Cola" }],
  "Energy":                 [{ symbol: "XOM", name: "Exxon Mobil" }, { symbol: "CVX", name: "Chevron" }, { symbol: "COP", name: "ConocoPhillips" }],
  "Financial Services":     [{ symbol: "JPM", name: "JPMorgan Chase" }, { symbol: "BAC", name: "Bank of America" }, { symbol: "GS", name: "Goldman Sachs" }],
  "Healthcare":             [{ symbol: "JNJ", name: "J&J" }, { symbol: "UNH", name: "UnitedHealth" }, { symbol: "PFE", name: "Pfizer" }],
  "Industrials":            [{ symbol: "BA", name: "Boeing" }, { symbol: "CAT", name: "Caterpillar" }, { symbol: "UPS", name: "UPS" }],
  "Basic Materials":        [{ symbol: "LIN", name: "Linde" }, { symbol: "FCX", name: "Freeport-McMoRan" }, { symbol: "NEM", name: "Newmont" }],
  "Real Estate":            [{ symbol: "PLD", name: "Prologis" }, { symbol: "AMT", name: "American Tower" }, { symbol: "EQIX", name: "Equinix" }],
  "Utilities":              [{ symbol: "NEE", name: "NextEra Energy" }, { symbol: "DUK", name: "Duke Energy" }, { symbol: "SO", name: "Southern Co." }],
};

async function fetchQuote(symbol: string, apiKey: string): Promise<FinnhubQuote | null> {
  try {
    const res = await fetch(
      `https://finnhub.io/api/v1/quote?symbol=${encodeURIComponent(symbol)}&token=${apiKey}`,
      { next: { revalidate: 300 } }
    );
    if (!res.ok) return null;
    const data: FinnhubQuote = await res.json();
    return data.c != null ? data : null;
  } catch { return null; }
}

async function fetchCandles(symbol: string, apiKey: string): Promise<FinnhubCandle | null> {
  const to = Math.floor(Date.now() / 1000);
  const from = to - 400 * 86400; // 13+ months back (covers YTD + 3M)
  try {
    const res = await fetch(
      `https://finnhub.io/api/v1/stock/candle?symbol=${encodeURIComponent(symbol)}&resolution=D&from=${from}&to=${to}&token=${apiKey}`,
      { next: { revalidate: 3600 } }
    );
    if (!res.ok) return null;
    const data: FinnhubCandle = await res.json();
    return data.s === "ok" && data.c?.length ? data : null;
  } catch { return null; }
}

function priceAtTarget(candle: FinnhubCandle, targetUnix: number): number | null {
  let best: number | null = null;
  let bestDiff = Infinity;
  for (let i = 0; i < candle.t.length; i++) {
    const diff = Math.abs(candle.t[i] - targetUnix);
    if (diff < bestDiff && candle.t[i] <= targetUnix + 86400) {
      bestDiff = diff;
      best = candle.c[i];
    }
  }
  return best;
}

function computePcts(candle: FinnhubCandle, currentPrice: number): SectorPcts {
  const now = Date.now() / 1000;
  const today = new Date();
  const ytdStart = new Date(today.getFullYear(), 0, 1).getTime() / 1000;

  const p1w  = priceAtTarget(candle, now - 7 * 86400);
  const p1m  = priceAtTarget(candle, now - 30 * 86400);
  const p3m  = priceAtTarget(candle, now - 90 * 86400);
  const pYtd = priceAtTarget(candle, ytdStart);

  const pct = (ref: number | null) =>
    ref && ref > 0 ? ((currentPrice - ref) / ref) * 100 : 0;

  return {
    d1:  0, // filled from quote below
    w1:  pct(p1w),
    m1:  pct(p1m),
    m3:  pct(p3m),
    ytd: pct(pYtd),
  };
}

export async function GET() {
  const requestId = createRequestId();
  const apiKey = process.env.FINNHUB_API_KEY;
  if (!apiKey) return fail("FINNHUB_API_KEY not set", "NO_API_KEY", 500, requestId);

  const sectorKeys = Object.keys(SECTOR_STOCKS);

  // Fetch sector ETF quotes + candles in parallel (22 calls total)
  const [etfQuotes, etfCandles] = await Promise.all([
    Promise.allSettled(sectorKeys.map((key) => fetchQuote(SECTOR_ETFS[key], apiKey))),
    Promise.allSettled(sectorKeys.map((key) => fetchCandles(SECTOR_ETFS[key], apiKey))),
  ]);

  // Fetch all stock quotes in parallel
  const allStocks = sectorKeys.flatMap((key) =>
    SECTOR_STOCKS[key].map((s) => ({ ...s, sectorKey: key }))
  );
  const stockSettled = await Promise.allSettled(
    allStocks.map(({ symbol }) => fetchQuote(symbol, apiKey))
  );

  const sectors: SectorData[] = sectorKeys.map((finnhubKey, i) => {
    const displayName = SECTOR_NAME_MAP[finnhubKey] ?? finnhubKey;
    const quote  = etfQuotes[i].status === "fulfilled" ? etfQuotes[i].value : null;
    const candle = etfCandles[i].status === "fulfilled" ? etfCandles[i].value : null;

    const currentPrice = quote?.c ?? 0;
    const pcts: SectorPcts = candle && currentPrice > 0
      ? { ...computePcts(candle, currentPrice), d1: quote?.dp ?? 0 }
      : { d1: quote?.dp ?? 0, w1: 0, m1: 0, m3: 0, ytd: 0 };

    const stocks: SectorStock[] = SECTOR_STOCKS[finnhubKey]
      .map((def) => {
        const idx = allStocks.findIndex((s) => s.symbol === def.symbol && s.sectorKey === finnhubKey);
        const q = idx >= 0 && stockSettled[idx].status === "fulfilled"
          ? stockSettled[idx].value
          : null;
        if (!q) return null;
        return {
          symbol: def.symbol,
          name: def.name,
          price: q.c ?? 0,
          pct: q.dp ?? 0,
          change: q.d ?? 0,
          up: (q.d ?? 0) >= 0,
        };
      })
      .filter((s): s is SectorStock => s !== null);

    return { name: displayName, pcts, stocks };
  });

  sectors.sort((a, b) => b.pcts.d1 - a.pcts.d1);

  const data: MarketSectorsData = { sectors, generatedAt: new Date().toISOString() };
  return ok(data, requestId);
}
