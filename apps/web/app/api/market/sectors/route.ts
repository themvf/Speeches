import { createRequestId, ok } from "@/lib/server/api-utils";
import { MARKET_SECTOR_COMPANIES } from "@/lib/server/market-sector-companies";
import type { MarketSectorsData, SectorData, SectorPcts, SectorStock } from "@/lib/server/types";
import { fetchYahooCandles, fetchYahooQuote } from "@/lib/server/yahoo";

export const runtime = "nodejs";
export const revalidate = 300;

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

function priceAt(candle: { t: number[]; c: number[] }, targetUnix: number): number | null {
  let best: number | null = null;
  let bestDiff = Infinity;
  for (let i = 0; i < candle.t.length; i++) {
    if (candle.t[i] > targetUnix + 86400) continue;
    const diff = targetUnix - candle.t[i];
    if (diff >= 0 && diff < bestDiff) { bestDiff = diff; best = candle.c[i]; }
  }
  return best;
}

function computePcts(candle: { t: number[]; c: number[] }): SectorPcts {
  const last = candle.c[candle.c.length - 1];
  const prev = candle.c.length > 1 ? candle.c[candle.c.length - 2] : null;
  const now = Date.now() / 1000;
  const ytdStart = new Date(new Date().getFullYear(), 0, 1).getTime() / 1000;
  const pct = (ref: number | null) =>
    ref && ref > 0 ? ((last - ref) / ref) * 100 : 0;
  return {
    d1:  prev && prev > 0 ? ((last - prev) / prev) * 100 : 0,
    w1:  pct(priceAt(candle, now - 7  * 86400)),
    m1:  pct(priceAt(candle, now - 30 * 86400)),
    m3:  pct(priceAt(candle, now - 90 * 86400)),
    ytd: pct(priceAt(candle, ytdStart)),
  };
}

export async function GET() {
  const requestId = createRequestId();
  const sectorKeys = Object.keys(MARKET_SECTOR_COMPANIES);

  const etfCandles = await Promise.allSettled(
    sectorKeys.map((key) => fetchYahooCandles(SECTOR_ETFS[key]))
  );

  const allStocks = sectorKeys.flatMap((key) =>
    MARKET_SECTOR_COMPANIES[key].map((s) => ({ ...s, sectorKey: key }))
  );
  const stockSettled = await Promise.allSettled(
    allStocks.map(({ symbol }) => fetchYahooQuote(symbol, 300))
  );
  const quoteIndex = new Map(
    allStocks.map((stock, index) => [`${stock.sectorKey}:${stock.symbol}`, index])
  );

  const sectors: SectorData[] = sectorKeys.map((key, i) => {
    const candle = etfCandles[i].status === "fulfilled" ? etfCandles[i].value : null;
    const pcts: SectorPcts = candle
      ? computePcts(candle)
      : { d1: 0, w1: 0, m1: 0, m3: 0, ytd: 0 };

    const stocks: SectorStock[] = MARKET_SECTOR_COMPANIES[key]
      .map((def) => {
        const idx = quoteIndex.get(`${key}:${def.symbol}`);
        const q = idx !== undefined && stockSettled[idx].status === "fulfilled" ? stockSettled[idx].value : null;
        if (!q) return null;
        return { symbol: def.symbol, name: def.name, price: q.price, pct: q.pct, change: q.change, up: q.change >= 0 };
      })
      .filter((s): s is SectorStock => s !== null);

    return { name: SECTOR_NAME_MAP[key] ?? key, pcts, stocks };
  });

  sectors.sort((a, b) => b.pcts.d1 - a.pcts.d1);
  const data: MarketSectorsData = { sectors, generatedAt: new Date().toISOString() };
  return ok(data, requestId);
}
