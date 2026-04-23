import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import type { MarketSectorsData, SectorData, SectorPcts, SectorStock } from "@/lib/server/types";

export const runtime = "nodejs";
export const revalidate = 300;

type FinnhubQuote = { c: number | null; d: number | null; dp: number | null };
type YahooCandle  = { t: number[]; c: number[] };

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
  "Technology":             [{ symbol: "AAPL", name: "Apple" }, { symbol: "MSFT", name: "Microsoft" }, { symbol: "NVDA", name: "Nvidia" }],
  "Communication Services": [{ symbol: "GOOGL", name: "Alphabet" }, { symbol: "META", name: "Meta" }, { symbol: "NFLX", name: "Netflix" }],
  "Consumer Cyclical":      [{ symbol: "AMZN", name: "Amazon" }, { symbol: "TSLA", name: "Tesla" }, { symbol: "HD", name: "Home Depot" }],
  "Consumer Defensive":     [{ symbol: "WMT", name: "Walmart" }, { symbol: "PG", name: "P&G" }, { symbol: "KO", name: "Coca-Cola" }],
  "Energy":                 [{ symbol: "XOM", name: "Exxon" }, { symbol: "CVX", name: "Chevron" }, { symbol: "COP", name: "ConocoPhillips" }],
  "Financial Services":     [{ symbol: "JPM", name: "JPMorgan" }, { symbol: "BAC", name: "Bank of America" }, { symbol: "GS", name: "Goldman Sachs" }],
  "Healthcare":             [{ symbol: "JNJ", name: "J&J" }, { symbol: "UNH", name: "UnitedHealth" }, { symbol: "PFE", name: "Pfizer" }],
  "Industrials":            [{ symbol: "BA", name: "Boeing" }, { symbol: "CAT", name: "Caterpillar" }, { symbol: "UPS", name: "UPS" }],
  "Basic Materials":        [{ symbol: "LIN", name: "Linde" }, { symbol: "FCX", name: "Freeport-McMoRan" }, { symbol: "NEM", name: "Newmont" }],
  "Real Estate":            [{ symbol: "PLD", name: "Prologis" }, { symbol: "AMT", name: "Amer. Tower" }, { symbol: "EQIX", name: "Equinix" }],
  "Utilities":              [{ symbol: "NEE", name: "NextEra" }, { symbol: "DUK", name: "Duke Energy" }, { symbol: "SO", name: "Southern Co." }],
};

const YH = { "User-Agent": "Mozilla/5.0 (compatible; market-data/1.0)" };

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

async function fetchYahooCandles(symbol: string): Promise<YahooCandle | null> {
  try {
    const res = await fetch(
      `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?range=1y&interval=1d`,
      { next: { revalidate: 3600 }, headers: YH }
    );
    if (!res.ok) return null;
    const json = await res.json();
    const result = json?.chart?.result?.[0];
    if (!result?.timestamp?.length) return null;
    const closes: (number | null)[] =
      result.indicators?.adjclose?.[0]?.adjclose ??
      result.indicators?.quote?.[0]?.close ?? [];
    const t: number[] = [];
    const c: number[] = [];
    (result.timestamp as number[]).forEach((ts: number, i: number) => {
      const price = closes[i];
      if (price != null && price > 0) { t.push(ts); c.push(price); }
    });
    return t.length ? { t, c } : null;
  } catch { return null; }
}

function priceAt(candle: YahooCandle, targetUnix: number): number | null {
  let best: number | null = null;
  let bestDiff = Infinity;
  for (let i = 0; i < candle.t.length; i++) {
    if (candle.t[i] > targetUnix + 86400) continue;
    const diff = targetUnix - candle.t[i];
    if (diff >= 0 && diff < bestDiff) { bestDiff = diff; best = candle.c[i]; }
  }
  return best;
}

function computePcts(candle: YahooCandle): SectorPcts {
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
  const apiKey = process.env.FINNHUB_API_KEY;
  if (!apiKey) return fail("FINNHUB_API_KEY not set", "NO_API_KEY", 500, requestId);

  const sectorKeys = Object.keys(SECTOR_STOCKS);

  const etfCandles = await Promise.allSettled(
    sectorKeys.map((key) => fetchYahooCandles(SECTOR_ETFS[key]))
  );

  const allStocks = sectorKeys.flatMap((key) =>
    SECTOR_STOCKS[key].map((s) => ({ ...s, sectorKey: key }))
  );
  const stockSettled = await Promise.allSettled(
    allStocks.map(({ symbol }) => fetchQuote(symbol, apiKey))
  );

  const sectors: SectorData[] = sectorKeys.map((key, i) => {
    const candle = etfCandles[i].status === "fulfilled" ? etfCandles[i].value : null;
    const pcts: SectorPcts = candle
      ? computePcts(candle)
      : { d1: 0, w1: 0, m1: 0, m3: 0, ytd: 0 };

    const stocks: SectorStock[] = SECTOR_STOCKS[key]
      .map((def) => {
        const idx = allStocks.findIndex((s) => s.symbol === def.symbol && s.sectorKey === key);
        const q = idx >= 0 && stockSettled[idx].status === "fulfilled" ? stockSettled[idx].value : null;
        if (!q) return null;
        return { symbol: def.symbol, name: def.name, price: q.c ?? 0, pct: q.dp ?? 0, change: q.d ?? 0, up: (q.d ?? 0) >= 0 };
      })
      .filter((s): s is SectorStock => s !== null);

    return { name: SECTOR_NAME_MAP[key] ?? key, pcts, stocks };
  });

  sectors.sort((a, b) => b.pcts.d1 - a.pcts.d1);
  const data: MarketSectorsData = { sectors, generatedAt: new Date().toISOString() };
  return ok(data, requestId);
}
