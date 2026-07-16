import { createRequestId, ok } from "@/lib/server/api-utils";
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

const SECTOR_STOCKS: Record<string, { symbol: string; name: string }[]> = {
  "Technology": [
    { symbol: "NVDA", name: "Nvidia" }, { symbol: "AAPL", name: "Apple" },
    { symbol: "MSFT", name: "Microsoft" }, { symbol: "AVGO", name: "Broadcom" },
    { symbol: "AMD", name: "AMD" }, { symbol: "MU", name: "Micron" },
    { symbol: "INTC", name: "Intel" }, { symbol: "CSCO", name: "Cisco" },
    { symbol: "AMAT", name: "Applied Materials" }, { symbol: "LRCX", name: "Lam Research" },
  ],
  "Communication Services": [
    { symbol: "GOOGL", name: "Alphabet" }, { symbol: "META", name: "Meta" },
    { symbol: "NFLX", name: "Netflix" }, { symbol: "TMUS", name: "T-Mobile" },
    { symbol: "VZ", name: "Verizon" }, { symbol: "T", name: "AT&T" },
    { symbol: "DIS", name: "Disney" }, { symbol: "CMCSA", name: "Comcast" },
    { symbol: "CHTR", name: "Charter" }, { symbol: "EA", name: "Electronic Arts" },
  ],
  "Consumer Cyclical": [
    { symbol: "AMZN", name: "Amazon" }, { symbol: "TSLA", name: "Tesla" },
    { symbol: "HD", name: "Home Depot" }, { symbol: "MCD", name: "McDonald's" },
    { symbol: "TJX", name: "TJX" }, { symbol: "BKNG", name: "Booking Holdings" },
    { symbol: "SBUX", name: "Starbucks" }, { symbol: "LOW", name: "Lowe's" },
    { symbol: "MAR", name: "Marriott" }, { symbol: "HLT", name: "Hilton" },
  ],
  "Consumer Defensive": [
    { symbol: "WMT", name: "Walmart" }, { symbol: "COST", name: "Costco" },
    { symbol: "PG", name: "P&G" }, { symbol: "KO", name: "Coca-Cola" },
    { symbol: "PM", name: "Philip Morris" }, { symbol: "PEP", name: "PepsiCo" },
    { symbol: "MDLZ", name: "Mondelez" }, { symbol: "MO", name: "Altria" },
    { symbol: "CL", name: "Colgate-Palmolive" }, { symbol: "TGT", name: "Target" },
  ],
  "Energy": [
    { symbol: "XOM", name: "Exxon Mobil" }, { symbol: "CVX", name: "Chevron" },
    { symbol: "COP", name: "ConocoPhillips" }, { symbol: "EOG", name: "EOG Resources" },
    { symbol: "SLB", name: "SLB" }, { symbol: "WMB", name: "Williams" },
    { symbol: "MPC", name: "Marathon Petroleum" }, { symbol: "OKE", name: "ONEOK" },
    { symbol: "PSX", name: "Phillips 66" }, { symbol: "VLO", name: "Valero" },
  ],
  "Financial Services": [
    { symbol: "BRK-B", name: "Berkshire Hathaway" }, { symbol: "JPM", name: "JPMorgan" },
    { symbol: "V", name: "Visa" }, { symbol: "MA", name: "Mastercard" },
    { symbol: "BAC", name: "Bank of America" }, { symbol: "WFC", name: "Wells Fargo" },
    { symbol: "GS", name: "Goldman Sachs" }, { symbol: "MS", name: "Morgan Stanley" },
    { symbol: "AXP", name: "American Express" }, { symbol: "SPGI", name: "S&P Global" },
  ],
  "Healthcare": [
    { symbol: "LLY", name: "Eli Lilly" }, { symbol: "JNJ", name: "Johnson & Johnson" },
    { symbol: "ABBV", name: "AbbVie" }, { symbol: "UNH", name: "UnitedHealth" },
    { symbol: "MRK", name: "Merck" }, { symbol: "ABT", name: "Abbott" },
    { symbol: "TMO", name: "Thermo Fisher" }, { symbol: "ISRG", name: "Intuitive Surgical" },
    { symbol: "AMGN", name: "Amgen" }, { symbol: "DHR", name: "Danaher" },
  ],
  "Industrials": [
    { symbol: "GE", name: "GE Aerospace" }, { symbol: "CAT", name: "Caterpillar" },
    { symbol: "RTX", name: "RTX" }, { symbol: "UBER", name: "Uber" },
    { symbol: "UNP", name: "Union Pacific" }, { symbol: "HON", name: "Honeywell" },
    { symbol: "BA", name: "Boeing" }, { symbol: "ETN", name: "Eaton" },
    { symbol: "DE", name: "Deere" }, { symbol: "ADP", name: "ADP" },
  ],
  "Basic Materials": [
    { symbol: "LIN", name: "Linde" }, { symbol: "SHW", name: "Sherwin-Williams" },
    { symbol: "FCX", name: "Freeport-McMoRan" }, { symbol: "NEM", name: "Newmont" },
    { symbol: "ECL", name: "Ecolab" }, { symbol: "CTVA", name: "Corteva" },
    { symbol: "APD", name: "Air Products" }, { symbol: "VMC", name: "Vulcan Materials" },
    { symbol: "MLM", name: "Martin Marietta" }, { symbol: "NUE", name: "Nucor" },
  ],
  "Real Estate": [
    { symbol: "PLD", name: "Prologis" }, { symbol: "AMT", name: "American Tower" },
    { symbol: "EQIX", name: "Equinix" }, { symbol: "WELL", name: "Welltower" },
    { symbol: "SPG", name: "Simon Property" }, { symbol: "DLR", name: "Digital Realty" },
    { symbol: "O", name: "Realty Income" }, { symbol: "PSA", name: "Public Storage" },
    { symbol: "CBRE", name: "CBRE" }, { symbol: "CCI", name: "Crown Castle" },
  ],
  "Utilities": [
    { symbol: "NEE", name: "NextEra Energy" }, { symbol: "SO", name: "Southern Co." },
    { symbol: "DUK", name: "Duke Energy" }, { symbol: "CEG", name: "Constellation Energy" },
    { symbol: "AEP", name: "American Electric Power" }, { symbol: "SRE", name: "Sempra" },
    { symbol: "VST", name: "Vistra" }, { symbol: "D", name: "Dominion Energy" },
    { symbol: "EXC", name: "Exelon" }, { symbol: "PEG", name: "PSEG" },
  ],
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
  const sectorKeys = Object.keys(SECTOR_STOCKS);

  const etfCandles = await Promise.allSettled(
    sectorKeys.map((key) => fetchYahooCandles(SECTOR_ETFS[key]))
  );

  const allStocks = sectorKeys.flatMap((key) =>
    SECTOR_STOCKS[key].map((s) => ({ ...s, sectorKey: key }))
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

    const stocks: SectorStock[] = SECTOR_STOCKS[key]
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
