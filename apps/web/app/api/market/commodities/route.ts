import { createRequestId, ok } from "@/lib/server/api-utils";
import type { CommodityCategory, CommodityQuote, MarketCommoditiesData } from "@/lib/server/types";

export const runtime = "nodejs";
export const revalidate = 120;

const COMMODITIES: { symbol: string; name: string; category: CommodityCategory }[] = [
  { symbol: "GC=F",  name: "Gold",        category: "metals"      },
  { symbol: "SI=F",  name: "Silver",      category: "metals"      },
  { symbol: "HG=F",  name: "Copper",      category: "metals"      },
  { symbol: "PL=F",  name: "Platinum",    category: "metals"      },
  { symbol: "CL=F",  name: "Oil (WTI)",   category: "energy"      },
  { symbol: "NG=F",  name: "Natural Gas", category: "energy"      },
  { symbol: "ZW=F",  name: "Wheat",       category: "agriculture" },
  { symbol: "ZC=F",  name: "Corn",        category: "agriculture" },
];

const YH = { "User-Agent": "Mozilla/5.0 (compatible; market-data/1.0)" };

async function fetchYahooQuote(symbol: string): Promise<{
  price: number; change: number; changePct: number;
} | null> {
  try {
    const res = await fetch(
      `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?range=1d&interval=1d`,
      { next: { revalidate: 120 }, headers: YH }
    );
    if (!res.ok) return null;
    const json = await res.json();
    const meta = json?.chart?.result?.[0]?.meta;
    if (!meta?.regularMarketPrice) return null;
    const price = meta.regularMarketPrice as number;
    const prev  = (meta.chartPreviousClose ?? meta.previousClose ?? price) as number;
    const change    = (meta.regularMarketChange    ?? (price - prev)) as number;
    const changePct = (meta.regularMarketChangePercent ?? (prev ? ((price - prev) / prev) * 100 : 0)) as number;
    return { price, change, changePct };
  } catch { return null; }
}

export async function GET() {
  const requestId = createRequestId();

  const settled = await Promise.allSettled(
    COMMODITIES.map(({ symbol }) => fetchYahooQuote(symbol))
  );

  const commodities = COMMODITIES.map(({ symbol, name, category }, i) => {
    const q = settled[i].status === "fulfilled" ? settled[i].value : null;
    if (!q) return null;
    return {
      symbol, name, category,
      price:  q.price,
      change: q.change,
      pct:    q.changePct,
      up:     q.change >= 0,
    };
  }).filter((c): c is CommodityQuote => c !== null && c.price > 0);

  const data: MarketCommoditiesData = { commodities, generatedAt: new Date().toISOString() };
  return ok(data, requestId);
}
