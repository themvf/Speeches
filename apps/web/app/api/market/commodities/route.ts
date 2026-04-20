import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import type { CommodityCategory, CommodityQuote, MarketCommoditiesData } from "@/lib/server/types";

export const runtime = "nodejs";
export const revalidate = 120;

type FinnhubQuote = { c: number | null; d: number | null; dp: number | null };

const COMMODITIES: { symbol: string; name: string; category: CommodityCategory }[] = [
  { symbol: "GLD",  name: "Gold",        category: "metals"      },
  { symbol: "SLV",  name: "Silver",      category: "metals"      },
  { symbol: "CPER", name: "Copper",      category: "metals"      },
  { symbol: "PPLT", name: "Platinum",    category: "metals"      },
  { symbol: "USO",  name: "Oil (WTI)",   category: "energy"      },
  { symbol: "UNG",  name: "Natural Gas", category: "energy"      },
  { symbol: "XLE",  name: "Energy ETF",  category: "energy"      },
  { symbol: "WEAT", name: "Wheat",       category: "agriculture" },
  { symbol: "CORN", name: "Corn",        category: "agriculture" },
];

async function fetchQuote(symbol: string, apiKey: string): Promise<FinnhubQuote | null> {
  try {
    const res = await fetch(
      `https://finnhub.io/api/v1/quote?symbol=${encodeURIComponent(symbol)}&token=${apiKey}`,
      { next: { revalidate: 120 } }
    );
    if (!res.ok) return null;
    const data: FinnhubQuote = await res.json();
    return data.c != null ? data : null;
  } catch { return null; }
}

export async function GET() {
  const requestId = createRequestId();
  const apiKey = process.env.FINNHUB_API_KEY;
  if (!apiKey) return fail("FINNHUB_API_KEY not set", "NO_API_KEY", 500, requestId);

  const settled = await Promise.allSettled(
    COMMODITIES.map(({ symbol }) => fetchQuote(symbol, apiKey))
  );

  const commodities: CommodityQuote[] = COMMODITIES.map(({ symbol, name, category }, i) => {
    const q = settled[i].status === "fulfilled" ? settled[i].value : null;
    if (!q) return null;
    return {
      symbol, name, category,
      price:  q.c ?? 0,
      change: q.d ?? 0,
      pct:    q.dp ?? 0,
      up:     (q.d ?? 0) >= 0,
    };
  }).filter((c): c is CommodityQuote => c !== null && c.price > 0);

  const data: MarketCommoditiesData = { commodities, generatedAt: new Date().toISOString() };
  return ok(data, requestId);
}
