import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import type { CryptoCoin, MarketCryptoData } from "@/lib/server/types";

export const runtime = "nodejs";
export const revalidate = 120;

interface CoinGeckoMarket {
  id: string;
  symbol: string;
  name: string;
  current_price: number;
  price_change_percentage_24h: number | null;
  market_cap: number;
  total_volume: number;
  market_cap_rank: number;
}

export async function GET(request: Request) {
  const requestId = createRequestId();
  const url = new URL(request.url);
  const limit = Math.min(50, Math.max(1, parseInt(url.searchParams.get("limit") ?? "20", 10)));

  try {
    const res = await fetch(
      `https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=${limit}&page=1&sparkline=false`,
      { next: { revalidate: 120 } }
    );

    if (!res.ok) {
      const text = await res.text();
      return fail(`CoinGecko error: ${res.status} ${text.slice(0, 200)}`, "COINGECKO_ERROR", 502, requestId);
    }

    const raw: CoinGeckoMarket[] = await res.json();

    const coins: CryptoCoin[] = raw.map((c, i) => ({
      rank: c.market_cap_rank ?? i + 1,
      id: c.id,
      symbol: c.symbol.toUpperCase(),
      name: c.name,
      price: c.current_price,
      pct24h: c.price_change_percentage_24h ?? 0,
      marketCap: c.market_cap,
      volume24h: c.total_volume,
      up: (c.price_change_percentage_24h ?? 0) >= 0,
    }));

    const data: MarketCryptoData = { coins, generatedAt: new Date().toISOString() };
    return ok(data, requestId);
  } catch (err) {
    return fail(
      `Failed to fetch crypto data: ${err instanceof Error ? err.message : "Unknown error"}`,
      "CRYPTO_FETCH_FAILED",
      500,
      requestId
    );
  }
}
