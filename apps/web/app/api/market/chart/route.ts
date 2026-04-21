import { createRequestId, fail, ok } from "@/lib/server/api-utils";

export const runtime = "nodejs";

const YH = { "User-Agent": "Mozilla/5.0 (compatible; market-data/1.0)" };

type ChartPoint = { t: number; c: number };

async function fetchYahooHistory(symbol: string): Promise<ChartPoint[] | null> {
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
    const points: ChartPoint[] = [];
    (result.timestamp as number[]).forEach((ts: number, i: number) => {
      const c = closes[i];
      if (c != null && c > 0) points.push({ t: ts, c });
    });
    return points.length ? points : null;
  } catch { return null; }
}

async function fetchCryptoHistory(id: string): Promise<ChartPoint[] | null> {
  try {
    const res = await fetch(
      `https://api.coingecko.com/api/v3/coins/${encodeURIComponent(id)}/market_chart?vs_currency=usd&days=365`,
      { next: { revalidate: 3600 } }
    );
    if (!res.ok) return null;
    const json = await res.json();
    const raw: [number, number][] = json?.prices ?? [];
    // CoinGecko returns hourly for <90 days, daily for longer — downsample to daily
    const daily: ChartPoint[] = [];
    let lastDay = -1;
    for (const [ms, c] of raw) {
      const day = Math.floor(ms / 1000 / 86400);
      if (day !== lastDay) { daily.push({ t: Math.floor(ms / 1000), c }); lastDay = day; }
    }
    return daily.length ? daily : null;
  } catch { return null; }
}

export async function GET(request: Request) {
  const requestId = createRequestId();
  const { searchParams } = new URL(request.url);
  const type = searchParams.get("type") ?? "yahoo";

  if (type === "crypto") {
    const id = searchParams.get("id");
    if (!id) return fail("Missing id", "MISSING_ID", 400, requestId);
    const prices = await fetchCryptoHistory(id);
    if (!prices) return fail("No chart data", "NO_DATA", 404, requestId);
    return ok({ prices }, requestId);
  }

  const symbol = searchParams.get("symbol");
  if (!symbol) return fail("Missing symbol", "MISSING_SYMBOL", 400, requestId);
  const prices = await fetchYahooHistory(symbol);
  if (!prices) return fail("No chart data", "NO_DATA", 404, requestId);
  return ok({ prices }, requestId);
}
