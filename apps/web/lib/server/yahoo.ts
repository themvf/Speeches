import type { MarketStatus } from "@/lib/server/types";

export type { MarketStatus };

export type YahooQuote = {
  price: number;
  change: number;
  pct: number;
  name: string;
  status: MarketStatus;
};

export type YahooCandle = { t: number[]; c: number[] };

export const YH = { "User-Agent": "Mozilla/5.0 (compatible; market-data/1.0)" };

export function mapMarketState(state?: string): MarketStatus {
  if (!state) return "CLOSED";
  if (state === "REGULAR") return "OPEN";
  if (state.startsWith("PRE")) return "PRE";
  if (state === "POST" || state === "POSTPOST") return "AFTER";
  return "CLOSED";
}

export async function fetchYahooQuote(symbol: string, revalidate = 300): Promise<YahooQuote | null> {
  try {
    const res = await fetch(
      `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?range=1d&interval=1d`,
      { next: { revalidate }, headers: YH }
    );
    if (!res.ok) return null;
    const json = await res.json();
    const meta = json?.chart?.result?.[0]?.meta;
    if (!meta?.regularMarketPrice) return null;
    const price = meta.regularMarketPrice as number;
    const prev = (meta.chartPreviousClose ?? meta.previousClose ?? price) as number;
    const change = (meta.regularMarketChange ?? (price - prev)) as number;
    const pct = (meta.regularMarketChangePercent ?? (prev ? ((price - prev) / prev) * 100 : 0)) as number;
    return {
      price,
      change,
      pct,
      name: String(meta.longName ?? meta.shortName ?? symbol),
      status: mapMarketState(meta.marketState as string),
    };
  } catch { return null; }
}

export async function fetchYahooCandles(symbol: string, revalidate = 3600): Promise<YahooCandle | null> {
  try {
    const res = await fetch(
      `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?range=1y&interval=1d`,
      { next: { revalidate }, headers: YH }
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
