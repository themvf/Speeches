import type { MarketStatus } from "@/lib/server/types";

export type { MarketStatus };

export type YahooQuote = {
  price: number;
  fiftyTwoWeekLow: number | null;
  fiftyTwoWeekHigh: number | null;
  change: number;
  pct: number;
  name: string;
  status: MarketStatus;
  asOf: string;
  currency: string;
  provider: "Yahoo Finance";
};

export type YahooCandle = { t: number[]; c: number[] };
export type YahooOhlcvCandle = { t: number; open: number; high: number; low: number; close: number; volume: number | null };

export const YH = { "User-Agent": "Mozilla/5.0 (compatible; market-data/1.0)" };

export function mapMarketState(state?: string): MarketStatus {
  if (!state) return "CLOSED";
  if (state === "REGULAR") return "OPEN";
  if (state.startsWith("PRE")) return "PRE";
  if (state === "POST" || state === "POSTPOST") return "AFTER";
  return "CLOSED";
}

export async function fetchYahooQuote(symbol: string, revalidate = 300): Promise<YahooQuote | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8_000);
  try {
    const res = await fetch(
      `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?range=1d&interval=1d`,
      { next: { revalidate }, headers: YH, signal: controller.signal }
    );
    if (!res.ok) return null;
    const json = await res.json();
    const meta = json?.chart?.result?.[0]?.meta;
    if (!meta?.regularMarketPrice) return null;
    const price = meta.regularMarketPrice as number;
    const prev = (meta.chartPreviousClose ?? meta.previousClose ?? price) as number;
    const change = (meta.regularMarketChange ?? (price - prev)) as number;
    const pct = (meta.regularMarketChangePercent ?? (prev ? ((price - prev) / prev) * 100 : 0)) as number;
    const marketTime = Number(meta.regularMarketTime);
    const fiftyTwoWeekLow = Number(meta.fiftyTwoWeekLow);
    const fiftyTwoWeekHigh = Number(meta.fiftyTwoWeekHigh);
    return {
      price,
      fiftyTwoWeekLow: Number.isFinite(fiftyTwoWeekLow) && fiftyTwoWeekLow > 0 ? fiftyTwoWeekLow : null,
      fiftyTwoWeekHigh: Number.isFinite(fiftyTwoWeekHigh) && fiftyTwoWeekHigh > 0 ? fiftyTwoWeekHigh : null,
      change,
      pct,
      name: String(meta.longName ?? meta.shortName ?? symbol),
      status: mapMarketState(meta.marketState as string),
      asOf: Number.isFinite(marketTime) && marketTime > 0
        ? new Date(marketTime * 1000).toISOString()
        : new Date().toISOString(),
      currency: String(meta.currency ?? "USD"),
      provider: "Yahoo Finance",
    };
  } catch { return null; } finally { clearTimeout(timer); }
}

export async function fetchYahooCandles(symbol: string, revalidate = 3600): Promise<YahooCandle | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8_000);
  try {
    const res = await fetch(
      `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?range=1y&interval=1d`,
      { next: { revalidate }, headers: YH, signal: controller.signal }
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
  } catch { return null; } finally { clearTimeout(timer); }
}

export function normalizeYahooOhlcv(result: unknown): YahooOhlcvCandle[] {
  if (!result || typeof result !== "object") return [];
  const row = result as { timestamp?: unknown; indicators?: { quote?: unknown } };
  const timestamps = Array.isArray(row.timestamp) ? row.timestamp : [];
  const quotes = Array.isArray(row.indicators?.quote) ? row.indicators?.quote[0] : null;
  if (!quotes || typeof quotes !== "object") return [];
  const values = quotes as Record<string, unknown>;
  const opens = Array.isArray(values.open) ? values.open : [];
  const highs = Array.isArray(values.high) ? values.high : [];
  const lows = Array.isArray(values.low) ? values.low : [];
  const closes = Array.isArray(values.close) ? values.close : [];
  const volumes = Array.isArray(values.volume) ? values.volume : [];
  const candles: YahooOhlcvCandle[] = [];
  timestamps.forEach((rawTimestamp, index) => {
    const t = Number(rawTimestamp);
    const open = Number(opens[index]);
    const high = Number(highs[index]);
    const low = Number(lows[index]);
    const close = Number(closes[index]);
    const rawVolume = volumes[index];
    const volume = rawVolume == null ? null : Number(rawVolume);
    if (![t, open, high, low, close].every(Number.isFinite)) return;
    if (t <= 0 || open <= 0 || high <= 0 || low <= 0 || close <= 0 || high < low) return;
    candles.push({ t, open, high, low, close, volume: volume != null && Number.isFinite(volume) && volume >= 0 ? volume : null });
  });
  return candles;
}

export async function fetchYahooOhlcv(symbol: string, revalidate = 300): Promise<YahooOhlcvCandle[] | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8_000);
  try {
    const res = await fetch(
      `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?range=6mo&interval=1d&events=div%2Csplits`,
      { next: { revalidate }, headers: YH, signal: controller.signal }
    );
    if (!res.ok) return null;
    const json = await res.json();
    const candles = normalizeYahooOhlcv(json?.chart?.result?.[0]);
    return candles.length ? candles : null;
  } catch { return null; } finally { clearTimeout(timer); }
}
