import { createRequestId, ok } from "@/lib/server/api-utils";
import type {
  FearGreedLabel,
  IndexPcts,
  MarketIndexQuote,
  MarketOverviewData,
  MarketStatus,
  VixQuote,
} from "@/lib/server/types";

export const runtime = "nodejs";
export const revalidate = 60;

type YahooCandle = { t: number[]; c: number[] };

const US_INDICES = [
  { symbol: "^GSPC", name: "S&P 500" },
  { symbol: "^DJI",  name: "Dow Jones" },
  { symbol: "^IXIC", name: "NASDAQ" },
  { symbol: "^RUT",  name: "Russell 2000" },
];

const GLOBAL_INDICES = [
  { symbol: "^FTSE",  name: "FTSE 100" },
  { symbol: "^GDAXI", name: "DAX" },
  { symbol: "^N225",  name: "Nikkei 225" },
  { symbol: "^HSI",   name: "Hang Seng" },
  { symbol: "^AXJO",  name: "ASX 200" },
  { symbol: "^FCHI",  name: "CAC 40" },
];

const YH = { "User-Agent": "Mozilla/5.0 (compatible; market-data/1.0)" };

function mapMarketState(state?: string): MarketStatus {
  if (!state) return "CLOSED";
  if (state === "REGULAR") return "OPEN";
  if (state.startsWith("PRE")) return "PRE";
  if (state === "POST" || state === "POSTPOST") return "AFTER";
  return "CLOSED";
}

async function fetchYahooQuote(symbol: string, rv = 60): Promise<{
  price: number; change: number; changePct: number; status: MarketStatus;
} | null> {
  try {
    const res = await fetch(
      `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?range=1d&interval=1d`,
      { next: { revalidate: rv }, headers: YH }
    );
    if (!res.ok) return null;
    const json = await res.json();
    const meta = json?.chart?.result?.[0]?.meta;
    if (!meta?.regularMarketPrice) return null;
    const price = meta.regularMarketPrice as number;
    const prev  = (meta.chartPreviousClose ?? meta.previousClose ?? price) as number;
    const change    = (meta.regularMarketChange    ?? (price - prev)) as number;
    const changePct = (meta.regularMarketChangePercent ?? (prev ? ((price - prev) / prev) * 100 : 0)) as number;
    return { price, change, changePct, status: mapMarketState(meta.marketState as string) };
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

function computeIndexPcts(candle: YahooCandle, d1: number, current: number): IndexPcts {
  const now = Date.now() / 1000;
  const ytdStart = new Date(new Date().getFullYear(), 0, 1).getTime() / 1000;
  const pct = (ref: number | null) =>
    ref && ref > 0 ? ((current - ref) / ref) * 100 : 0;
  return {
    d1,
    w1:  pct(priceAt(candle, now - 7  * 86400)),
    m1:  pct(priceAt(candle, now - 30 * 86400)),
    ytd: pct(priceAt(candle, ytdStart)),
  };
}

function fearGreedLabel(vix: number): FearGreedLabel {
  if (vix < 15) return "GREED";
  if (vix < 25) return "CALM";
  if (vix < 35) return "CONCERN";
  return "PANIC";
}

export async function GET() {
  const requestId = createRequestId();

  const [usQuotes, usCandles, vixQuote, globalQuotes] = await Promise.all([
    Promise.allSettled(US_INDICES.map(({ symbol }) => fetchYahooQuote(symbol))),
    Promise.allSettled(US_INDICES.map(({ symbol }) => fetchYahooCandles(symbol))),
    fetchYahooQuote("^VIX"),
    Promise.allSettled(GLOBAL_INDICES.map(({ symbol }) => fetchYahooQuote(symbol))),
  ]);

  const indices = US_INDICES.map(({ symbol, name }, i) => {
    const q = usQuotes[i].status === "fulfilled" ? usQuotes[i].value : null;
    const c = usCandles[i].status === "fulfilled" ? usCandles[i].value : null;
    const price  = q?.price ?? 0;
    const d1     = q?.changePct ?? 0;
    const pcts   = c && price > 0 ? computeIndexPcts(c, d1, price) : { d1, w1: 0, m1: 0, ytd: 0 };
    const sparkline: number[] = c ? c.c.slice(-30) : [];
    return {
      symbol, name, price,
      change: q?.change ?? 0,
      pct: d1,
      pcts,
      sparkline,
      up: (q?.change ?? 0) >= 0,
      status: q?.status ?? ("CLOSED" as MarketStatus),
    };
  }).filter((q) => q.price > 0);

  let vix: VixQuote | null = null;
  if (vixQuote?.price) {
    const v = vixQuote.price;
    vix = {
      value: v,
      change: vixQuote.change,
      pct: vixQuote.changePct,
      label: fearGreedLabel(v),
      gradientPct: Math.min(100, Math.max(0, ((v - 10) / 35) * 100)),
    };
  }

  const globalIndices = GLOBAL_INDICES.map(({ symbol, name }, i) => {
    const q = globalQuotes[i].status === "fulfilled" ? globalQuotes[i].value : null;
    if (!q) return null;
    return {
      symbol, name,
      price: q.price,
      change: q.change,
      pct: q.changePct,
      pcts: { d1: q.changePct, w1: 0, m1: 0, ytd: 0 },
      sparkline: [] as number[],
      up: q.change >= 0,
      status: q.status,
    };
  }).filter((q): q is MarketIndexQuote => q !== null && q.price > 0);

  const data: MarketOverviewData = { indices, vix, globalIndices, generatedAt: new Date().toISOString() };
  return ok(data, requestId);
}
