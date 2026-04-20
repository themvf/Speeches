import { createRequestId, fail, ok } from "@/lib/server/api-utils";
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

type FinnhubQuote  = { c: number | null; d: number | null; dp: number | null; t?: number };
type FinnhubCandle = { c: number[]; t: number[]; s: string };

const US_INDICES = [
  { symbol: "SPY", name: "S&P 500" },
  { symbol: "DIA", name: "Dow Jones" },
  { symbol: "QQQ", name: "NASDAQ" },
  { symbol: "IWM", name: "Russell 2000" },
];

const GLOBAL_INDICES = [
  { symbol: "EWU", name: "FTSE 100" },
  { symbol: "EWG", name: "DAX" },
  { symbol: "EWJ", name: "Nikkei 225" },
  { symbol: "EWH", name: "Hang Seng" },
  { symbol: "EWA", name: "ASX 200" },
  { symbol: "EWQ", name: "CAC 40" },
];

async function fetchQuote(symbol: string, apiKey: string, rv = 60): Promise<FinnhubQuote | null> {
  try {
    const res = await fetch(
      `https://finnhub.io/api/v1/quote?symbol=${encodeURIComponent(symbol)}&token=${apiKey}`,
      { next: { revalidate: rv } }
    );
    if (!res.ok) return null;
    const data: FinnhubQuote = await res.json();
    return data.c != null ? data : null;
  } catch { return null; }
}

async function fetchCandles(symbol: string, apiKey: string): Promise<FinnhubCandle | null> {
  const to   = Math.floor(Date.now() / 1000);
  const from = to - 400 * 86400;
  try {
    const res = await fetch(
      `https://finnhub.io/api/v1/stock/candle?symbol=${encodeURIComponent(symbol)}&resolution=D&from=${from}&to=${to}&token=${apiKey}`,
      { next: { revalidate: 3600 } }
    );
    if (!res.ok) return null;
    const data: FinnhubCandle = await res.json();
    return data.s === "ok" && data.c?.length ? data : null;
  } catch { return null; }
}

function priceAt(candle: FinnhubCandle, targetUnix: number): number | null {
  let best: number | null = null;
  let bestDiff = Infinity;
  for (let i = 0; i < candle.t.length; i++) {
    if (candle.t[i] > targetUnix + 86400) continue;
    const diff = targetUnix - candle.t[i];
    if (diff >= 0 && diff < bestDiff) { bestDiff = diff; best = candle.c[i]; }
  }
  return best;
}

function computeIndexPcts(candle: FinnhubCandle, d1: number, current: number): IndexPcts {
  const now  = Date.now() / 1000;
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

function deriveStatus(q: FinnhubQuote): MarketStatus {
  if (!q.t) return "CLOSED";
  return (Date.now() / 1000 - q.t) < 1200 ? "OPEN" : "CLOSED";
}

function fearGreedLabel(vix: number): FearGreedLabel {
  if (vix < 15) return "GREED";
  if (vix < 25) return "CALM";
  if (vix < 35) return "CONCERN";
  return "PANIC";
}

export async function GET() {
  const requestId = createRequestId();
  const apiKey = process.env.FINNHUB_API_KEY;
  if (!apiKey) return fail("FINNHUB_API_KEY not set", "NO_API_KEY", 500, requestId);

  // Fetch US index quotes + candles + VIX + global quotes — all in parallel
  const [usQuotes, usCandles, vixQuote, globalQuotes] = await Promise.all([
    Promise.allSettled(US_INDICES.map(({ symbol }) => fetchQuote(symbol, apiKey))),
    Promise.allSettled(US_INDICES.map(({ symbol }) => fetchCandles(symbol, apiKey))),
    fetchQuote("^VIX", apiKey),
    Promise.allSettled(GLOBAL_INDICES.map(({ symbol }) => fetchQuote(symbol, apiKey))),
  ]);

  const indices = US_INDICES.map(({ symbol, name }, i) => {
    const q = usQuotes[i].status === "fulfilled" ? usQuotes[i].value : null;
    const c = usCandles[i].status === "fulfilled" ? usCandles[i].value : null;
    const price  = q?.c ?? 0;
    const d1     = q?.dp ?? 0;
    const pcts   = c && price > 0 ? computeIndexPcts(c, d1, price) : { d1, w1: 0, m1: 0, ytd: 0 };
    const sparkline: number[] = c ? c.c.slice(-30) : [];
    return {
      symbol, name, price,
      change: q?.d ?? 0,
      pct: d1,
      pcts,
      sparkline,
      up: (q?.d ?? 0) >= 0,
      status: q ? deriveStatus(q) : "CLOSED",
    };
  }).filter((q) => q.price > 0);

  let vix: VixQuote | null = null;
  if (vixQuote?.c) {
    const v = vixQuote.c;
    vix = {
      value: v,
      change: vixQuote.d ?? 0,
      pct: vixQuote.dp ?? 0,
      label: fearGreedLabel(v),
      gradientPct: Math.min(100, Math.max(0, ((v - 10) / 35) * 100)),
    };
  }

  const globalIndices = GLOBAL_INDICES.map(({ symbol, name }, i) => {
    const q = globalQuotes[i].status === "fulfilled" ? globalQuotes[i].value : null;
    if (!q) return null;
    return {
      symbol, name,
      price: q.c ?? 0,
      change: q.d ?? 0,
      pct: q.dp ?? 0,
      pcts: { d1: q.dp ?? 0, w1: 0, m1: 0, ytd: 0 },
      sparkline: [] as number[],
      up: (q.d ?? 0) >= 0,
      status: deriveStatus(q),
    };
  }).filter((q): q is MarketIndexQuote => q !== null && q.price > 0);

  const data: MarketOverviewData = { indices, vix, globalIndices, generatedAt: new Date().toISOString() };
  return ok(data, requestId);
}
