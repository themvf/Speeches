import { createRequestId, ok } from "@/lib/server/api-utils";
import type {
  FearGreedLabel,
  IndexPcts,
  MarketIndexQuote,
  MarketOverviewData,
  VixQuote,
} from "@/lib/server/types";
import { fetchYahooCandles, fetchYahooQuote } from "@/lib/server/yahoo";
import { buildBreadthPair, summarizeBreadth } from "@/lib/market-breadth";

export const runtime = "nodejs";
export const revalidate = 60;

const US_INDICES = [
  { symbol: "^GSPC", name: "S&P 500" },
  { symbol: "^DJI",  name: "Dow Jones" },
  { symbol: "^IXIC", name: "NASDAQ" },
  { symbol: "^RUT",  name: "Russell 2000" },
];

/**
 * Equal-weight vs cap-weight ETF pairs. Compared ETF-to-ETF rather than against
 * ^GSPC/^IXIC deliberately: ^IXIC is the Nasdaq COMPOSITE (~3,000 names) while
 * QQQE is equal-weight Nasdaq-100, so that comparison would be between two
 * different universes rather than two weightings of one.
 */
const BREADTH_PAIRS = [
  { id: "sp500", label: "S&P 500", capSymbol: "SPY", equalSymbol: "RSP" },
  { id: "nasdaq100", label: "Nasdaq 100", capSymbol: "QQQ", equalSymbol: "QQQE" },
] as const;

const GLOBAL_INDICES = [
  { symbol: "^FTSE",  name: "FTSE 100" },
  { symbol: "^GDAXI", name: "DAX" },
  { symbol: "^N225",  name: "Nikkei 225" },
  { symbol: "^HSI",   name: "Hang Seng" },
  { symbol: "^AXJO",  name: "ASX 200" },
  { symbol: "^FCHI",  name: "CAC 40" },
];

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

function computeIndexPcts(candle: { t: number[]; c: number[] }, d1: number, current: number): IndexPcts {
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

  const breadthSymbols = BREADTH_PAIRS.flatMap((pair) => [pair.capSymbol, pair.equalSymbol]);

  const [usQuotes, usCandles, vixQuote, globalQuotes, breadthQuotes] = await Promise.all([
    Promise.allSettled(US_INDICES.map(({ symbol }) => fetchYahooQuote(symbol, 60))),
    Promise.allSettled(US_INDICES.map(({ symbol }) => fetchYahooCandles(symbol))),
    fetchYahooQuote("^VIX", 60),
    Promise.allSettled(GLOBAL_INDICES.map(({ symbol }) => fetchYahooQuote(symbol, 60))),
    Promise.allSettled(breadthSymbols.map((symbol) => fetchYahooQuote(symbol, 60))),
  ]);

  const indices = US_INDICES.map(({ symbol, name }, i) => {
    const q = usQuotes[i].status === "fulfilled" ? usQuotes[i].value : null;
    const c = usCandles[i].status === "fulfilled" ? usCandles[i].value : null;
    const price  = q?.price ?? 0;
    const d1     = q?.pct ?? 0;
    const pcts   = c && price > 0 ? computeIndexPcts(c, d1, price) : { d1, w1: 0, m1: 0, ytd: 0 };
    const sparkline: number[] = c ? c.c.slice(-30) : [];
    return {
      symbol, name, price,
      change: q?.change ?? 0,
      pct: d1,
      pcts,
      sparkline,
      up: (q?.change ?? 0) >= 0,
      status: q?.status ?? "CLOSED" as const,
    };
  }).filter((q) => q.price > 0);

  let vix: VixQuote | null = null;
  if (vixQuote?.price) {
    const v = vixQuote.price;
    vix = {
      value: v,
      change: vixQuote.change,
      pct: vixQuote.pct,
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
      pct: q.pct,
      pcts: { d1: q.pct, w1: 0, m1: 0, ytd: 0 },
      sparkline: [] as number[],
      up: q.change >= 0,
      status: q.status,
    };
  }).filter((q): q is MarketIndexQuote => q !== null && q.price > 0);

  // Breadth: a pair is dropped entirely if either leg failed, rather than
  // shown half-computed - half a spread is not a spread.
  const breadthPct = new Map<string, number>();
  breadthSymbols.forEach((symbol, index) => {
    const settled = breadthQuotes[index];
    const quote = settled.status === "fulfilled" ? settled.value : null;
    if (quote && Number.isFinite(quote.pct)) breadthPct.set(symbol, quote.pct);
  });

  const pairs = BREADTH_PAIRS.flatMap((pair) => {
    const built = buildBreadthPair({
      id: pair.id,
      label: pair.label,
      capSymbol: pair.capSymbol,
      capPct: breadthPct.get(pair.capSymbol),
      equalSymbol: pair.equalSymbol,
      equalPct: breadthPct.get(pair.equalSymbol),
    });
    return built ? [built] : [];
  });

  // Small vs large rides on indices already fetched above - no extra request.
  const largePct = indices.find((index) => index.symbol === "^GSPC")?.pct;
  const smallPct = indices.find((index) => index.symbol === "^RUT")?.pct;
  const smallVsLarge =
    typeof largePct === "number" && typeof smallPct === "number"
      ? { smallPct, largePct, spreadPp: smallPct - largePct }
      : null;

  const breadth = pairs.length
    ? { pairs, summary: summarizeBreadth(pairs), smallVsLarge }
    : null;

  const data: MarketOverviewData = { indices, vix, globalIndices, breadth, generatedAt: new Date().toISOString() };
  return ok(data, requestId);
}
