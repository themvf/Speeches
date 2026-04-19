import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import type {
  FearGreedLabel,
  MarketIndexQuote,
  MarketOverviewData,
  MarketStatus,
  VixQuote,
} from "@/lib/server/types";

export const runtime = "nodejs";
export const revalidate = 60;

type FinnhubQuote = { c: number | null; d: number | null; dp: number | null; t?: number };

const US_INDICES = [
  { symbol: "SPY", name: "S&P 500" },
  { symbol: "DIA", name: "Dow Jones" },
  { symbol: "QQQ", name: "NASDAQ" },
  { symbol: "IWM", name: "Russell 2000" },
];

// Free-tier ETF proxies for global indices
const GLOBAL_INDICES = [
  { symbol: "EWU", name: "FTSE 100" },
  { symbol: "EWG", name: "DAX" },
  { symbol: "EWJ", name: "Nikkei 225" },
  { symbol: "EWH", name: "Hang Seng" },
  { symbol: "EWA", name: "ASX 200" },
  { symbol: "EWQ", name: "CAC 40" },
];

async function fetchQuote(symbol: string, apiKey: string): Promise<FinnhubQuote | null> {
  try {
    const res = await fetch(
      `https://finnhub.io/api/v1/quote?symbol=${encodeURIComponent(symbol)}&token=${apiKey}`,
      { next: { revalidate: 60 } }
    );
    if (!res.ok) return null;
    const data: FinnhubQuote = await res.json();
    return data.c != null ? data : null;
  } catch {
    return null;
  }
}

function deriveStatus(q: FinnhubQuote): MarketStatus {
  if (!q.t) return "CLOSED";
  const ageSec = Date.now() / 1000 - q.t;
  return ageSec < 1200 ? "OPEN" : "CLOSED";
}

function toIndexQuote(symbol: string, name: string, q: FinnhubQuote): MarketIndexQuote {
  return {
    symbol,
    name,
    price: q.c ?? 0,
    change: q.d ?? 0,
    pct: q.dp ?? 0,
    up: (q.d ?? 0) >= 0,
    status: deriveStatus(q),
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
  const apiKey = process.env.FINNHUB_API_KEY;
  if (!apiKey) return fail("FINNHUB_API_KEY not set", "NO_API_KEY", 500, requestId);

  const allSymbols = [...US_INDICES, { symbol: "^VIX", name: "VIX" }, ...GLOBAL_INDICES];

  const settled = await Promise.allSettled(
    allSymbols.map(({ symbol }) => fetchQuote(symbol, apiKey))
  );

  const quotes = settled.map((r, i) => ({
    ...allSymbols[i],
    q: r.status === "fulfilled" ? r.value : null,
  }));

  const indices: MarketIndexQuote[] = quotes
    .slice(0, 4)
    .filter((q) => q.q != null)
    .map(({ symbol, name, q }) => toIndexQuote(symbol, name, q!));

  const vixRaw = quotes[4];
  let vix: VixQuote | null = null;
  if (vixRaw.q) {
    const v = vixRaw.q.c ?? 20;
    vix = {
      value: v,
      change: vixRaw.q.d ?? 0,
      pct: vixRaw.q.dp ?? 0,
      label: fearGreedLabel(v),
      gradientPct: Math.min(100, Math.max(0, ((v - 10) / 35) * 100)),
    };
  }

  const globalIndices: MarketIndexQuote[] = quotes
    .slice(5)
    .filter((q) => q.q != null)
    .map(({ symbol, name, q }) => toIndexQuote(symbol, name, q!));

  const data: MarketOverviewData = {
    indices,
    vix,
    globalIndices,
    generatedAt: new Date().toISOString(),
  };

  return ok(data, requestId);
}
