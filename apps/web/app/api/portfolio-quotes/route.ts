import { NextResponse } from "next/server";
import { normalizeSymbol } from "@/lib/portfolio-calculations";
import { fetchYahooQuote } from "@/lib/server/yahoo";
import { getClientIp, getPortfolioQuoteLimiter, isRateLimited } from "@/lib/server/rate-limit";
import type { PortfolioQuote } from "@/lib/portfolio-types";

export const dynamic = "force-dynamic";

export async function GET(req: Request): Promise<NextResponse> {
  if (await isRateLimited(getPortfolioQuoteLimiter(), getClientIp(req.headers))) {
    return NextResponse.json({ ok: false, error: "Too many quote refreshes. Try again shortly." }, { status: 429 });
  }
  const requested = new URL(req.url).searchParams.get("symbols")?.split(",") ?? [];
  const symbols = Array.from(new Set(requested.map(normalizeSymbol).filter(Boolean))).slice(0, 100);
  const headers = { "Cache-Control": "public, s-maxage=60, stale-while-revalidate=120" };
  if (!symbols.length) return NextResponse.json({ ok: true, data: [], missingSymbols: [], generatedAt: new Date().toISOString() }, { headers });
  const settled = await Promise.allSettled(symbols.map(async (symbol) => {
    const quote = await fetchYahooQuote(symbol, 60);
    return quote ? { symbol, ...quote } satisfies PortfolioQuote : null;
  }));
  const data = settled.flatMap((result) => result.status === "fulfilled" && result.value ? [result.value] : []);
  const priced = new Set(data.map((quote) => quote.symbol));
  const missingSymbols = symbols.filter((symbol) => !priced.has(symbol));
  return NextResponse.json({
    ok: true,
    data,
    requestedCount: symbols.length,
    pricedCount: data.length,
    missingSymbols,
    generatedAt: new Date().toISOString(),
  }, { headers });
}
