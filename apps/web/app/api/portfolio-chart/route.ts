import { NextResponse } from "next/server";
import { normalizeSymbol } from "@/lib/portfolio-calculations";
import { getClientIp, getPortfolioChartLimiter, isRateLimited } from "@/lib/server/rate-limit";
import { fetchYahooOhlcv } from "@/lib/server/yahoo";
import type { PortfolioCandle } from "@/lib/portfolio-types";

export const dynamic = "force-dynamic";

export async function GET(req: Request): Promise<NextResponse> {
  if (await isRateLimited(getPortfolioChartLimiter(), getClientIp(req.headers))) {
    return NextResponse.json({ ok: false, error: "Too many chart requests. Try again shortly." }, { status: 429 });
  }
  const symbol = normalizeSymbol(new URL(req.url).searchParams.get("symbol"));
  if (!symbol) return NextResponse.json({ ok: false, error: "A valid ticker is required." }, { status: 400 });
  const candles = await fetchYahooOhlcv(symbol, 300);
  if (!candles) return NextResponse.json({ ok: false, error: `Chart data is unavailable for ${symbol}.` }, { status: 404 });
  return NextResponse.json({
    ok: true,
    symbol,
    data: candles satisfies PortfolioCandle[],
    generatedAt: new Date().toISOString(),
  }, { headers: { "Cache-Control": "public, s-maxage=300, stale-while-revalidate=3600" } });
}
