import { NextResponse } from "next/server";
import { loadUserPortfolios, saveUserPortfolios } from "@/lib/server/portfolio-store";
import { PORTFOLIO_SYNC_HEADER, normalizePortfolioSyncToken, portfolioWorkspaceId } from "@/lib/server/portfolio-sync";
import { getClientIp, getPortfolioSyncLimiter, isRateLimited } from "@/lib/server/rate-limit";

export const dynamic = "force-dynamic";

function workspaceId(req: Request): string | null {
  const token = normalizePortfolioSyncToken(req.headers.get(PORTFOLIO_SYNC_HEADER));
  return token ? portfolioWorkspaceId(token) : null;
}

export async function GET(req: Request): Promise<NextResponse> {
  if (await isRateLimited(getPortfolioSyncLimiter(), getClientIp(req.headers))) {
    return NextResponse.json({ ok: false, error: "Too many sync requests. Try again shortly." }, { status: 429 });
  }
  const userId = workspaceId(req);
  if (!userId) return NextResponse.json({ ok: false, error: "A valid private sync link is required." }, { status: 401 });
  try {
    return NextResponse.json({ ok: true, data: await loadUserPortfolios(userId) }, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : "Portfolio sync is unavailable." }, { status: 503 });
  }
}
export async function PUT(req: Request): Promise<NextResponse> {
  if (await isRateLimited(getPortfolioSyncLimiter(), getClientIp(req.headers))) {
    return NextResponse.json({ ok: false, error: "Too many sync requests. Try again shortly." }, { status: 429 });
  }
  const userId = workspaceId(req);
  if (!userId) return NextResponse.json({ ok: false, error: "A valid private sync link is required." }, { status: 401 });
  const contentLength = Number(req.headers.get("content-length") ?? 0);
  if (contentLength > 1_000_000) return NextResponse.json({ ok: false, error: "Portfolio data is too large." }, { status: 413 });
  try {
    return NextResponse.json({ ok: true, data: await saveUserPortfolios(userId, await req.json()) });
  } catch (error) {
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : "Portfolio sync failed." }, { status: 503 });
  }
}
// End portfolio persistence route.
