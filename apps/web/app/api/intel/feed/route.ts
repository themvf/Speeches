import { NextRequest, NextResponse } from "next/server";
import { getRecentArticles } from "@/lib/server/neon";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest): Promise<NextResponse> {
  const { searchParams } = req.nextUrl;
  const limit = Math.min(Number(searchParams.get("limit") ?? "100"), 200);
  const feedKey = searchParams.get("feedKey") ?? undefined;
  const sinceParam = searchParams.get("since");
  const since = sinceParam ? new Date(sinceParam) : undefined;

  try {
    const articles = await getRecentArticles({ limit, feedKey, since });
    return NextResponse.json({
      ok: true,
      data: { articles, generatedAt: new Date().toISOString() },
    });
  } catch (err) {
    return NextResponse.json(
      { ok: false, error: String(err) },
      { status: 500 }
    );
  }
}
