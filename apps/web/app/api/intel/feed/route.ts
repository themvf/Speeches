import { NextRequest, NextResponse } from "next/server";
import { getRecentArticles, getTopicRules } from "@/lib/server/neon";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest): Promise<NextResponse> {
  const { searchParams } = req.nextUrl;
  const limit = Math.min(Number(searchParams.get("limit") ?? "100"), 200);
  const feedKey = searchParams.get("feedKey") ?? undefined;
  const sinceParam = searchParams.get("since");
  const since = sinceParam ? new Date(sinceParam) : undefined;

  try {
    const [articles, topicRules] = await Promise.all([
      getRecentArticles({ limit, feedKey, since }),
      getTopicRules(true),
    ]);
    return NextResponse.json(
      {
        ok: true,
        data: { articles, topicRules, generatedAt: new Date().toISOString() },
      },
      {
        headers: {
          "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
        },
      }
    );
  } catch (err) {
    return NextResponse.json(
      { ok: false, error: String(err) },
      { status: 500 }
    );
  }
}
