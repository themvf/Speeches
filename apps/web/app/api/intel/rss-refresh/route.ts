import { NextRequest, NextResponse } from "next/server";
import { fetchRssFeed, WSJ_FEEDS } from "@/lib/server/rss-fetcher";
import { upsertRssArticles, ensureSchema } from "@/lib/server/neon";

export const dynamic = "force-dynamic";
export const maxDuration = 55;

export async function GET(req: NextRequest): Promise<NextResponse> {
  return handleRefresh(req);
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  return handleRefresh(req);
}

async function handleRefresh(req: NextRequest): Promise<NextResponse> {
  if (process.env.NODE_ENV === "production") {
    const authHeader = req.headers.get("authorization") ?? "";
    const secret = process.env.CRON_SECRET ?? "";
    if (!secret || authHeader !== `Bearer ${secret}`) {
      return NextResponse.json({ ok: false, error: "Unauthorized" }, { status: 401 });
    }
  }

  try {
    await ensureSchema();
  } catch (err) {
    return NextResponse.json(
      { ok: false, error: `Schema init failed: ${String(err)}` },
      { status: 500 }
    );
  }

  const feedResults = await Promise.allSettled(
    Object.entries(WSJ_FEEDS).map(async ([feedKey, { feedUrl }]) => {
      const articles = await fetchRssFeed(feedUrl, 50);
      const inserted = await upsertRssArticles(articles, feedKey);
      return { feedKey, fetched: articles.length, inserted };
    })
  );

  const feeds: Array<{ feedKey: string; fetched: number; inserted: number; error?: string }> = [];
  let totalInserted = 0;

  for (const result of feedResults) {
    if (result.status === "fulfilled") {
      feeds.push(result.value);
      totalInserted += result.value.inserted;
    } else {
      feeds.push({ feedKey: "unknown", fetched: 0, inserted: 0, error: String(result.reason) });
    }
  }

  return NextResponse.json({
    ok: true,
    data: { inserted: totalInserted, feeds },
  });
}
