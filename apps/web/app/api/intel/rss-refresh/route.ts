import { NextRequest, NextResponse } from "next/server";
import { fetchRssFeed } from "@/lib/server/rss-fetcher";
import { upsertRssArticles, ensureSchema, getFeeds } from "@/lib/server/neon";

export const dynamic = "force-dynamic";
export const maxDuration = 55;

export async function GET(req: NextRequest): Promise<NextResponse> {
  return handleRefresh(req);
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  return handleRefresh(req);
}

async function handleRefresh(req: NextRequest): Promise<NextResponse> {
  const secret = process.env.CRON_SECRET ?? "";
  if (secret) {
    const authHeader = req.headers.get("authorization") ?? "";
    if (authHeader !== `Bearer ${secret}`) {
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

  const activeFeeds = await getFeeds(true);

  const feedResults = await Promise.allSettled(
    activeFeeds.map(async (feed) => {
      const articles = await fetchRssFeed(feed.feed_url, 50);
      const inserted = await upsertRssArticles(articles, feed.feed_key);
      return { feedKey: feed.feed_key, label: feed.label, fetched: articles.length, inserted };
    })
  );

  const feeds: Array<{ feedKey: string; label: string; fetched: number; inserted: number; error?: string }> = [];
  let totalInserted = 0;

  for (const result of feedResults) {
    if (result.status === "fulfilled") {
      feeds.push(result.value);
      totalInserted += result.value.inserted;
    } else {
      feeds.push({ feedKey: "unknown", label: "unknown", fetched: 0, inserted: 0, error: String(result.reason) });
    }
  }

  return NextResponse.json({
    ok: true,
    data: { inserted: totalInserted, feeds },
  });
}
