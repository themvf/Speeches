import { type NextRequest, NextResponse } from "next/server";
import { listXTimelineFeeds } from "@/lib/server/x-timeline-ingestion";
import { upsertFeedSource } from "@/lib/server/neon";
import {
  normalizeXUsername,
  xTimelineFeedKey,
  xTimelineFeedLabel,
  xTimelineFeedUrl,
  xTimelineUsernameFromFeed,
} from "@/lib/server/x-syndication";

export const dynamic = "force-dynamic";

function normalizeInterval(value: unknown): number {
  return Math.max(15, Math.min(1440, Math.round(Number(value || 180) || 180)));
}

export async function GET(): Promise<NextResponse> {
  try {
    const feeds = await listXTimelineFeeds(false);
    return NextResponse.json({
      ok: true,
      data: {
        accounts: feeds.map((feed) => ({
          ...feed,
          username: xTimelineUsernameFromFeed(feed),
        })),
      },
    });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Failed to load X accounts." },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const body = await req.json().catch(() => ({})) as Record<string, unknown>;
    const username = normalizeXUsername(String(body.account || body.username || ""));
    if (!username) {
      return NextResponse.json({ ok: false, error: "A valid X account handle is required." }, { status: 400 });
    }

    const feed = await upsertFeedSource(
      xTimelineFeedKey(username),
      xTimelineFeedLabel(username),
      xTimelineFeedUrl(username),
      normalizeInterval(body.refreshIntervalMinutes)
    );
    return NextResponse.json({
      ok: true,
      data: {
        account: {
          ...feed,
          username: xTimelineUsernameFromFeed(feed),
        },
      },
    });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Failed to save X account." },
      { status: 500 }
    );
  }
}
