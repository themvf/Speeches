import { NextRequest, NextResponse } from "next/server";
import { getFeeds, addFeed } from "@/lib/server/neon";

export const dynamic = "force-dynamic";

export async function GET(): Promise<NextResponse> {
  try {
    const feeds = await getFeeds();
    return NextResponse.json({ ok: true, data: { feeds } });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const body = (await req.json()) as { label?: string; feedUrl?: string };
    const label = String(body.label ?? "").trim();
    const feedUrl = String(body.feedUrl ?? "").trim();
    if (!label || !feedUrl) {
      return NextResponse.json({ ok: false, error: "label and feedUrl are required" }, { status: 400 });
    }
    new URL(feedUrl); // validates URL format
    const feed = await addFeed(label, feedUrl);
    return NextResponse.json({ ok: true, data: { feed } });
  } catch (err) {
    const msg = String(err);
    if (msg.includes("Invalid URL")) {
      return NextResponse.json({ ok: false, error: "Invalid feed URL" }, { status: 400 });
    }
    return NextResponse.json({ ok: false, error: msg }, { status: 500 });
  }
}
