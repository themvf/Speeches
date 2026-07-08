import { type NextRequest, NextResponse } from "next/server";

import {
  getYouTubeChannelPayload,
  upsertYouTubeChannel,
  YOUTUBE_CHANNELS_BLOB,
} from "@/lib/server/youtube-channel-config";

export const dynamic = "force-dynamic";

export async function GET(): Promise<NextResponse> {
  try {
    const payload = await getYouTubeChannelPayload();
    return NextResponse.json({ ok: true, data: { ...payload, blobName: YOUTUBE_CHANNELS_BLOB } });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const body = await req.json() as {
      label?: string;
      channelRef?: string;
      extractionLimit?: unknown;
      enrichLimit?: unknown;
      maxPages?: unknown;
    };
    const payload = await upsertYouTubeChannel({
      label: String(body.label || "").trim(),
      channelRef: String(body.channelRef || "").trim(),
      extractionLimit: body.extractionLimit,
      enrichLimit: body.enrichLimit,
      maxPages: body.maxPages,
    });
    return NextResponse.json({ ok: true, data: payload });
  } catch (err) {
    const message = String(err);
    return NextResponse.json({ ok: false, error: message }, { status: message.includes("required") ? 400 : 500 });
  }
}
