import { type NextRequest, NextResponse } from "next/server";

import {
  deleteYouTubeChannel,
  updateYouTubeChannel,
} from "@/lib/server/youtube-channel-config";

export const dynamic = "force-dynamic";

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ channelId: string }> }
): Promise<NextResponse> {
  try {
    const { channelId } = await params;
    const body = await req.json() as {
      active?: boolean;
      label?: string;
      extraction_limit?: unknown;
      enrich_limit?: unknown;
      max_pages?: unknown;
    };
    const payload = await updateYouTubeChannel(channelId, body);
    return NextResponse.json({ ok: true, data: payload });
  } catch (err) {
    const message = String(err);
    return NextResponse.json({ ok: false, error: message }, { status: message.includes("not found") ? 404 : 500 });
  }
}

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ channelId: string }> }
): Promise<NextResponse> {
  try {
    const { channelId } = await params;
    const payload = await deleteYouTubeChannel(channelId);
    return NextResponse.json({ ok: true, data: payload });
  } catch (err) {
    const message = String(err);
    return NextResponse.json({ ok: false, error: message }, { status: message.includes("not found") ? 404 : 400 });
  }
}
