import { NextRequest, NextResponse } from "next/server";
import { toggleFeed, deleteFeed } from "@/lib/server/neon";

export const dynamic = "force-dynamic";

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ feedId: string }> }
): Promise<NextResponse> {
  try {
    const { feedId } = await params;
    const id = parseInt(feedId, 10);
    if (!Number.isFinite(id)) return NextResponse.json({ ok: false, error: "Invalid id" }, { status: 400 });
    const body = (await req.json()) as { active?: boolean };
    if (typeof body.active !== "boolean") {
      return NextResponse.json({ ok: false, error: "active (boolean) is required" }, { status: 400 });
    }
    await toggleFeed(id, body.active);
    return NextResponse.json({ ok: true });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ feedId: string }> }
): Promise<NextResponse> {
  try {
    const { feedId } = await params;
    const id = parseInt(feedId, 10);
    if (!Number.isFinite(id)) return NextResponse.json({ ok: false, error: "Invalid id" }, { status: 400 });
    await deleteFeed(id);
    return NextResponse.json({ ok: true });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}
