import { NextRequest, NextResponse } from "next/server";
import { loadYouTubeChannelSources, saveYouTubeChannelSources } from "@/lib/server/data-store";

export const dynamic = "force-dynamic";

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ sourceId: string }> }
): Promise<NextResponse> {
  try {
    const { sourceId } = await params;
    const body = (await req.json()) as { active?: boolean };
    if (typeof body.active !== "boolean") {
      return NextResponse.json({ ok: false, error: "active (boolean) is required" }, { status: 400 });
    }
    const active = body.active;

    const payload = await loadYouTubeChannelSources();
    let found = false;
    const sources = payload.sources.map((source) => {
      if (source.id !== sourceId) {
        return source;
      }
      found = true;
      return { ...source, active, updated_at: new Date().toISOString() };
    });

    if (!found) {
      return NextResponse.json({ ok: false, error: "YouTube source not found" }, { status: 404 });
    }

    const saved = await saveYouTubeChannelSources({ ...payload, sources });
    if (!saved.saved) {
      return NextResponse.json({ ok: false, error: "Failed to save YouTube source settings" }, { status: 500 });
    }

    return NextResponse.json({ ok: true, data: saved });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ sourceId: string }> }
): Promise<NextResponse> {
  try {
    const { sourceId } = await params;
    const payload = await loadYouTubeChannelSources();
    const sources = payload.sources.filter((source) => source.id !== sourceId);
    if (sources.length === payload.sources.length) {
      return NextResponse.json({ ok: false, error: "YouTube source not found" }, { status: 404 });
    }

    const saved = await saveYouTubeChannelSources({ ...payload, sources });
    if (!saved.saved) {
      return NextResponse.json({ ok: false, error: "Failed to save YouTube source settings" }, { status: 500 });
    }

    return NextResponse.json({ ok: true, data: saved });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}
