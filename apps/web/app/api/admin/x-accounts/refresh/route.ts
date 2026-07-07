import { type NextRequest, NextResponse } from "next/server";
import { refreshXTimelines } from "@/lib/server/x-timeline-ingestion";

export const dynamic = "force-dynamic";
export const maxDuration = 55;

function boundedInt(value: unknown, fallback: number, max: number): number {
  return Math.max(0, Math.min(max, Math.round(Number(value || fallback) || fallback)));
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const body = await req.json().catch(() => ({})) as Record<string, unknown>;
    const data = await refreshXTimelines({
      limit: boundedInt(body.limit, 20, 50),
      analysisLimit: boundedInt(body.analysisLimit, 10, 50),
    });
    if (data.feeds.length === 0) {
      return NextResponse.json({ ok: false, error: "No active X accounts configured." }, { status: 400 });
    }
    return NextResponse.json({ ok: true, data });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Failed to refresh X accounts." },
      { status: 500 }
    );
  }
}
