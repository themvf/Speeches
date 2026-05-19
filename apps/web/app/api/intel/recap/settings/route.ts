import { NextRequest, NextResponse } from "next/server";
import { getRecapSettings, saveRecapSettings } from "@/lib/server/neon";

export const dynamic = "force-dynamic";

export async function GET(): Promise<NextResponse> {
  try {
    const topicKeys = await getRecapSettings();
    return NextResponse.json({ ok: true, data: { topicKeys } });
  } catch (err) {
    console.error("[recap/settings GET]", err);
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const body = (await req.json()) as { topicKeys?: unknown };
    const topicKeys = Array.isArray(body.topicKeys)
      ? (body.topicKeys as unknown[]).filter((k): k is string => typeof k === "string")
      : [];
    await saveRecapSettings(topicKeys);
    return NextResponse.json({ ok: true });
  } catch (err) {
    console.error("[recap/settings POST]", err);
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}
