import { NextRequest, NextResponse } from "next/server";
import { getTopicRules, addTopicRule } from "@/lib/server/neon";

export const dynamic = "force-dynamic";

export async function GET(): Promise<NextResponse> {
  try {
    const rules = await getTopicRules(false);
    return NextResponse.json({ ok: true, data: { rules } });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const body = (await req.json()) as {
      topicKey?: string;
      label?: string;
      keywords?: string;
      active?: boolean;
      sortOrder?: number;
    };
    const topicKey = String(body.topicKey ?? "").trim().toUpperCase().replace(/[^A-Z0-9_]/g, "_");
    const label = String(body.label ?? "").trim();
    const keywords = String(body.keywords ?? "").trim();
    if (!topicKey || !label) {
      return NextResponse.json({ ok: false, error: "topicKey and label are required" }, { status: 400 });
    }
    const rule = await addTopicRule({
      topicKey,
      label,
      keywords,
      active: body.active ?? true,
      sortOrder: body.sortOrder ?? 100,
    });
    return NextResponse.json({ ok: true, data: { rule } });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}
