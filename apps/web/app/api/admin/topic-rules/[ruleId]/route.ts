import { NextRequest, NextResponse } from "next/server";
import { updateTopicRule, deleteTopicRule } from "@/lib/server/neon";

export const dynamic = "force-dynamic";

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ ruleId: string }> }
): Promise<NextResponse> {
  try {
    const { ruleId } = await params;
    const id = parseInt(ruleId, 10);
    if (!Number.isFinite(id)) return NextResponse.json({ ok: false, error: "Invalid id" }, { status: 400 });
    const body = (await req.json()) as {
      label?: string;
      keywords?: string;
      active?: boolean;
      sortOrder?: number;
    };
    await updateTopicRule(id, body);
    return NextResponse.json({ ok: true });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ ruleId: string }> }
): Promise<NextResponse> {
  try {
    const { ruleId } = await params;
    const id = parseInt(ruleId, 10);
    if (!Number.isFinite(id)) return NextResponse.json({ ok: false, error: "Invalid id" }, { status: 400 });
    await deleteTopicRule(id);
    return NextResponse.json({ ok: true });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}
