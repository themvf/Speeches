import { type NextRequest, NextResponse } from "next/server";
import { checkCronAuth } from "@/lib/server/api-utils";
import { runScheduledEditorial } from "@/lib/server/scheduled-editorial";

export const dynamic = "force-dynamic";
export const maxDuration = 300;

async function handle(req: NextRequest): Promise<NextResponse> {
  const auth = checkCronAuth(req);
  if (!auth.ok) return NextResponse.json({ ok: false, error: auth.error }, { status: auth.status });
  try {
    const result = await runScheduledEditorial("scheduled");
    return NextResponse.json({ ok: true, data: result });
  } catch (error) {
    console.error("[briefings/scheduled/run]", error);
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : String(error) }, { status: 500 });
  }
}

export async function GET(req: NextRequest): Promise<NextResponse> {
  return handle(req);
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  return handle(req);
}
