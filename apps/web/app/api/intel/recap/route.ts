import { type NextRequest, NextResponse } from "next/server";
import { getTodaysRecap } from "@/lib/server/neon";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest): Promise<NextResponse> {
  try {
    const date = req.nextUrl.searchParams.get("date") ?? undefined;
    const recap = await getTodaysRecap(date);
    return NextResponse.json({ ok: true, data: { recap, date: date ?? new Date().toISOString().split("T")[0] } });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}
