import { NextResponse } from "next/server";
import { getTodaysRecap } from "@/lib/server/neon";

export const dynamic = "force-dynamic";

export async function GET(): Promise<NextResponse> {
  try {
    const recap = await getTodaysRecap();
    const date = new Date().toISOString().split("T")[0];
    return NextResponse.json({ ok: true, data: { recap, date } });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}
