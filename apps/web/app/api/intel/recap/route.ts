import { type NextRequest, NextResponse } from "next/server";
import { getTodaysRecap } from "@/lib/server/neon";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest): Promise<NextResponse> {
  const rawDate = req.nextUrl.searchParams.get("date") ?? undefined;
  const date = rawDate ?? new Date().toISOString().split("T")[0];

  if (rawDate && !/^\d{4}-\d{2}-\d{2}$/.test(rawDate)) {
    return NextResponse.json({ ok: false, error: "Invalid date format - use YYYY-MM-DD" }, { status: 400 });
  }

  try {
    const recap = await getTodaysRecap(rawDate);
    return NextResponse.json({ ok: true, data: { recap, date } });
  } catch (err) {
    console.error("[recap/GET]", err);
    return NextResponse.json({
      ok: true,
      data: {
        recap: [],
        date,
        warning: `Recap store unavailable: ${err instanceof Error ? err.message : "Unknown error"}`
      }
    });
  }
}
