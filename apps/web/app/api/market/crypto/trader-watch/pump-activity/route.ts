import { NextRequest, NextResponse } from "next/server";
import { readPumpActivity, unavailablePumpActivity, validPumpCursor } from "@/lib/server/pump-activity";
import { PUMP_ACTIONS, traderWatchProfile, type PumpAction } from "@/lib/trader-watch";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: NextRequest): Promise<NextResponse> {
  const profile = traderWatchProfile(request.nextUrl.searchParams.get("trader") ?? "lbexplorer");
  const action = request.nextUrl.searchParams.get("action");
  const cursor = request.nextUrl.searchParams.get("cursor");
  if (!profile || !PUMP_ACTIONS.includes(action as PumpAction) || (cursor !== null && !validPumpCursor(cursor))) {
    return NextResponse.json({ ok: false, error: "Invalid wallet activity request" }, { status: 400 });
  }
  const page = await readPumpActivity(profile, action as PumpAction, cursor ?? undefined).catch(unavailablePumpActivity);
  return NextResponse.json({ ok: true, page }, {
    headers: { "Cache-Control": "public, s-maxage=60, stale-while-revalidate=120" },
  });
}
