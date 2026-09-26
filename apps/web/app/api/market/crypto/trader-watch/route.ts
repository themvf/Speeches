import { NextRequest, NextResponse } from "next/server";
import { labelWalletActivity, readWalletActivity, readWalletHoldings } from "@/lib/server/trader-watch";
import { traderWatchProfile, type TraderWatchData } from "@/lib/trader-watch";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: NextRequest): Promise<NextResponse> {
  const id = request.nextUrl.searchParams.get("trader") ?? "lbexplorer";
  const profile = traderWatchProfile(id);
  if (!profile) return NextResponse.json({ ok: false, error: "Unknown wallet profile" }, { status: 400 });

  const [walletResult, holdingsResult] = await Promise.allSettled([
    readWalletActivity(profile),
    readWalletHoldings(profile),
  ]);
  const activity = walletResult.status === "fulfilled"
    ? await labelWalletActivity(walletResult.value, holdingsResult.status === "fulfilled" ? holdingsResult.value.items : [])
    : [];
  const data: TraderWatchData = {
    profile,
    walletActivity: walletResult.status === "fulfilled"
      ? { status: "available", items: activity, note: activity.some((item) => item.status === "details_unavailable")
        ? "Some transaction details could not be read from the RPC. Open the explorer links for those signatures."
        : walletResult.value.length ? null : "No recent transactions returned by the RPC." }
      : { status: "unavailable", items: [], note: "Solana activity is currently unavailable. Try again later or configure SOLANA_RPC_URL." },
    walletHoldings: holdingsResult.status === "fulfilled"
      ? holdingsResult.value
      : { status: "unavailable", items: [], sol: null, observedAt: null, note: "Current holdings could not be read from Solana RPC." },
    generatedAt: new Date().toISOString(),
  };
  return NextResponse.json({ ok: true, data }, {
    headers: { "Cache-Control": "public, s-maxage=60, stale-while-revalidate=120" },
  });
}
