import { NextRequest, NextResponse } from "next/server";
import { readTraderPosts, readWalletActivity } from "@/lib/server/trader-watch";
import { traderWatchProfile, type TraderWatchData } from "@/lib/trader-watch";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: NextRequest): Promise<NextResponse> {
  const id = request.nextUrl.searchParams.get("trader") ?? "lbexplorer";
  const profile = traderWatchProfile(id);
  if (!profile) return NextResponse.json({ ok: false, error: "Unknown trader" }, { status: 400 });

  const [postResult, walletResult] = await Promise.allSettled([
    readTraderPosts(profile),
    readWalletActivity(profile),
  ]);
  const data: TraderWatchData = {
    profile,
    posts: postResult.status === "fulfilled"
      ? { status: "available", items: postResult.value, note: postResult.value.length ? null : "No saved posts for this account yet. Add it to the X account collector in Admin." }
      : { status: "unavailable", items: [], note: "Saved X posts are currently unavailable." },
    walletActivity: walletResult.status === "fulfilled"
      ? { status: "available", items: walletResult.value, note: walletResult.value.some((item) => item.status === "details_unavailable")
        ? "Some transaction details could not be read from the RPC. Open the explorer links for those signatures."
        : walletResult.value.length ? null : "No recent transactions returned by the RPC." }
      : { status: "unavailable", items: [], note: "Solana activity is currently unavailable. Try again later or configure SOLANA_RPC_URL." },
    generatedAt: new Date().toISOString(),
  };
  return NextResponse.json({ ok: true, data }, {
    headers: { "Cache-Control": "public, s-maxage=60, stale-while-revalidate=120" },
  });
}
