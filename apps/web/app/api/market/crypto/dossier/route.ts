import { fail, ok } from "@/lib/server/api-utils";
import { isCoin } from "@/lib/crypto-coins";
import { readClaimDossier } from "@/lib/server/intelligence-fusion-store";
import { withCdnCache } from "@/lib/server/crypto-ranking-cache";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(request: Request) {
  const coin = (new URL(request.url).searchParams.get("coin") ?? "").toUpperCase();
  if (!isCoin(coin)) return fail("Unknown or missing coin", "INVALID_COIN", 400);
  try {
    return withCdnCache(ok(await readClaimDossier(coin)), 120);
  } catch {
    return fail("The saved claim dossier is temporarily unavailable", "CLAIM_DOSSIER_READ_FAILED", 503);
  }
}
