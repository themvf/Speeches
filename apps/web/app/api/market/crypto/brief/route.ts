import { fail, ok } from "@/lib/server/api-utils";
import { isCoin } from "@/lib/crypto-coins";
import { buildCryptoBrief } from "@/lib/crypto-brief";
import type { CryptoBriefWindow } from "@/lib/crypto-brief-types";
import { readCryptoBriefStoredData } from "@/lib/server/crypto-brief-store";
import { withCdnCache } from "@/lib/server/crypto-ranking-cache";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
const WINDOWS = new Set<CryptoBriefWindow>(["24h", "7d", "30d"]);

export async function GET(request: Request) {
  const params = new URL(request.url).searchParams;
  const coin = (params.get("coin") ?? "").toUpperCase();
  const window = (params.get("window") ?? "24h") as CryptoBriefWindow;
  if (!isCoin(coin)) return fail("Unknown or missing coin", "INVALID_COIN", 400);
  if (!WINDOWS.has(window)) return fail("Window must be 24h, 7d, or 30d", "INVALID_WINDOW", 400);
  const asOf = new Date().toISOString();
  try {
    const stored = await readCryptoBriefStoredData(coin, window, asOf);
    return withCdnCache(ok(buildCryptoBrief({ coin, window, asOf, ...stored })), 120);
  } catch {
    return fail("The saved crypto brief is temporarily unavailable", "CRYPTO_BRIEF_READ_FAILED", 503);
  }
}
