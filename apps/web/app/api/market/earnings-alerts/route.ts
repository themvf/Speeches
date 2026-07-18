import { type NextRequest } from "next/server";
import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import type { MarketEarningsAlertsData } from "@/lib/server/types";
import { getPolymarketSharpAlerts } from "@/lib/server/neon";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// SEC-29: recent sharp-wallet (early_sharp/longshot) entries into a ticker's
// still-open earnings market. Lazy-fetched per card from earnings-tab.tsx
// (same pattern as the existing Headlines fetch), not folded into
// earnings-week's payload - that route has a static-snapshot fallback with
// no concept of alert events, and this is inherently a live-only, per-ticker
// concern. A DB read failure degrades to an empty list with a warning
// rather than a 500, matching every other Polymarket reader in this app.
const SHARP_ARCHETYPES = new Set(["early_sharp", "longshot"]);

export async function GET(req: NextRequest) {
  const requestId = createRequestId();
  const ticker = (req.nextUrl.searchParams.get("ticker") ?? "").trim().toUpperCase();
  if (!ticker || !/^[A-Z0-9.-]{1,8}$/.test(ticker)) {
    return fail("A valid ticker is required.", "BAD_TICKER", 400, requestId);
  }

  const data: MarketEarningsAlertsData = { ticker, alerts: [] };
  try {
    data.alerts = (await getPolymarketSharpAlerts(ticker))
      .filter((row) => SHARP_ARCHETYPES.has(row.archetype))
      .map((row) => ({
        wallet: row.wallet,
        name: row.name || row.wallet.slice(0, 8),
        archetype: row.archetype as "early_sharp" | "longshot",
        side: row.side as "BUY" | "SELL",
        outcome: row.outcome,
        size: row.size,
        price: row.price,
        filledAt: row.filled_at,
      }));
  } catch {
    data.warning = "Sharp-alert history unavailable (DB unreachable).";
  }
  return ok(data, requestId);
}
