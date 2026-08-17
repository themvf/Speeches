import { NextResponse } from "next/server";

import { getAttentionAlerts } from "@/lib/server/neon";

export const dynamic = "force-dynamic";

// Attention alerts (enhancement 3). Lazy-fetched by the Attention tab rather
// than folded into /api/market/attention, matching the split already made for
// earnings-alerts: alerts are event-shaped (many rows per ticker, each with
// its own date) while the attention payload is one row per ticker, and the
// attention route has a snapshot fallback with no concept of alert events.
//
// Fail-soft on purpose. attention_alerts is Python-owned and is created by the
// daily rollup, so it does not exist until that job has run once; a missing
// table must degrade to an empty feed with a visible warning, never a 500 that
// takes the tab down.
export async function GET(request: Request) {
  const url = new URL(request.url);
  const tickerParam = (url.searchParams.get("ticker") || "").trim().toUpperCase();
  const limitParam = Number(url.searchParams.get("limit") || 100);
  const limit = Number.isFinite(limitParam) ? Math.max(1, Math.min(limitParam, 500)) : 100;

  try {
    const alerts = await getAttentionAlerts({ ticker: tickerParam || undefined, limit });
    return NextResponse.json({ ok: true, data: { alerts, ticker: tickerParam || null } });
  } catch (error) {
    return NextResponse.json({
      ok: true,
      data: {
        alerts: [],
        ticker: tickerParam || null,
        warning:
          error instanceof Error && /relation .*attention_alerts.* does not exist/i.test(error.message)
            ? "Attention alerts have not been generated yet - they appear after the next daily rollup."
            : "Attention alerts are unavailable right now.",
      },
    });
  }
}
