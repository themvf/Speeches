import { type NextRequest } from "next/server";
import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import type { TickerEventsData } from "@/lib/server/types";
import { fetchYahooCandles } from "@/lib/server/yahoo";
import {
  getFilingEventsForTicker,
  getPolymarketEventsForTicker,
  getStockAttentionHistory,
} from "@/lib/server/neon";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// SEC-51: one payload composing the price series with every event layer we
// already ingest - filings (SEC-50 table), earnings markets (SEC-26), and
// Reddit attention history. Price is required; each event layer is
// fail-soft enrichment.
const WINDOW_DAYS = 120;

export async function GET(req: NextRequest) {
  const requestId = createRequestId();
  const ticker = (req.nextUrl.searchParams.get("ticker") ?? "").trim().toUpperCase();
  if (!ticker || !/^[A-Z0-9.-]{1,8}$/.test(ticker)) {
    return fail("A valid ticker is required.", "BAD_TICKER", 400, requestId);
  }

  const candlesRaw = await fetchYahooCandles(ticker, 3600);
  if (!candlesRaw || candlesRaw.t.length === 0) {
    return fail(`No price history available for ${ticker}.`, "NO_CANDLES", 404, requestId);
  }
  const cutoff = Date.now() / 1000 - WINDOW_DAYS * 86400;
  const candles: { t: number; c: number }[] = [];
  candlesRaw.t.forEach((t, i) => {
    const c = candlesRaw.c[i];
    if (t >= cutoff && c != null) candles.push({ t, c });
  });

  const data: TickerEventsData = {
    ticker,
    candles,
    filings: [],
    earnings: [],
    attention: [],
  };
  const warnings: string[] = [];

  try {
    data.filings = (await getFilingEventsForTicker(ticker, WINDOW_DAYS)).map((row) => ({
      form: row.form,
      filedAt: row.filed_at,
      label: row.summary || (row.form === "8-K" ? "8-K filed" : "Insider transaction"),
      url: row.url,
    }));
  } catch {
    warnings.push("filings");
  }
  try {
    data.earnings = (await getPolymarketEventsForTicker(ticker))
      .filter((row) => row.report_date)
      .map((row) => ({
        date: row.report_date!,
        resolved: row.status === "resolved",
        outcome: row.status === "resolved" ? (row.winner === "Yes" ? "beat" : "miss") : null,
      }));
  } catch {
    warnings.push("earnings markets");
  }
  try {
    data.attention = (await getStockAttentionHistory(ticker, WINDOW_DAYS)).map((row) => ({
      date: row.attention_date,
      mentions: row.total_mention_count,
    }));
  } catch {
    warnings.push("attention history");
  }

  if (warnings.length) {
    data.warning = `Event layers unavailable: ${warnings.join(", ")} (DB unreachable) - price still shown.`;
  }
  return ok(data, requestId);
}
