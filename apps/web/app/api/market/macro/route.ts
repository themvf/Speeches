import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import { fetchFredMacroIndicators, FRED_MACRO_CACHE_SECONDS } from "@/lib/server/fred-macro";
import type { MarketMacroData } from "@/lib/server/types";

export const runtime = "nodejs";
export const revalidate = 900;

export async function GET() {
  const requestId = createRequestId();
  try {
    const indicators = await fetchFredMacroIndicators();
    const data: MarketMacroData = {
      indicators,
      generatedAt: new Date().toISOString(),
      cacheSeconds: FRED_MACRO_CACHE_SECONDS,
      source: "FRED",
    };
    return ok(data, requestId);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to load FRED macro indicators.";
    const missingKey = message.includes("FRED_API_KEY");
    return fail(message, missingKey ? "FRED_NOT_CONFIGURED" : "FRED_UPSTREAM_ERROR", missingKey ? 503 : 502, requestId);
  }
}
