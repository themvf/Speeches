import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import { fetchRatesCreditData, RATES_CREDIT_DEFINITIONS } from "@/lib/server/rates-credit";
import { loadRatesCreditHistory, persistRatesCreditSnapshots } from "@/lib/server/rates-credit-store";

export const runtime = "nodejs";
export const revalidate = 900;

export async function GET() {
  const requestId = createRequestId();
  try {
    let history = {};
    let historyWarning = "";
    try {
      history = await loadRatesCreditHistory(RATES_CREDIT_DEFINITIONS.map((definition) => definition.seriesId));
    } catch (error) {
      historyWarning = error instanceof Error ? error.message : "Snapshot history is unavailable.";
    }

    const data = await fetchRatesCreditData(undefined, undefined, history);
    if (historyWarning) data.warnings.push(`Durable history unavailable: ${historyWarning}`);
    try {
      await persistRatesCreditSnapshots(data);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Snapshot persistence failed.";
      if (!historyWarning || message !== historyWarning) data.warnings.push(`Snapshot persistence unavailable: ${message}`);
    }
    return ok(data, requestId);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to load rates and credit data.";
    const missingKey = message.includes("FRED_API_KEY");
    return fail(message, missingKey ? "FRED_NOT_CONFIGURED" : "FRED_UPSTREAM_ERROR", missingKey ? 503 : 502, requestId);
  }
}
