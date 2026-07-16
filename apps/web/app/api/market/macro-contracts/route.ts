import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import { fetchPolymarketMacroPredictions } from "@/lib/server/polymarket-macro";

export const runtime = "nodejs";
export const revalidate = 300;

export async function GET() {
  const requestId = createRequestId();
  try {
    return ok(await fetchPolymarketMacroPredictions(), requestId);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to load Polymarket macro contracts.";
    return fail(message, "POLYMARKET_UPSTREAM_ERROR", 502, requestId);
  }
}
