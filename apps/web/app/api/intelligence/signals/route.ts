import { createRequestId, ok } from "@/lib/server/api-utils";
import { buildIntelligenceSignalsData } from "@/lib/server/intelligence-signals";

export const runtime = "nodejs";

export async function GET() {
  const requestId = createRequestId();
  return ok(buildIntelligenceSignalsData(), requestId);
}
