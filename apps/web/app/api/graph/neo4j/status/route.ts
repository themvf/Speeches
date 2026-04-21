import { createRequestId, ok } from "@/lib/server/api-utils";
import { getNeo4jStatus } from "@/lib/server/neo4j";
import type { Neo4jStatusResponseData } from "@/lib/server/types";

export const runtime = "nodejs";

export async function GET() {
  const requestId = createRequestId();

  return ok<Neo4jStatusResponseData>(getNeo4jStatus(), requestId);
}
