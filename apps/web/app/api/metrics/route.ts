import { loadCorpusDocuments, loadCustomDocuments, loadEnrichmentState, loadNewsConnectorSettings } from "@/lib/server/data-store";
import { getApiRuntimeInfo } from "@/lib/server/env";
import { createRequestId, fail, ok } from "@/lib/server/api-utils";

export const runtime = "nodejs";

function toMs(value: string): number {
  const ms = Date.parse(String(value || ""));
  return Number.isFinite(ms) ? ms : 0;
}

export async function GET() {
  const requestId = createRequestId();

  try {
    const [corpus, custom, enrichment, settings] = await Promise.all([
      loadCorpusDocuments(),
      loadCustomDocuments(),
      loadEnrichmentState(),
      loadNewsConnectorSettings()
    ]);

    const documents = corpus || [];
    const orgSet = new Set<string>();
    const sourceCounts = new Map<string, number>();

    for (const doc of documents) {
      const metadata = doc.metadata || {};
      const org = String(metadata.organization || "unknown").trim() || "unknown";
      const kind = String(metadata.source_kind || "unknown").trim() || "unknown";
      orgSet.add(org);
      sourceCounts.set(kind, (sourceCounts.get(kind) || 0) + 1);
    }

    const entries = enrichment.entries || {};
    let enrichedCount = 0;
    let pendingReviewCount = 0;

    for (const entry of Object.values(entries)) {
      const status = String(entry.status || "").toLowerCase();
      const decision = String(entry.review?.decision || "pending").toLowerCase();

      if (["enriched", "fallback_enriched", "reviewed"].includes(status)) {
        enrichedCount += 1;
      }
      if (["enriched", "fallback_enriched"].includes(status) && !["accepted", "edited", "rejected"].includes(decision)) {
        pendingReviewCount += 1;
      }
    }

    const nowMs = Date.now();
    const recentWindowMs = 24 * 60 * 60 * 1000;
    const processedCount = documents.filter((doc) => {
      const m = doc.metadata || {};
      const isNews = String(m.source_kind || "") === "newsapi_article";
      if (!isNews) {
        return false;
      }
      const updatedAt = toMs(String(m.last_reviewed_or_updated || m.updated_date || custom.updated_at || ""));
      return updatedAt > 0 && nowMs - updatedAt <= recentWindowMs;
    }).length;

    const sortByCount = [...sourceCounts.entries()]
      .map(([source_kind, count]) => ({ source_kind, count }))
      .sort((a, b) => b.count - a.count);

    const payload = {
      totals: {
        documents: documents.length,
        organizations: orgSet.size,
        enriched: enrichedCount,
        pending_review: pendingReviewCount
      },
      recent_ingest: {
        last_run_at:
          [custom.updated_at, enrichment.updated_at, settings.updated_at]
            .map((item) => ({ value: item, ms: toMs(item) }))
            .sort((a, b) => b.ms - a.ms)[0]?.value || "",
        processed_count: processedCount,
        failed_count: 0
      },
      by_source_kind: sortByCount,
      runtime: getApiRuntimeInfo()
    };

    return ok(payload, requestId);
  } catch (error) {
    return fail(
      `Failed to build metrics payload: ${error instanceof Error ? error.message : "Unknown error"}`,
      "METRICS_BUILD_FAILED",
      500,
      requestId
    );
  }
}
