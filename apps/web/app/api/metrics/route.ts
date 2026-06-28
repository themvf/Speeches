import {
  buildDocumentListItems,
  loadCorpusDocuments,
  loadCustomDocuments,
  loadEnrichmentState,
  loadNewsConnectorSettings,
  selectNewsFeedDocuments
} from "@/lib/server/data-store";
import { getApiRuntimeInfo } from "@/lib/server/env";
import { createRequestId, fail, ok } from "@/lib/server/api-utils";

export const runtime = "nodejs";

function toMs(value: string): number {
  const ms = Date.parse(String(value || ""));
  return Number.isFinite(ms) ? ms : 0;
}

function metadataText(metadata: Record<string, unknown>, key: string): string {
  return String(metadata[key] || "").trim();
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
    const sevenDaysMs = 7 * recentWindowMs;
    const thirtyDaysMs = 30 * recentWindowMs;
    const processedCount = documents.filter((doc) => {
      const m = doc.metadata || {};
      const isNews = String(m.source_kind || "") === "newsapi_article";
      if (!isNews) {
        return false;
      }
      const updatedAt = toMs(String(m.last_reviewed_or_updated || m.updated_date || ""));
      return updatedAt > 0 && nowMs - updatedAt <= recentWindowMs;
    }).length;

    const documentItems = buildDocumentListItems(documents, enrichment);
    const feedDocuments = selectNewsFeedDocuments(documentItems);
    const feedSourceCounts = new Map<string, number>();
    for (const item of feedDocuments) {
      const kind = String(item.source_kind || "unknown").trim() || "unknown";
      feedSourceCounts.set(kind, (feedSourceCounts.get(kind) || 0) + 1);
    }

    const newsApiDocs = documents
      .filter((doc) => String(doc.metadata?.source_kind || "") === "newsapi_article")
      .map((doc) => {
        const metadata = (doc.metadata || {}) as unknown as Record<string, unknown>;
        const publishedAt =
          metadataText(metadata, "published_at") ||
          metadataText(metadata, "published_date") ||
          metadataText(metadata, "date");
        const publishedMs = toMs(publishedAt);
        return {
          title: metadataText(metadata, "title"),
          url: metadataText(metadata, "url"),
          source_name: metadataText(metadata, "source_name") || metadataText(metadata, "speaker") || metadataText(metadata, "organization"),
          published_at: publishedAt,
          published_ms: publishedMs,
          extraction_mode: metadataText(metadata, "newsapi_extraction_mode")
        };
      })
      .sort((a, b) => b.published_ms - a.published_ms);

    const newsApiSourceCounts = new Map<string, number>();
    for (const doc of newsApiDocs) {
      const source = doc.source_name || "Unknown";
      newsApiSourceCounts.set(source, (newsApiSourceCounts.get(source) || 0) + 1);
    }

    const newestNewsApi = newsApiDocs[0] || null;
    const newsApiRecent24h = newsApiDocs.filter((doc) => doc.published_ms > 0 && nowMs - doc.published_ms <= recentWindowMs).length;
    const newsApiRecent7d = newsApiDocs.filter((doc) => doc.published_ms > 0 && nowMs - doc.published_ms <= sevenDaysMs).length;
    const newsApiRecent30d = newsApiDocs.filter((doc) => doc.published_ms > 0 && nowMs - doc.published_ms <= thirtyDaysMs).length;

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
      connector_audit: {
        newsapi: {
          total: newsApiDocs.length,
          in_feed: feedSourceCounts.get("newsapi_article") || 0,
          recent_24h: newsApiRecent24h,
          recent_7d: newsApiRecent7d,
          recent_30d: newsApiRecent30d,
          newest: newestNewsApi
            ? {
                title: newestNewsApi.title,
                url: newestNewsApi.url,
                source_name: newestNewsApi.source_name,
                published_at: newestNewsApi.published_at,
                extraction_mode: newestNewsApi.extraction_mode
              }
            : null,
          by_source: [...newsApiSourceCounts.entries()]
            .map(([source_name, count]) => ({ source_name, count }))
            .sort((a, b) => b.count - a.count || a.source_name.localeCompare(b.source_name))
            .slice(0, 12)
        },
        feed_documents: {
          total: feedDocuments.length,
          by_source_kind: [...feedSourceCounts.entries()]
            .map(([source_kind, count]) => ({ source_kind, count }))
            .sort((a, b) => b.count - a.count || a.source_kind.localeCompare(b.source_kind))
        }
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
