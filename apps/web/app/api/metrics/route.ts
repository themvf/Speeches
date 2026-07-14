import {
  loadNewsConnectorSettings,
  loadNewsFeedDocumentsFromNeon,
} from "@/lib/server/data-store";
import { getApiRuntimeInfo } from "@/lib/server/env";
import {
  getMirroredDocumentMetricsSnapshot,
  type NeonDocumentMetricsSnapshot,
} from "@/lib/server/neon";
import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import type { NewsConnectorSettingsPayload } from "@/lib/server/types";

export const runtime = "nodejs";

const METRICS_LOAD_TIMEOUT_MS = 6_000;

type LoadResult<T> = {
  data: T;
  warning?: string;
};

function emptyNewsConnectorSettings(): NewsConnectorSettingsPayload {
  return {
    updated_at: "",
    query: "",
    lookback_days: 7,
    max_pages: 4,
    page_size: 50,
    target_count: 100,
    sort_by: "publishedAt",
    organization_label: "News",
    domains: "",
    exclude_domains: "",
    tags_csv: "",
    doj_usao_exclude_terms: ""
  };
}

function emptyDocumentMetrics(): NeonDocumentMetricsSnapshot {
  return {
    documents: 0,
    organizations: 0,
    enriched: 0,
    pendingReview: 0,
    lastRunAt: "",
    processedCount: 0,
    sourceCounts: [],
    newsApi: {
      total: 0,
      recent24h: 0,
      recent7d: 0,
      recent30d: 0,
      newest: null,
      bySource: [],
    },
    enrichmentAvailable: false,
  };
}

function loadWithBudget<T>(
  label: string,
  promise: Promise<T>,
  fallback: () => T,
  timeoutMs = METRICS_LOAD_TIMEOUT_MS
): Promise<LoadResult<T>> {
  const guarded = promise
    .then((data) => ({ data }))
    .catch((error) => {
      console.error(`[metrics] ${label} failed`, error);
      return {
        data: fallback(),
        warning: `${label} is temporarily unavailable`,
      };
    });

  return Promise.race([
    guarded,
    new Promise<LoadResult<T>>((resolve) => {
      setTimeout(() => resolve({ data: fallback(), warning: `${label} exceeded ${timeoutMs}ms budget` }), timeoutMs);
    })
  ]);
}

export async function GET() {
  const requestId = createRequestId();

  try {
    const [metricsResult, feedResult, settingsResult] = await Promise.all([
      loadWithBudget("document metrics", getMirroredDocumentMetricsSnapshot(), emptyDocumentMetrics),
      loadWithBudget(
        "feed document projection",
        loadNewsFeedDocumentsFromNeon({ limit: 250, pinnedSourceKindLimit: 25 }),
        () => ({ documents: [], source: "unavailable" as const, metadata_only: true as const })
      ),
      // This settings blob is small; the expensive corpus and enrichment
      // snapshots are deliberately absent from this route.
      loadWithBudget("news connector settings", loadNewsConnectorSettings(), emptyNewsConnectorSettings),
    ]);

    const metrics = metricsResult.data;
    const feedDocuments = feedResult.data.documents || [];
    const feedSourceCounts = new Map<string, number>();
    for (const item of feedDocuments) {
      const kind = String(item.source_kind || "unknown").trim() || "unknown";
      feedSourceCounts.set(kind, (feedSourceCounts.get(kind) || 0) + 1);
    }

    const lastRunAt = [metrics.lastRunAt, settingsResult.data.updated_at]
      .filter(Boolean)
      .sort((a, b) => Date.parse(b) - Date.parse(a))[0] || "";
    const warnings = [
      metricsResult.warning,
      feedResult.warning,
      feedResult.data.warning,
      settingsResult.warning,
      ...(!metrics.enrichmentAvailable ? ["document enrichment projection is unavailable"] : []),
    ].filter(Boolean);

    const payload = {
      totals: {
        documents: metrics.documents,
        organizations: metrics.organizations,
        enriched: metrics.enriched,
        pending_review: metrics.pendingReview
      },
      recent_ingest: {
        last_run_at: lastRunAt,
        processed_count: metrics.processedCount,
        failed_count: 0
      },
      connector_audit: {
        newsapi: {
          total: metrics.newsApi.total,
          in_feed: feedSourceCounts.get("newsapi_article") || 0,
          recent_24h: metrics.newsApi.recent24h,
          recent_7d: metrics.newsApi.recent7d,
          recent_30d: metrics.newsApi.recent30d,
          newest: metrics.newsApi.newest,
          by_source: metrics.newsApi.bySource,
        },
        feed_documents: {
          total: feedDocuments.length,
          by_source_kind: [...feedSourceCounts.entries()]
            .map(([source_kind, count]) => ({ source_kind, count }))
            .sort((a, b) => b.count - a.count || a.source_kind.localeCompare(b.source_kind))
        }
      },
      by_source_kind: metrics.sourceCounts,
      corpus_source: metricsResult.warning ? "unavailable" : "neon",
      runtime: getApiRuntimeInfo(),
      warnings
    };

    return ok(payload, requestId);
  } catch (error) {
    console.error("[metrics] failed to build payload", { requestId, error });
    return fail(
      "Failed to build metrics payload.",
      "METRICS_BUILD_FAILED",
      500,
      requestId
    );
  }
}
