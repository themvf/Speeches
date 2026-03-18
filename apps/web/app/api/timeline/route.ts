import {
  buildDocumentListItems,
  buildDocumentsFacets,
  loadCorpusDocuments,
  loadEnrichmentState,
  parseComparableDate
} from "@/lib/server/data-store";
import { createRequestId, fail, normalizeText, ok, parseDate } from "@/lib/server/api-utils";
import { buildFullTextById, filterDocumentListItems } from "@/lib/server/document-query";
import type { TimelineBucket, TimelineResponseData } from "@/lib/server/types";

export const runtime = "nodejs";

type TimelineGrain = TimelineResponseData["grain"];

function parseGrain(value: string | null): TimelineGrain {
  const grain = String(value || "").trim().toLowerCase();
  if (grain === "year" || grain === "quarter") {
    return grain;
  }
  return "month";
}

function utcDate(year: number, month: number, day: number): Date {
  return new Date(Date.UTC(year, month, day));
}

function formatDateOnlyUtc(value: Date): string {
  return value.toISOString().slice(0, 10);
}

function formatBucketLabel(start: Date, grain: TimelineGrain): string {
  const year = start.getUTCFullYear();
  if (grain === "year") {
    return String(year);
  }
  if (grain === "quarter") {
    const quarter = Math.floor(start.getUTCMonth() / 3) + 1;
    return `Q${quarter} ${year}`;
  }
  return start.toLocaleDateString("en-US", {
    month: "short",
    year: "numeric",
    timeZone: "UTC"
  });
}

function bucketBounds(date: Date, grain: TimelineGrain): Pick<TimelineBucket, "key" | "label" | "start" | "end"> {
  const year = date.getUTCFullYear();
  const month = date.getUTCMonth();

  if (grain === "year") {
    const start = utcDate(year, 0, 1);
    const end = utcDate(year, 11, 31);
    return {
      key: formatDateOnlyUtc(start),
      label: formatBucketLabel(start, grain),
      start: formatDateOnlyUtc(start),
      end: formatDateOnlyUtc(end)
    };
  }

  if (grain === "quarter") {
    const quarterStartMonth = Math.floor(month / 3) * 3;
    const start = utcDate(year, quarterStartMonth, 1);
    const end = utcDate(year, quarterStartMonth + 3, 0);
    return {
      key: formatDateOnlyUtc(start),
      label: formatBucketLabel(start, grain),
      start: formatDateOnlyUtc(start),
      end: formatDateOnlyUtc(end)
    };
  }

  const start = utcDate(year, month, 1);
  const end = utcDate(year, month + 1, 0);
  return {
    key: formatDateOnlyUtc(start),
    label: formatBucketLabel(start, grain),
    start: formatDateOnlyUtc(start),
    end: formatDateOnlyUtc(end)
  };
}

export async function GET(request: Request) {
  const requestId = createRequestId();

  try {
    const url = new URL(request.url);
    const q = normalizeText(url.searchParams.get("q"));
    const org = normalizeText(url.searchParams.get("org"));
    const sourceKind = normalizeText(url.searchParams.get("source_kind") || url.searchParams.get("source"));
    const topic = normalizeText(url.searchParams.get("topic"));
    const keyword = normalizeText(url.searchParams.get("keyword"));
    const tag = normalizeText(url.searchParams.get("tag"));
    const status = normalizeText(url.searchParams.get("status"));
    const fromDate = parseDate(url.searchParams.get("date_from"));
    const toDate = parseDate(url.searchParams.get("date_to"));
    const grain = parseGrain(url.searchParams.get("grain"));

    const [corpusDocs, enrichment] = await Promise.all([loadCorpusDocuments(), loadEnrichmentState()]);
    const items = buildDocumentListItems(corpusDocs, enrichment);
    const facets = buildDocumentsFacets(items);
    const fullTextById = buildFullTextById(corpusDocs);
    const filtered = filterDocumentListItems(items, fullTextById, {
      q,
      org,
      sourceKind,
      topic,
      keyword,
      tag,
      status,
      fromDate,
      toDate
    });

    const bucketMap = new Map<
      string,
      {
        bucket: Pick<TimelineBucket, "key" | "label" | "start" | "end">;
        count: number;
        sourceCounts: Map<string, number>;
      }
    >();

    let datedDocuments = 0;
    let undatedDocuments = 0;
    let minDateMs = 0;
    let maxDateMs = 0;

    for (const item of filtered) {
      const comparable = parseComparableDate(item.published_at || item.date);
      if (!comparable) {
        undatedDocuments += 1;
        continue;
      }

      datedDocuments += 1;
      minDateMs = minDateMs === 0 ? comparable : Math.min(minDateMs, comparable);
      maxDateMs = maxDateMs === 0 ? comparable : Math.max(maxDateMs, comparable);

      const bounds = bucketBounds(new Date(comparable), grain);
      const existing = bucketMap.get(bounds.key) || {
        bucket: bounds,
        count: 0,
        sourceCounts: new Map<string, number>()
      };

      existing.count += 1;
      existing.sourceCounts.set(item.source_kind, (existing.sourceCounts.get(item.source_kind) || 0) + 1);
      bucketMap.set(bounds.key, existing);
    }

    const buckets = [...bucketMap.values()]
      .sort((a, b) => a.bucket.key.localeCompare(b.bucket.key))
      .map((entry) => ({
        key: entry.bucket.key,
        label: entry.bucket.label,
        start: entry.bucket.start,
        end: entry.bucket.end,
        count: entry.count,
        source_counts: [...entry.sourceCounts.entries()]
          .map(([source_kind, count]) => ({ source_kind, count }))
          .sort((a, b) => b.count - a.count || a.source_kind.localeCompare(b.source_kind))
      }));

    const peakBucket =
      buckets.reduce<TimelineBucket | null>((best, bucket) => {
        if (!best || bucket.count > best.count) {
          return bucket;
        }
        if (best && bucket.count === best.count && bucket.key < best.key) {
          return bucket;
        }
        return best;
      }, null) || null;

    return ok<TimelineResponseData>(
      {
        grain,
        buckets,
        totals: {
          matching_documents: filtered.length,
          dated_documents: datedDocuments,
          undated_documents: undatedDocuments,
          bucket_count: buckets.length,
          peak_bucket_key: peakBucket?.key || "",
          peak_bucket_label: peakBucket?.label || "",
          peak_bucket_count: peakBucket?.count || 0,
          start_date: minDateMs ? formatDateOnlyUtc(new Date(minDateMs)) : "",
          end_date: maxDateMs ? formatDateOnlyUtc(new Date(maxDateMs)) : ""
        },
        facets
      },
      requestId
    );
  } catch (error) {
    return fail(
      `Failed to build timeline data: ${error instanceof Error ? error.message : "Unknown error"}`,
      "TIMELINE_BUILD_FAILED",
      500,
      requestId
    );
  }
}
