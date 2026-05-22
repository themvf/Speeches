import {
  buildDocumentListItems,
  loadCorpusDocuments,
  loadEnrichmentState,
  parseComparableDate
} from "@/lib/server/data-store";
import { buildFullTextById, normalizeFacetToken } from "@/lib/server/document-query";
import { createRequestId, fail, normalizeText, ok, parseDate } from "@/lib/server/api-utils";
import type { DocumentListItem, EnrichmentEntry } from "@/lib/server/types";

export const runtime = "nodejs";

type BriefingStyle = "executive" | "compliance" | "analyst" | "digest";

interface BriefingFilters {
  date_from: string;
  date_to: string;
  agencies: string[];
  topics: string[];
  source_kinds: string[];
  entities: string[];
  style: BriefingStyle;
}

interface BriefingSource {
  document_id: string;
  title: string;
  organization: string;
  source_kind: string;
  doc_type: string;
  published_at: string;
  url: string;
  summary: string;
  topics: string[];
  keywords: string[];
}

function normalizeStringList(value: unknown, maxItems = 30): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const out: string[] = [];
  const seen = new Set<string>();
  for (const item of value) {
    const text = normalizeText(item).slice(0, 120);
    const key = text.toLowerCase();
    if (!text || seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(text);
    if (out.length >= maxItems) {
      break;
    }
  }
  return out;
}

function dayMs(): number {
  return 24 * 60 * 60 * 1000;
}

function endOfDate(date: Date): Date {
  return new Date(date.getTime() + dayMs() - 1);
}

function previousWindow(fromDate: Date, toDate: Date): { from: Date; to: Date } {
  const spanMs = Math.max(dayMs(), endOfDate(toDate).getTime() - fromDate.getTime() + 1);
  const to = new Date(fromDate.getTime() - 1);
  const from = new Date(to.getTime() - spanMs + 1);
  return { from, to };
}

function dateLabel(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function inRange(item: DocumentListItem, fromDate: Date, toDate: Date): boolean {
  const value = parseComparableDate(item.published_at || item.date);
  return Boolean(value && value >= fromDate.getTime() && value <= endOfDate(toDate).getTime());
}

function itemText(item: DocumentListItem, fullTextById: Map<string, string>): string {
  return [
    item.title,
    item.organization,
    item.source_kind,
    item.doc_type,
    item.speaker,
    item.url,
    ...(item.tags || []),
    ...(item.topics || []),
    ...(item.keywords || []),
    fullTextById.get(item.document_id) || ""
  ].join("\n").toLowerCase();
}

function matchesFilters(item: DocumentListItem, filters: BriefingFilters, fullTextById: Map<string, string>): boolean {
  if (filters.agencies.length && !filters.agencies.includes(item.organization)) {
    return false;
  }
  if (filters.source_kinds.length && !filters.source_kinds.includes(item.source_kind)) {
    return false;
  }
  if (filters.topics.length) {
    const itemTopics = (item.topics || []).map((topic) => normalizeFacetToken(topic));
    const hasTopic = filters.topics.some((topic) => {
      const selected = normalizeFacetToken(topic);
      return itemTopics.some((itemTopic) => itemTopic === selected || itemTopic.includes(selected) || selected.includes(itemTopic));
    });
    if (!hasTopic) {
      return false;
    }
  }
  if (filters.entities.length) {
    const haystack = itemText(item, fullTextById);
    const hasEntity = filters.entities.some((entity) => haystack.includes(entity.toLowerCase()));
    if (!hasEntity) {
      return false;
    }
  }
  return true;
}

function countBy<T extends string>(items: DocumentListItem[], getValues: (item: DocumentListItem) => T[]): Map<T, number> {
  const counts = new Map<T, number>();
  for (const item of items) {
    const seen = new Set<T>();
    for (const value of getValues(item)) {
      if (!value || seen.has(value)) {
        continue;
      }
      seen.add(value);
      counts.set(value, (counts.get(value) || 0) + 1);
    }
  }
  return counts;
}

function topEntries(counts: Map<string, number>, limit: number): Array<{ label: string; count: number }> {
  return [...counts.entries()]
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label))
    .slice(0, limit);
}

function enrichmentSummary(enrichmentById: Record<string, EnrichmentEntry>, docId: string): string {
  return normalizeText(enrichmentById[docId]?.enrichment?.summary || "");
}

function buildSource(item: DocumentListItem, enrichmentById: Record<string, EnrichmentEntry>): BriefingSource {
  return {
    document_id: item.document_id,
    title: item.title || "Untitled document",
    organization: item.organization,
    source_kind: item.source_kind,
    doc_type: item.doc_type,
    published_at: item.published_at || item.date,
    url: item.url,
    summary: enrichmentSummary(enrichmentById, item.document_id),
    topics: item.topics || [],
    keywords: item.keywords || []
  };
}

function sourceScore(item: DocumentListItem): number {
  return (item.word_count || 0) + (item.enrichment_status === "enriched" ? 500 : 0) + (item.url ? 100 : 0);
}

function sourceKindLabel(value: string): string {
  return value.replace(/[_-]+/g, " ").replace(/\b\w/g, (ch) => ch.toUpperCase());
}

export async function POST(request: Request) {
  const requestId = createRequestId();

  try {
    const body = (await request.json().catch(() => ({}))) as Record<string, unknown>;
    const fromDate = parseDate(normalizeText(body.date_from));
    const toDate = parseDate(normalizeText(body.date_to));

    if (!fromDate || !toDate) {
      return fail("Valid date_from and date_to values are required.", "BRIEFING_DATES_REQUIRED", 400, requestId);
    }
    if (fromDate.getTime() > toDate.getTime()) {
      return fail("date_from must be before or equal to date_to.", "BRIEFING_DATE_RANGE_INVALID", 400, requestId);
    }

    const filters: BriefingFilters = {
      date_from: dateLabel(fromDate),
      date_to: dateLabel(toDate),
      agencies: normalizeStringList(body.agencies),
      topics: normalizeStringList(body.topics),
      source_kinds: normalizeStringList(body.source_kinds),
      entities: normalizeStringList(body.entities, 12),
      style: (["executive", "compliance", "analyst", "digest"].includes(normalizeText(body.style))
        ? normalizeText(body.style)
        : "executive") as BriefingStyle
    };

    const [corpusDocs, enrichment] = await Promise.all([loadCorpusDocuments(), loadEnrichmentState()]);
    const items = buildDocumentListItems(corpusDocs, enrichment);
    const fullTextById = buildFullTextById(corpusDocs);
    const comparable = items.filter((item) => matchesFilters(item, filters, fullTextById));
    const current = comparable
      .filter((item) => inRange(item, fromDate, toDate))
      .sort((a, b) => parseComparableDate(b.published_at || b.date) - parseComparableDate(a.published_at || a.date));
    const previousRange = previousWindow(fromDate, toDate);
    const previous = comparable.filter((item) => inRange(item, previousRange.from, previousRange.to));

    const currentTopicCounts = countBy(current, (item) => item.topics || []);
    const previousTopicCounts = countBy(previous, (item) => item.topics || []);
    const currentAgencyCounts = countBy(current, (item) => [item.organization]);
    const currentSourceCounts = countBy(current, (item) => [sourceKindLabel(item.source_kind)]);

    const acceleratedTopics = topEntries(currentTopicCounts, 12)
      .map((entry) => ({ ...entry, previous_count: previousTopicCounts.get(entry.label) || 0 }))
      .map((entry) => ({ ...entry, delta: entry.count - entry.previous_count }))
      .filter((entry) => entry.count > 0)
      .sort((a, b) => b.delta - a.delta || b.count - a.count || a.label.localeCompare(b.label))
      .slice(0, 8);

    const selectedTopicTokens = new Set(filters.topics.map((topic) => normalizeFacetToken(topic)));
    const topicSectionLabels = (filters.topics.length ? filters.topics : acceleratedTopics.map((topic) => topic.label)).slice(0, 8);
    const topicSections = topicSectionLabels.map((label) => {
      const token = normalizeFacetToken(label);
      const docs = current
        .filter((item) => (item.topics || []).some((topic) => {
          const itemToken = normalizeFacetToken(topic);
          return itemToken === token || itemToken.includes(token) || token.includes(itemToken);
        }))
        .sort((a, b) => sourceScore(b) - sourceScore(a))
        .slice(0, 6);
      const previousCount = previous.filter((item) => (item.topics || []).some((topic) => normalizeFacetToken(topic) === token)).length;
      return {
        label,
        document_count: docs.length,
        previous_count: previousCount,
        delta: docs.length - previousCount,
        risk_level: docs.length >= 5 || docs.length - previousCount >= 3 ? "high" : docs.length >= 2 ? "medium" : "low",
        why_it_matters:
          docs.length > previousCount
            ? `${label} activity increased in the selected window, suggesting this theme deserves follow-up review.`
            : `${label} remained present in the selected window; review the source set for posture or emphasis changes.`,
        sources: docs.map((item) => buildSource(item, enrichment.entries || {}))
      };
    }).filter((section) => section.document_count > 0 || selectedTopicTokens.has(normalizeFacetToken(section.label)));

    const topSources = current
      .slice()
      .sort((a, b) => sourceScore(b) - sourceScore(a))
      .slice(0, 20)
      .map((item) => buildSource(item, enrichment.entries || {}));

    const delta = current.length - previous.length;
    const summaryBullets = [
      `${current.length} matching documents were found for ${filters.date_from} through ${filters.date_to}, compared with ${previous.length} in the prior comparable period.`,
      delta > 0
        ? `Activity increased by ${delta} document${delta === 1 ? "" : "s"} versus the prior window.`
        : delta < 0
          ? `Activity decreased by ${Math.abs(delta)} document${Math.abs(delta) === 1 ? "" : "s"} versus the prior window.`
          : "Activity was flat versus the prior comparable window.",
      acceleratedTopics.length
        ? `Most active themes: ${acceleratedTopics.slice(0, 4).map((topic) => topic.label).join(", ")}.`
        : "No dominant topic cluster appeared in the selected document set.",
      currentAgencyCounts.size
        ? `Most active agencies/sources: ${topEntries(currentAgencyCounts, 4).map((entry) => `${entry.label} (${entry.count})`).join(", ")}.`
        : "No agency activity matched the selected filters."
    ];

    return ok({
      id: `briefing_${Date.now().toString(36)}`,
      generated_at: new Date().toISOString(),
      title: `Custom Regulatory Briefing: ${filters.date_from} to ${filters.date_to}`,
      filters,
      comparison_window: {
        date_from: dateLabel(previousRange.from),
        date_to: dateLabel(previousRange.to)
      },
      metrics: {
        current_document_count: current.length,
        previous_document_count: previous.length,
        delta,
        agency_count: currentAgencyCounts.size,
        topic_count: currentTopicCounts.size,
        source_kind_count: currentSourceCounts.size
      },
      executive_summary: summaryBullets,
      changed_topics: acceleratedTopics,
      agency_activity: topEntries(currentAgencyCounts, 10),
      source_type_activity: topEntries(currentSourceCounts, 10),
      topic_sections: topicSections,
      source_appendix: topSources,
      empty: current.length === 0
    }, requestId);
  } catch (error) {
    console.error("[briefings/generate]", error);
    return fail("Failed to generate briefing.", "BRIEFING_GENERATE_FAILED", 500, requestId);
  }
}
