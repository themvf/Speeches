import { loadCustomDocuments, loadEnrichmentState, parseComparableDate } from "@/lib/server/data-store";
import { createRequestId, fail, normalizeText, ok } from "@/lib/server/api-utils";
import type { CustomDocumentRecord, EnrichmentEntry } from "@/lib/server/types";

export const runtime = "nodejs";

interface NoticeCommentItem {
  document_id: string;
  title: string;
  commenter_name: string;
  commenter_org: string;
  speaker: string;
  url: string;
  comment_url: string;
  pdf_url: string;
  published_at: string;
  summary: string;
  tags: string[];
  keywords: string[];
  enrichment_status: string;
  review_decision: string;
  comment_position: {
    label: string;
    confidence: number;
    rationale: string;
  };
}

interface NoticeGroupItem {
  notice_key: string;
  notice_document_id: string;
  notice_number: string;
  title: string;
  summary: string;
  organization: string;
  url: string;
  pdf_url: string;
  published_at: string;
  effective_date: string;
  comment_deadline: string;
  tags: string[];
  keywords: string[];
  enrichment_status: string;
  review_decision: string;
  comment_count: number;
  latest_comment_at: string;
  comments: NoticeCommentItem[];
}

interface NoticeCommentsResponse {
  groups: NoticeGroupItem[];
  totals: {
    notices: number;
    comments: number;
    enriched_comments: number;
    pending_review_comments: number;
  };
}

const ENRICHED_STATUSES = new Set(["enriched", "fallback_enriched", "reviewed"]);

function splitCsv(value: unknown): string[] {
  return String(value ?? "")
    .split(",")
    .map((item) => normalizeText(item))
    .filter(Boolean);
}

function dedupList(items: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const item of items) {
    const value = normalizeText(item);
    const key = value.toLowerCase();
    if (!value || seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(value);
  }
  return out;
}

function enrichmentStatus(entry: EnrichmentEntry | undefined): string {
  return normalizeText(entry?.status || "not_enriched") || "not_enriched";
}

function reviewDecision(entry: EnrichmentEntry | undefined): string {
  return normalizeText(entry?.review?.decision || "pending") || "pending";
}

function stanceLabel(entry: EnrichmentEntry | undefined): string {
  const raw = entry?.enrichment?.stance;
  if (!raw || typeof raw !== "object") {
    return "";
  }
  return normalizeText((raw as Record<string, unknown>).label || "");
}

function commentPosition(entry: EnrichmentEntry | undefined): NoticeCommentItem["comment_position"] {
  const raw = entry?.enrichment?.comment_position;
  if (raw && typeof raw === "object") {
    const label = normalizeText((raw as Record<string, unknown>).label || "").toLowerCase();
    const confidence = Number.parseFloat(String((raw as Record<string, unknown>).confidence ?? "0")) || 0;
    const rationale = normalizeText((raw as Record<string, unknown>).rationale || "");
    if (label) {
      return {
        label,
        confidence: Math.max(0, Math.min(1, confidence)),
        rationale
      };
    }
  }

  const legacy = stanceLabel(entry).toLowerCase();
  if (legacy === "supportive") {
    return {
      label: "supportive",
      confidence: 0.42,
      rationale: "Derived from legacy enrichment stance while comment-position enrichment is unavailable."
    };
  }
  if (legacy === "critical") {
    return {
      label: "opposed",
      confidence: 0.42,
      rationale: "Derived from legacy enrichment stance while comment-position enrichment is unavailable."
    };
  }
  if (legacy === "neutral") {
    return {
      label: "neutral",
      confidence: 0.35,
      rationale: "Derived from legacy enrichment stance while comment-position enrichment is unavailable."
    };
  }
  if (legacy === "cautious") {
    return {
      label: "unclear",
      confidence: 0.25,
      rationale: "Legacy stance is cautious, which does not map cleanly to support or opposition."
    };
  }

  return { label: "unclear", confidence: 0, rationale: "" };
}

function summaryFor(record: CustomDocumentRecord, entry: EnrichmentEntry | undefined): string {
  const enriched = normalizeText(entry?.enrichment?.summary || "");
  if (enriched) {
    return enriched;
  }

  const raw = String(record.content?.full_text || "");
  const blocks = raw
    .split(/\n\s*\n/)
    .map((item) => normalizeText(item))
    .filter(Boolean);

  for (const block of blocks) {
    const looksLikeHeader =
      /Source URL:/i.test(block) &&
      /(Notice Number:|Published Date:|Commenter:|Professional Affiliation:|Date:|Title:)/i.test(block);
    if (!looksLikeHeader) {
      return block.slice(0, 420);
    }
  }

  return normalizeText(record.metadata?.title || "").slice(0, 420);
}

function noticeGroupKey(metadata: CustomDocumentRecord["metadata"]): string {
  const noticeNumber = normalizeText(metadata.notice_number || "");
  if (noticeNumber) {
    return noticeNumber.toLowerCase();
  }

  const noticeUrl = normalizeText(metadata.notice_url || metadata.source_notice_url || metadata.source_index_url || metadata.url || "");
  if (noticeUrl) {
    return noticeUrl.toLowerCase();
  }

  const title = normalizeText(metadata.notice_title || metadata.title || "");
  return title.toLowerCase();
}

function buildNoticeTags(record: CustomDocumentRecord, entry: EnrichmentEntry | undefined): string[] {
  const metadataTags = splitCsv(record.metadata?.tags || "");
  const enrichTags = Array.isArray(entry?.enrichment?.tags)
    ? entry.enrichment.tags.map((item) => normalizeText(item)).filter(Boolean)
    : [];
  return dedupList([...metadataTags, ...enrichTags]);
}

function buildKeywords(entry: EnrichmentEntry | undefined): string[] {
  if (!Array.isArray(entry?.enrichment?.keywords)) {
    return [];
  }
  return dedupList(entry.enrichment.keywords.map((item) => normalizeText(item)).filter(Boolean));
}

export async function GET() {
  const requestId = createRequestId();

  try {
    const [customPayload, enrichmentState] = await Promise.all([loadCustomDocuments(), loadEnrichmentState()]);
    const entries = enrichmentState.entries || {};

    const groups = new Map<string, NoticeGroupItem>();

    for (const record of customPayload.documents || []) {
      const metadata = record.metadata || ({} as CustomDocumentRecord["metadata"]);
      if (metadata.source_kind !== "finra_regulatory_notice") {
        continue;
      }

      const docId = normalizeText(metadata.document_id || "");
      const entry = entries[docId];
      const key = noticeGroupKey(metadata);
      const publishedAt = normalizeText(metadata.published_date || metadata.date || "");

      groups.set(key, {
        notice_key: key,
        notice_document_id: docId,
        notice_number: normalizeText(metadata.notice_number || ""),
        title: normalizeText(metadata.title || metadata.notice_title || "FINRA Regulatory Notice"),
        summary: summaryFor(record, entry),
        organization: normalizeText(metadata.organization || "FINRA") || "FINRA",
        url: normalizeText(metadata.url || metadata.notice_url || ""),
        pdf_url: normalizeText(metadata.pdf_url || ""),
        published_at: publishedAt,
        effective_date: normalizeText(metadata.effective_date || ""),
        comment_deadline: normalizeText(metadata.comment_deadline || ""),
        tags: buildNoticeTags(record, entry),
        keywords: buildKeywords(entry),
        enrichment_status: enrichmentStatus(entry),
        review_decision: reviewDecision(entry),
        comment_count: 0,
        latest_comment_at: "",
        comments: []
      });
    }

    for (const record of customPayload.documents || []) {
      const metadata = record.metadata || ({} as CustomDocumentRecord["metadata"]);
      if (metadata.source_kind !== "finra_comment_letter") {
        continue;
      }

      const docId = normalizeText(metadata.document_id || "");
      const entry = entries[docId];
      const key = noticeGroupKey(metadata);
      const publishedAt = normalizeText(metadata.published_date || metadata.date || "");

      const existing = groups.get(key);
      const group: NoticeGroupItem =
        existing ||
        {
          notice_key: key,
          notice_document_id: "",
          notice_number: normalizeText(metadata.notice_number || ""),
          title: normalizeText(metadata.notice_title || metadata.title || "FINRA Notice"),
          summary: "",
          organization: "FINRA",
          url: normalizeText(metadata.notice_url || metadata.source_notice_url || metadata.source_index_url || ""),
          pdf_url: "",
          published_at: "",
          effective_date: "",
          comment_deadline: "",
          tags: [],
          keywords: [],
          enrichment_status: "not_enriched",
          review_decision: "pending",
          comment_count: 0,
          latest_comment_at: "",
          comments: []
        };

      group.comments.push({
        document_id: docId,
        title: normalizeText(metadata.title || "Comment Letter"),
        commenter_name: normalizeText(metadata.commenter_name || ""),
        commenter_org: normalizeText(metadata.commenter_org || ""),
        speaker: normalizeText(metadata.speaker || ""),
        url: normalizeText(metadata.url || ""),
        comment_url: normalizeText(metadata.comment_url || metadata.url || ""),
        pdf_url: normalizeText(metadata.pdf_url || ""),
        published_at: publishedAt,
        summary: summaryFor(record, entry),
        tags: buildNoticeTags(record, entry),
        keywords: buildKeywords(entry),
        enrichment_status: enrichmentStatus(entry),
        review_decision: reviewDecision(entry),
        comment_position: commentPosition(entry)
      });

      group.comment_count = group.comments.length;
      if (parseComparableDate(publishedAt) >= parseComparableDate(group.latest_comment_at)) {
        group.latest_comment_at = publishedAt;
      }

      if (!existing) {
        groups.set(key, group);
      }
    }

    const orderedGroups = [...groups.values()]
      .map((group) => ({
        ...group,
        comments: [...group.comments].sort(
          (a, b) => parseComparableDate(b.published_at) - parseComparableDate(a.published_at)
        ),
        comment_count: group.comments.length
      }))
      .sort((a, b) => {
        const noticeDiff = parseComparableDate(b.published_at) - parseComparableDate(a.published_at);
        if (noticeDiff !== 0) {
          return noticeDiff;
        }
        const latestCommentDiff = parseComparableDate(b.latest_comment_at) - parseComparableDate(a.latest_comment_at);
        if (latestCommentDiff !== 0) {
          return latestCommentDiff;
        }
        return a.notice_number.localeCompare(b.notice_number);
      });

    const allComments = orderedGroups.flatMap((group) => group.comments);
    const payload: NoticeCommentsResponse = {
      groups: orderedGroups,
      totals: {
        notices: orderedGroups.length,
        comments: allComments.length,
        enriched_comments: allComments.filter((item) => ENRICHED_STATUSES.has(item.enrichment_status)).length,
        pending_review_comments: allComments.filter(
          (item) =>
            ENRICHED_STATUSES.has(item.enrichment_status) &&
            !["accepted", "edited", "rejected"].includes(item.review_decision)
        ).length
      }
    };

    return ok(payload, requestId);
  } catch (error) {
    return fail(
      `Failed to load FINRA notices/comments: ${error instanceof Error ? error.message : "Unknown error"}`,
      "NOTICE_COMMENT_LOAD_FAILED",
      500,
      requestId
    );
  }
}
