import { createRequestId, fail, normalizeText, ok } from "@/lib/server/api-utils";
import { loadCorpusDocuments, loadEnrichmentState, parseComparableDate, saveEnrichmentState } from "@/lib/server/data-store";
import { analysisToJsonValue, generateEnforcementAnalysis, jsonValueToAnalysis } from "@/lib/server/enforcement-analysis";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 55;

function toInt(value: unknown, fallback: number, minValue: number, maxValue: number): number {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  return Math.max(minValue, Math.min(maxValue, Number.isFinite(parsed) ? parsed : fallback));
}

function isSecEnforcementDoc(sourceKind: string, url: string): boolean {
  return sourceKind === "sec_enforcement_litigation" || url.includes("/enforcement-litigation/litigation-releases/");
}

export async function POST(request: Request) {
  const requestId = createRequestId();

  try {
    const body = await request.json().catch(() => ({})) as Record<string, unknown>;
    const limit = toInt(body.limit, 10, 1, 25);
    const mode = normalizeText(body.mode) === "all" ? "all" : "missing";
    const model = normalizeText(body.model);

    const [corpus, enrichmentState] = await Promise.all([loadCorpusDocuments(), loadEnrichmentState()]);
    const entries = enrichmentState.entries || {};
    const candidates = corpus
      .filter((doc) => {
        const metadata = doc.metadata || {};
        const docId = normalizeText(metadata.document_id);
        const sourceKind = normalizeText(metadata.source_kind);
        const url = normalizeText(metadata.url).toLowerCase();
        if (!docId || !isSecEnforcementDoc(sourceKind, url)) {
          return false;
        }
        if (mode === "all") {
          return true;
        }
        return !jsonValueToAnalysis(entries[docId]?.enforcement_analysis);
      })
      .sort((a, b) =>
        parseComparableDate(normalizeText(b.metadata?.published_date) || normalizeText(b.metadata?.date)) -
        parseComparableDate(normalizeText(a.metadata?.published_date) || normalizeText(a.metadata?.date))
      )
      .slice(0, limit);

    const processed: Array<{ document_id: string; title: string; ok: boolean; error?: string }> = [];
    for (const doc of candidates) {
      const metadata = doc.metadata || {};
      const docId = normalizeText(metadata.document_id);
      const title = normalizeText(metadata.title);
      try {
        const generated = await generateEnforcementAnalysis(doc, model);
        const existing = entries[docId] || {};
        entries[docId] = {
          doc_id: docId,
          organization: normalizeText(metadata.organization) || "SEC",
          org_key: "sec",
          title,
          speaker: normalizeText(metadata.speaker),
          date: normalizeText(metadata.published_date) || normalizeText(metadata.date),
          url: normalizeText(metadata.url),
          doc_type: normalizeText(metadata.doc_type) || "Litigation Release",
          word_count: Number.parseInt(String(metadata.word_count ?? "0"), 10) || 0,
          status: normalizeText(existing.status) || "enriched",
          error: "",
          model: generated.model,
          pipeline_version: normalizeText(existing.pipeline_version) || "v1",
          updated_at: new Date().toISOString(),
          enrichment: existing.enrichment || {
            summary: "",
            tags: [],
            keywords: [],
            entities: [],
            stance: {},
            comment_position: {},
            evidence_spans: [],
            confidence: 0,
          },
          review: existing.review || { decision: "pending", notes: "", reviewed_at: "" },
          sentiment: existing.sentiment,
          enforcement_analysis: analysisToJsonValue(generated.analysis, generated.model),
          reward: existing.reward,
          auto_review: existing.auto_review,
        };
        processed.push({ document_id: docId, title, ok: true });
      } catch (error) {
        processed.push({
          document_id: docId,
          title,
          ok: false,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }

    enrichmentState.entries = entries;
    const saved = processed.some((item) => item.ok)
      ? await saveEnrichmentState(enrichmentState)
      : { saved: false, local_saved: false, remote_saved: false };

    return ok({
      mode,
      limit,
      selected_count: candidates.length,
      saved_count: processed.filter((item) => item.ok).length,
      failed_count: processed.filter((item) => !item.ok).length,
      saved,
      processed,
    }, requestId);
  } catch (error) {
    console.error("[admin/enforcement-analysis]", error);
    return fail(
      `Failed to generate enforcement analyses: ${error instanceof Error ? error.message : "Unknown error"}`,
      "ENFORCEMENT_ANALYSIS_BATCH_FAILED",
      500,
      requestId
    );
  }
}
