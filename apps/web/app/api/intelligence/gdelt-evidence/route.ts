import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import { fetchGdeltGkgEvidenceForProductCategory, fetchGdeltGkgEvidenceForProfile } from "@/lib/server/gdelt-gkg";
import { buildIntelligenceSignalsData } from "@/lib/server/intelligence-signals";
import type { IntelligenceEvidenceArticle } from "@/lib/intelligence-types";
import { PRODUCT_CATEGORY_ORDER, PRODUCT_CATEGORY_LABELS, type ProductCategory } from "@/lib/theme-intelligence";

export const runtime = "nodejs";

function sourceDistributionFromEvidence(evidence: readonly IntelligenceEvidenceArticle[]) {
  const counts = new Map<string, number>();

  for (const article of evidence) {
    counts.set(article.source, (counts.get(article.source) ?? 0) + 1);
  }

  return Array.from(counts, ([source, count]) => ({ source, count }))
    .sort((a, b) => b.count - a.count || a.source.localeCompare(b.source))
    .slice(0, 10);
}

export async function GET(request: Request) {
  const requestId = createRequestId();
  const { searchParams } = new URL(request.url);
  const category = searchParams.get("category") ?? "";
  const focusId = searchParams.get("focusId") ?? "";
  const profileId = searchParams.get("profileId") ?? "";

  if (category) {
    if (!PRODUCT_CATEGORY_ORDER.includes(category as ProductCategory)) {
      return fail("Unknown intelligence category.", "UNKNOWN_CATEGORY", 404, requestId);
    }

    let liveEvidence: IntelligenceEvidenceArticle[] = [];
    try {
      liveEvidence = await fetchGdeltGkgEvidenceForProductCategory(category as ProductCategory, focusId || null);
    } catch {
      liveEvidence = [];
    }

    return ok({
      category,
      label: PRODUCT_CATEGORY_LABELS[category as ProductCategory],
      focusId: focusId || null,
      source: liveEvidence.length > 0 ? "gdelt-gkg" : "seed",
      evidence: liveEvidence,
      coverage: {
        totalArticles: liveEvidence.length,
        sourceCount: new Set(liveEvidence.map((article) => article.source)).size
      },
      sourceDistribution: sourceDistributionFromEvidence(liveEvidence)
    }, requestId);
  }

  const data = buildIntelligenceSignalsData();
  const profile = data.profiles.find((item) => item.id === profileId);

  if (!profile) {
    return fail("Unknown intelligence profile.", "UNKNOWN_PROFILE", 404, requestId);
  }

  let liveEvidence: IntelligenceEvidenceArticle[] = [];
  try {
    liveEvidence = await fetchGdeltGkgEvidenceForProfile(profile);
  } catch {
    liveEvidence = [];
  }

  const evidence = liveEvidence.length > 0 ? liveEvidence : profile.evidence;
  const source = liveEvidence.length > 0 ? "gdelt-gkg" : "seed";

  return ok({
    profileId,
    source,
    evidence,
    coverage: {
      totalArticles: evidence.length,
      sourceCount: new Set(evidence.map((article) => article.source)).size
    },
    sourceDistribution: sourceDistributionFromEvidence(evidence)
  }, requestId);
}
