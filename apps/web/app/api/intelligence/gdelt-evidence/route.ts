import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import { fetchGdeltEvidenceForProfile } from "@/lib/server/gdelt-doc";
import { buildIntelligenceSignalsData } from "@/lib/server/intelligence-signals";
import type { IntelligenceEvidenceArticle } from "@/lib/intelligence-types";

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
  const profileId = searchParams.get("profileId") ?? "";
  const data = buildIntelligenceSignalsData();
  const profile = data.profiles.find((item) => item.id === profileId);

  if (!profile) {
    return fail("Unknown intelligence profile.", "UNKNOWN_PROFILE", 404, requestId);
  }

  const gdeltEvidence = await fetchGdeltEvidenceForProfile(profile);
  const evidence = gdeltEvidence.length > 0 ? gdeltEvidence : profile.evidence;
  const source = gdeltEvidence.length > 0 ? "gdelt-doc" : "seed";

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
