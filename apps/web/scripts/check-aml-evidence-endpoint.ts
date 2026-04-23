import assert from "node:assert/strict";

type AmlEvidenceArticle = {
  id?: string;
  headline?: string;
  url?: string;
  source?: string;
  timestamp?: string;
  explanation?: string;
  matchedTerms?: string[];
  focusAreaId?: string;
  focusAreaLabel?: string;
};

type AmlEvidenceResponse = {
  ok: boolean;
  data?: {
    category?: string;
    source?: string;
    evidence?: AmlEvidenceArticle[];
    coverage?: {
      totalArticles?: number;
      sourceCount?: number;
    };
    sourceDistribution?: { source: string; count: number }[];
  };
  error?: string;
};

const category = process.env.EVIDENCE_CATEGORY ?? "AML";
const endpointBase = process.env.EVIDENCE_ENDPOINT_BASE ?? "https://speeches-zeta.vercel.app/api/intelligence/gdelt-evidence";
const endpoint =
  process.env.EVIDENCE_ENDPOINT_URL ??
  process.env.AML_EVIDENCE_ENDPOINT_URL ??
  `${endpointBase}?category=${encodeURIComponent(category)}`;
const expectedSource = process.env.EVIDENCE_EXPECTED_SOURCE;

const response = await fetch(endpoint, {
  cache: "no-store",
  headers: {
    "user-agent": "PolicyResearchHub/1.0 category evidence endpoint smoke test"
  }
});

assert.equal(response.ok, true, `${category} evidence endpoint failed with HTTP ${response.status}`);

const payload = (await response.json()) as AmlEvidenceResponse;
assert.equal(payload.ok, true, payload.error ?? `${category} evidence endpoint returned ok=false`);
assert.ok(payload.data, `${category} evidence endpoint must include data`);
assert.equal(payload.data.category, category, `${category} evidence endpoint returned the wrong category`);

const evidence = payload.data.evidence ?? [];
assert.ok(evidence.length > 0, `${category} evidence endpoint must return at least one article`);
assert.equal(payload.data.coverage?.totalArticles, evidence.length, "coverage.totalArticles must match evidence length");
assert.ok((payload.data.coverage?.sourceCount ?? 0) > 0, `${category} evidence endpoint must report at least one source`);
assert.ok(["gdelt-doc", "gdelt-gkg", "stored-news"].includes(payload.data.source ?? ""), `${category} evidence source must be gdelt-doc, gdelt-gkg, or stored-news`);

if (expectedSource) {
  assert.equal(payload.data.source, expectedSource, `${category} evidence endpoint should use ${expectedSource}`);
}

for (const article of evidence.slice(0, 10)) {
  assert.ok(article.id, `${category} evidence article must include id`);
  assert.ok(article.headline, `${category} evidence article ${article.id} must include headline`);
  assert.ok(article.source, `${category} evidence article ${article.id} must include source`);
  assert.ok(article.timestamp, `${category} evidence article ${article.id} must include timestamp`);
  assert.ok(article.url?.startsWith("http"), `${category} evidence article ${article.id} must include source URL`);
  assert.ok(article.focusAreaId, `${category} evidence article ${article.id} must include focusAreaId`);
  assert.ok(article.focusAreaLabel, `${category} evidence article ${article.id} must include focusAreaLabel`);
  assert.ok(article.matchedTerms?.length, `${category} evidence article ${article.id} must include matchedTerms`);
}

const focusCounts = evidence.reduce<Record<string, number>>((counts, article) => {
  const label = article.focusAreaLabel ?? "Unclassified";
  counts[label] = (counts[label] ?? 0) + 1;
  return counts;
}, {});

console.log(JSON.stringify({
  ok: true,
  category,
  endpoint,
  source: payload.data.source,
  articles: evidence.length,
  sourceCount: payload.data.coverage?.sourceCount,
  focusCounts,
  sourceDistribution: payload.data.sourceDistribution ?? []
}, null, 2));
