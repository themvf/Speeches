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

const endpoint = process.env.AML_EVIDENCE_ENDPOINT_URL ?? "https://speeches-zeta.vercel.app/api/intelligence/gdelt-evidence?category=AML";
const response = await fetch(endpoint, {
  cache: "no-store",
  headers: {
    "user-agent": "PolicyResearchHub/1.0 AML evidence endpoint smoke test"
  }
});

assert.equal(response.ok, true, `AML evidence endpoint failed with HTTP ${response.status}`);

const payload = (await response.json()) as AmlEvidenceResponse;
assert.equal(payload.ok, true, payload.error ?? "AML evidence endpoint returned ok=false");
assert.ok(payload.data, "AML evidence endpoint must include data");

const evidence = payload.data.evidence ?? [];
assert.ok(evidence.length > 0, "AML evidence endpoint must return at least one article");
assert.equal(payload.data.coverage?.totalArticles, evidence.length, "coverage.totalArticles must match evidence length");
assert.ok((payload.data.coverage?.sourceCount ?? 0) > 0, "AML evidence endpoint must report at least one source");
assert.ok(["stored-news", "gdelt-gkg"].includes(payload.data.source ?? ""), "AML evidence source must be stored-news or gdelt-gkg");

for (const article of evidence.slice(0, 10)) {
  assert.ok(article.id, "AML evidence article must include id");
  assert.ok(article.headline, `AML evidence article ${article.id} must include headline`);
  assert.ok(article.source, `AML evidence article ${article.id} must include source`);
  assert.ok(article.timestamp, `AML evidence article ${article.id} must include timestamp`);
  assert.ok(article.url?.startsWith("http"), `AML evidence article ${article.id} must include source URL`);
  assert.ok(article.focusAreaId, `AML evidence article ${article.id} must include focusAreaId`);
  assert.ok(article.focusAreaLabel, `AML evidence article ${article.id} must include focusAreaLabel`);
  assert.ok(article.matchedTerms?.length, `AML evidence article ${article.id} must include matchedTerms`);
}

const focusCounts = evidence.reduce<Record<string, number>>((counts, article) => {
  const label = article.focusAreaLabel ?? "Unclassified";
  counts[label] = (counts[label] ?? 0) + 1;
  return counts;
}, {});

console.log(JSON.stringify({
  ok: true,
  endpoint,
  source: payload.data.source,
  articles: evidence.length,
  sourceCount: payload.data.coverage?.sourceCount,
  focusCounts,
  sourceDistribution: payload.data.sourceDistribution ?? []
}, null, 2));
