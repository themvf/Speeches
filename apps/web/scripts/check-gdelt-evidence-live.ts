import assert from "node:assert/strict";
import { fetchGdeltGkgEvidenceForProductCategory } from "../lib/server/gdelt-gkg.ts";
import { PRODUCT_CATEGORY_ORDER, type ProductCategory } from "../lib/theme-intelligence.ts";

const liveRequired = process.env.GDELT_LIVE_REQUIRED === "1";
const category = (process.env.GDELT_EVIDENCE_CATEGORY ?? "AML") as ProductCategory;
const forbiddenAmlTerms = new Set(["EMBARGO", "RESTRICTIONS"]);

assert.ok(PRODUCT_CATEGORY_ORDER.includes(category), `Unknown GDELT evidence category: ${category}`);

const evidence = await fetchGdeltGkgEvidenceForProductCategory(category, null, { archiveCount: 1 });

assert.ok(Array.isArray(evidence), `${category} evidence response must be an array`);

if (liveRequired) {
  assert.ok(evidence.length > 0, `expected at least one live ${category} evidence article`);
}

for (const article of evidence) {
  const visibleSourceText = `${article.url ?? ""} ${article.headline}`.toUpperCase();
  assert.ok(article.id, "evidence article must include an id");
  assert.ok(article.headline, `evidence article ${article.id} must include a headline`);
  assert.ok(article.source, `evidence article ${article.id} must include a source`);
  assert.ok(article.url?.startsWith("http"), `evidence article ${article.id} must include a source URL`);
  assert.ok(article.focusAreaId, `evidence article ${article.id} must include a ${category} focus area`);
  assert.ok(article.focusAreaLabel, `evidence article ${article.id} must include a ${category} focus label`);
  assert.ok(article.matchedTerms?.length, `evidence article ${article.id} must include matched ${category} terms`);
  if (category === "AML") {
    assert.equal(
      article.matchedTerms.some((term) => forbiddenAmlTerms.has(term)),
      false,
      `evidence article ${article.id} matched broad sanctions-adjacent terms instead of strict AML terms`
    );
  }
  if (category === "AML" && article.matchedTerms.includes("SANCTIONS")) {
    assert.match(
      visibleSourceText,
      /SANCTION|OFAC/,
      `evidence article ${article.id} used broad GDELT sanctions tagging without visible sanctions source text`
    );
  }
}

const focusCounts = evidence.reduce<Record<string, number>>((counts, article) => {
  const label = article.focusAreaLabel ?? "Unclassified";
  counts[label] = (counts[label] ?? 0) + 1;
  return counts;
}, {});

console.log(
  JSON.stringify(
    {
      ok: true,
      category,
      articles: evidence.length,
      focusCounts,
      liveRequired
    },
    null,
    2
  )
);
