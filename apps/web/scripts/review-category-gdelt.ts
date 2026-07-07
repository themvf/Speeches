import { buildGdeltDocCategoryQueries, fetchGdeltDocEvidenceForProductCategory } from "../lib/server/gdelt-doc.ts";
import { fetchGdeltGkgEvidenceForProductCategory } from "../lib/server/gdelt-gkg.ts";
import { PRODUCT_CATEGORY_LABELS, type ProductCategory } from "../lib/theme-intelligence.ts";

type ReviewRow = {
  provider: string;
  focus: string;
  timestamp: string;
  source: string;
  headline: string;
  keywords: string;
  url: string;
};

const REVIEW_CATEGORIES: ProductCategory[] = ["AML", "CAPITAL_FORMATION"];

function escapePipe(value: string): string {
  return value.replace(/\|/g, "\\|").replace(/\s+/g, " ").trim();
}

function rowsForProvider(provider: string, rows: ReviewRow[]): string[] {
  if (rows.length === 0) {
    return ["| provider | focus | timestamp | source | headline | keywords | url |", "|---|---|---|---|---|---|---|", `| ${provider} | - | - | - | No accepted articles | - | - |`];
  }

  return [
    "| provider | focus | timestamp | source | headline | keywords | url |",
    "|---|---|---|---|---|---|---|",
    ...rows.map((row) => `| ${row.provider} | ${escapePipe(row.focus)} | ${escapePipe(row.timestamp)} | ${escapePipe(row.source)} | ${escapePipe(row.headline)} | ${escapePipe(row.keywords)} | ${escapePipe(row.url)} |`)
  ];
}

for (const category of REVIEW_CATEGORIES) {
  const [docEvidence, gkgEvidence] = await Promise.all([
    fetchGdeltDocEvidenceForProductCategory(category, null),
    fetchGdeltGkgEvidenceForProductCategory(category, null)
  ]);

  const docRows: ReviewRow[] = docEvidence.slice(0, 15).map((article) => ({
    provider: "gdelt-doc",
    focus: article.focusAreaLabel ?? "",
    timestamp: article.timestamp,
    source: article.source,
    headline: article.headline,
    keywords: (article.matchedTerms ?? []).join(", "),
    url: article.url ?? ""
  }));
  const gkgRows: ReviewRow[] = gkgEvidence.slice(0, 15).map((article) => ({
    provider: "gdelt-gkg",
    focus: article.focusAreaLabel ?? "",
    timestamp: article.timestamp,
    source: article.source,
    headline: article.headline,
    keywords: (article.matchedTerms ?? []).join(", "),
    url: article.url ?? ""
  }));
  const chosenRows = (docEvidence.length > 0 ? docRows : gkgRows).slice(0, 15);
  const chosenProvider = docEvidence.length > 0 ? "gdelt-doc" : gkgEvidence.length > 0 ? "gdelt-gkg" : "none";

  console.log(`\n## ${PRODUCT_CATEGORY_LABELS[category]} (${category})`);
  console.log(`DOC query count: ${buildGdeltDocCategoryQueries(category).length}`);
  console.log(`DOC accepted: ${docEvidence.length}`);
  console.log(`GKG accepted: ${gkgEvidence.length}`);
  console.log(`Chosen provider: ${chosenProvider}`);

  console.log("\n### Accepted Stream");
  console.log(rowsForProvider(chosenProvider, chosenRows).join("\n"));

  console.log("\n### DOC Accepted");
  console.log(rowsForProvider("gdelt-doc", docRows).join("\n"));

  console.log("\n### GKG Accepted");
  console.log(rowsForProvider("gdelt-gkg", gkgRows).join("\n"));
}
