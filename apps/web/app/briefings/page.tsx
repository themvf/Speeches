import type { Metadata } from "next";
import { BriefingDashboard } from "@/components/briefing-dashboard";
import { buildDocumentListItems, buildDocumentsFacets, loadCorpusDocuments, loadEnrichmentState } from "@/lib/server/data-store";
import type { DocumentsFacets } from "@/lib/server/types";

export const revalidate = 300;

export const metadata: Metadata = {
  title: "Briefings | Policy Research Hub",
  description: "Generate tailored regulatory briefings from selected agencies, topics, source types, and dates."
};

const EMPTY_FACETS: DocumentsFacets = {
  sources: [],
  organizations: [],
  topics: [],
  key_topics: [],
  keywords: [],
  statuses: []
};

export default async function BriefingsPage() {
  let facets = EMPTY_FACETS;

  try {
    const [corpusDocs, enrichment] = await Promise.all([loadCorpusDocuments(), loadEnrichmentState()]);
    facets = buildDocumentsFacets(buildDocumentListItems(corpusDocs, enrichment));
  } catch {
    // Data source may be unavailable in a new local/Vercel environment. Render the shell.
  }

  return (
    <main className="mx-auto w-full max-w-7xl px-4 py-6 md:px-8">
      <BriefingDashboard facets={facets} />
    </main>
  );
}
