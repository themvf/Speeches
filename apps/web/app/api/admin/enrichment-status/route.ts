import { NextResponse } from "next/server";
import { loadEnrichmentState, loadCorpusDocuments } from "@/lib/server/data-store";

export const dynamic = "force-dynamic";

const ORG_LABELS: Record<string, string> = {
  sec: "SEC",
  finra: "FINRA",
  doj: "DOJ",
  federal_reserve: "Federal Reserve",
  cisa: "CISA",
  cftc: "CFTC",
  treasury: "Treasury",
  sifma: "SIFMA",
  ici: "ICI",
  isda: "ISDA",
  mfa: "MFA",
  fia: "FIA",
  aba: "ABA",
  bpi: "BPI",
  icba: "ICBA",
  lsta: "LSTA",
  trade_media: "Trade Media",
  krebs_on_security: "Krebs on Security",
  the_hacker_news: "The Hacker News",
  eset: "ESET",
  sophos: "Sophos",
  flashpoint: "Flashpoint",
  recorded_future: "Recorded Future",
  intel_471: "Intel 471",
  securityweek: "SecurityWeek",
  dark_reading: "Dark Reading",
  cyber: "Cyber",
  congress: "Congress",
  news: "News",
};

export async function GET(): Promise<NextResponse> {
  try {
    const [enrichmentState, corpusDocs] = await Promise.all([
      loadEnrichmentState(),
      loadCorpusDocuments(),
    ]);

    const entries = enrichmentState.entries ?? {};
    const totalDocs = corpusDocs.length;

    let enriched = 0;
    let failed = 0;
    let pending = 0;

    for (const doc of corpusDocs) {
      const entry = entries[doc.metadata.document_id];
      const status = entry?.status ?? "not_enriched";
      if (status === "enriched") enriched++;
      else if (status === "failed") failed++;
      else pending++;
    }

    // Per-org breakdown from enrichment entries
    const byOrgMap: Record<string, { total: number; enriched: number; failed: number; pending: number }> = {};
    for (const entry of Object.values(entries)) {
      const org = entry.org_key || "other";
      if (!byOrgMap[org]) byOrgMap[org] = { total: 0, enriched: 0, failed: 0, pending: 0 };
      byOrgMap[org].total++;
      if (entry.status === "enriched") byOrgMap[org].enriched++;
      else if (entry.status === "failed") byOrgMap[org].failed++;
      else byOrgMap[org].pending++;
    }
    const byOrg = Object.entries(byOrgMap)
      .map(([org_key, counts]) => ({
        org_key,
        org_label: ORG_LABELS[org_key] ?? org_key.toUpperCase(),
        ...counts,
      }))
      .sort((a, b) => b.total - a.total);

    // Top 25 most recently failed docs
    const failedDocs = Object.values(entries)
      .filter((e) => e.status === "failed")
      .sort((a, b) => (b.updated_at > a.updated_at ? 1 : -1))
      .slice(0, 25)
      .map((e) => ({
        doc_id: e.doc_id,
        title: e.title || e.doc_id,
        org_key: e.org_key,
        org_label: ORG_LABELS[e.org_key] ?? e.org_key?.toUpperCase() ?? "—",
        error: e.error || "Unknown error",
        updated_at: e.updated_at,
      }));

    return NextResponse.json({
      ok: true,
      data: {
        total: totalDocs,
        enriched,
        failed,
        pending,
        updated_at: enrichmentState.updated_at ?? null,
        by_org: byOrg,
        failed_docs: failedDocs,
      },
    });
  } catch (err) {
    console.error("[admin/enrichment-status]", err);
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}
