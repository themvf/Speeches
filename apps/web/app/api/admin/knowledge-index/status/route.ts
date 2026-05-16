import { NextResponse } from "next/server";
import { loadVectorStoreState } from "@/lib/server/vector-state";
import { loadCorpusDocuments } from "@/lib/server/data-store";

export const dynamic = "force-dynamic";

const SOURCE_KIND_TO_ORG: Record<string, string> = {
  sec_speech: "sec",
  sec_tm_faq: "sec",
  sec_enforcement_litigation: "sec",
  finra_regulatory_notice: "finra",
  finra_key_topic: "finra",
  finra_comment_letter: "finra",
  finra_awc: "finra",
  doj_usao_press_release: "doj",
  federal_reserve_speech_testimony: "federal_reserve",
  cftc_press_release: "cftc",
  cftc_public_statement_remark: "cftc",
  treasury_featured_story: "treasury",
  treasury_press_release: "treasury",
  treasury_statement_remark: "treasury",
  sifma_news_item: "sifma",
  jdsupra_article: "trade_media",
  investmentnews_article: "trade_media",
  citywire_article: "trade_media",
  congress_crs_product: "congress",
};

const ORG_LABELS: Record<string, string> = {
  sec: "SEC",
  finra: "FINRA",
  doj: "DOJ",
  federal_reserve: "Federal Reserve",
  cftc: "CFTC",
  treasury: "Treasury",
  sifma: "SIFMA",
  trade_media: "Trade Media",
  congress: "Congress",
};

export async function GET(): Promise<NextResponse> {
  try {
    const [vectorState, corpusDocs] = await Promise.all([
      loadVectorStoreState(),
      loadCorpusDocuments(),
    ]);

    // Count corpus docs per org
    const corpusCountByOrg: Record<string, number> = {};
    for (const doc of corpusDocs) {
      const sourceKind = String(doc.metadata?.source_kind ?? "").trim();
      const orgKey = SOURCE_KIND_TO_ORG[sourceKind] ?? "other";
      if (orgKey === "other") continue;
      corpusCountByOrg[orgKey] = (corpusCountByOrg[orgKey] ?? 0) + 1;
    }

    // Merge corpus orgs + state orgs
    const allOrgKeys = new Set([
      ...Object.keys(corpusCountByOrg),
      ...Object.keys(vectorState.stores ?? {}),
    ]);

    const orgs = Array.from(allOrgKeys).sort().map((orgKey) => {
      const store = (vectorState.stores ?? {})[orgKey];
      const corpusCount = corpusCountByOrg[orgKey] ?? 0;
      const indexedCount = store?.doc_count_indexed ?? 0;
      const pending = Math.max(0, corpusCount - indexedCount);
      return {
        org_key: orgKey,
        org_label: store?.org_label ?? ORG_LABELS[orgKey] ?? orgKey.toUpperCase(),
        vector_store_id: store?.vector_store_id ?? null,
        corpus_count: corpusCount,
        indexed_count: indexedCount,
        pending,
        updated_at: store?.updated_at ?? null,
        last_sync: (store as unknown as Record<string, unknown> | undefined)?.last_sync ?? null,
      };
    });

    return NextResponse.json({
      ok: true,
      data: {
        updated_at: vectorState.updated_at ?? null,
        orgs,
      },
    });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}
