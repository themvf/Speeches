import type { CorpusEventChip } from "@/lib/server/types";

/**
 * Pure chip-building for corpus events. Kept free of any database import so
 * the gating rules below can be tested directly - they are the part that
 * matters, because they decide what gets asserted about a real company on a
 * public page.
 */

export interface CorpusEventInput {
  ticker: string;
  document_id: string;
  title: string;
  source_kind: string;
  published_date: string;
  url: string;
  confidence: number;
}

export const MAX_CHIPS_PER_TICKER = 2;

/** Title-match confidence written by index_document_tickers.py. */
export const SUBJECT_CONFIDENCE = 1.0;

/**
 * Source kinds where a chip carries an accusation about a real company. These
 * show only when the company was named in the document's TITLE - the signal
 * that it is the subject, rather than one name among many in the body.
 *
 * Taken from the corpus's actual source_kind values (/api/metrics
 * by_source_kind, checked 2026-08-21), not from what the kinds ought to be
 * called. The first version of this list was guessed: it named
 * `doj_press_release` and `cftc_enforcement`, neither of which exists, while
 * missing `doj_usao_press_release` - at 1,667 documents the single largest
 * enforcement source in the corpus, and therefore ungated.
 */
export const ACCUSATORY_SOURCE_KINDS: ReadonlySet<string> = new Set([
  "doj_usao_press_release", // 1,667 documents
  "sec_enforcement_litigation", // 1,276
  "finra_awc", // 869
]);

const SOURCE_LABELS: Readonly<Record<string, string>> = {
  sec_speech: "SEC speech",
  sec_enforcement_litigation: "SEC enforcement",
  sec_press_release: "SEC release",
  finra_awc: "FINRA action",
  finra_regulatory_notice: "FINRA notice",
  cftc_enforcement: "CFTC enforcement",
  doj_usao_press_release: "DOJ charge",
  pcaob_update: "PCAOB update",
  msrb_press_release: "MSRB release",
  occ_news_release: "OCC release",
  fdic_press_release: "FDIC release",
  cfpb_newsroom: "CFPB release",
  nydfs_press_release: "NYDFS release",
  rule_comment: "Rule comment",
};

export function sourceLabel(sourceKind: string): string {
  if (SOURCE_LABELS[sourceKind]) return SOURCE_LABELS[sourceKind];
  // Fall back to a readable form of the kind itself rather than inventing one.
  return sourceKind.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Group rows into per-ticker chips. Rows must arrive newest-first; the cap is
 * applied in arrival order so the freshest documents win.
 */
export function buildCorpusChips(rows: CorpusEventInput[]): Map<string, CorpusEventChip[]> {
  const byTicker = new Map<string, CorpusEventChip[]>();
  for (const row of rows) {
    const subject = Number(row.confidence) >= SUBJECT_CONFIDENCE;
    // A mention is not a subject: an enforcement chip on a company merely
    // named in the body would read as an accusation we cannot support.
    if (!subject && ACCUSATORY_SOURCE_KINDS.has(row.source_kind)) continue;
    if (!row.title || !row.url || !row.ticker) continue;

    const chips = byTicker.get(row.ticker) ?? [];
    if (chips.length >= MAX_CHIPS_PER_TICKER) continue;
    chips.push({
      documentId: row.document_id,
      title: row.title,
      sourceKind: row.source_kind,
      sourceLabel: sourceLabel(row.source_kind),
      publishedDate: row.published_date,
      url: row.url,
      subject,
    });
    byTicker.set(row.ticker, chips);
  }
  return byTicker;
}
