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

const MONTHS: Readonly<Record<string, string>> = {
  january: "01", february: "02", march: "03", april: "04", may: "05", june: "06",
  july: "07", august: "08", september: "09", october: "10", november: "11", december: "12",
};

/**
 * documents.published_date is TEXT and holds at least two shapes, both live in
 * the corpus today: "2026-08-19T02:07:24Z" from Bloomberg-style ingests and
 * "August 18, 2026" from newsapi ones.
 *
 * They cannot be compared as strings. "August 18, 2026" >= "2026-07-22" is
 * true only because "A" sorts above "2", which passes every such row through a
 * lower bound regardless of its year while failing every upper bound - so the
 * newsapi half of the corpus, its largest source at ~6,000 documents, was
 * silently dropped from chips. Normalize to YYYY-MM-DD, then compare.
 *
 * Returns null for anything unrecognized, which the caller drops rather than
 * guesses at.
 */
export function normalizePublishedDate(value: string): string | null {
  const raw = (value ?? "").trim();
  if (!raw) return null;

  const iso = /^(\d{4})-(\d{2})-(\d{2})/.exec(raw);
  if (iso) return `${iso[1]}-${iso[2]}-${iso[3]}`;

  const written = /^([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})$/.exec(raw);
  if (written) {
    const month = MONTHS[written[1].toLowerCase()];
    if (month) return `${written[3]}-${month}-${written[2].padStart(2, "0")}`;
  }
  return null;
}

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
  newsapi_article: "News",
  bloomberg_public_article: "Bloomberg",
  wsj_dow_jones: "WSJ",
  substack_public_article: "Substack",
  dark_reading_article: "Dark Reading",
};

export function sourceLabel(sourceKind: string): string {
  if (SOURCE_LABELS[sourceKind]) return SOURCE_LABELS[sourceKind];
  // Fall back to a readable form of the kind itself rather than inventing one.
  return sourceKind.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Group rows into per-ticker chips, newest first.
 *
 * Windowing and ordering happen here rather than in SQL because
 * published_date cannot be range-filtered as text (see
 * normalizePublishedDate). The query is a permissive prefilter; this is where
 * the window actually holds.
 */
export function buildCorpusChips(
  rows: CorpusEventInput[],
  window?: { since: string; until: string },
): Map<string, CorpusEventChip[]> {
  const byTicker = new Map<string, CorpusEventChip[]>();

  const dated = rows
    .flatMap((row) => {
      const date = normalizePublishedDate(row.published_date);
      if (!date) return [];
      if (window && (date < window.since || date > window.until)) return [];
      return [{ row, date }];
    })
    .sort((left, right) => right.date.localeCompare(left.date));

  for (const { row, date } of dated) {
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
      publishedDate: date,
      url: row.url,
      subject,
    });
    byTicker.set(row.ticker, chips);
  }
  return byTicker;
}
