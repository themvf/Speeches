// Single source of truth for the CAPITAL_FORMATION vocabulary (see CLAUDE.md).
//
// The data lives in capital-formation-vocabulary.json and is consumed by every
// surface that used to hold its own copy:
//   - topic-rule-recommendations.ts  (seeds rss_topic_rules keywords)
//   - theme-intelligence.ts          (PRODUCT_FOCUS_AREAS + PRODUCT_TAXONOMY)
//   - server/gdelt-doc.ts            (GKG query terms + pattern aliases)
//   - server/stored-category-evidence.ts (pattern aliases)
//   - components/intelbeta-dashboard.tsx (document fallback regex)
//   - neon_feeds.py                  (via capital_formation_vocabulary.py)
//
// Those seven copies had already drifted apart once, which is the whole reason
// this module exists. Never fork the data back into a consumer: add terms here
// and let every surface pick them up.
//
// The JSON is imported statically (webpack inlines it at build time, so there
// is no runtime fs read to break under Vercel's file tracing); the import
// attribute keeps the same file loadable under plain `node --test`.
import vocabulary from "./capital-formation-vocabulary.json" with { type: "json" };

export type CapitalFormationFocusArea = {
  id: string;
  label: string;
  weight: number;
  normalizedThemes: readonly string[];
  rawPatterns: readonly string[];
  queryTerms: readonly string[];
};

export const CAPITAL_FORMATION_TOPIC_KEY: string = vocabulary.topicKey;
export const CAPITAL_FORMATION_LABEL: string = vocabulary.label;
export const CAPITAL_FORMATION_SORT_ORDER: number = vocabulary.sortOrder;
export const CAPITAL_FORMATION_FOCUS: string = vocabulary.focus;
export const CAPITAL_FORMATION_KEYWORDS: readonly string[] = vocabulary.keywords;
export const CAPITAL_FORMATION_BROAD_TERMS: readonly string[] = vocabulary.broadTerms;
export const CAPITAL_FORMATION_NOTES: readonly string[] = vocabulary.notes;
export const CAPITAL_FORMATION_FOCUS_AREAS: readonly CapitalFormationFocusArea[] =
  vocabulary.focusAreas as CapitalFormationFocusArea[];
export const CAPITAL_FORMATION_PATTERN_ALIASES: Readonly<Record<string, readonly string[]>> =
  vocabulary.patternAliases;

/** Focus-area id -> the natural-language terms matched against visible text
 * (URL, source, headline) by the GDELT surfaces. Shaped for
 * CATEGORY_DOC_QUERY_TERMS in gdelt-doc.ts. */
export function capitalFormationQueryTerms(): Record<string, readonly string[]> {
  const out: Record<string, readonly string[]> = {};
  for (const area of CAPITAL_FORMATION_FOCUS_AREAS) out[area.id] = area.queryTerms;
  return out;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Alternation regex over the keyword list, for the one consumer that needs a
 * plain regex rather than the topic-rule matcher (intelbeta-dashboard.tsx's
 * document fallback).
 *
 * Boundaries are `(?<![a-z0-9]) ... (?![a-z0-9])` rather than \b, mirroring
 * keywordPattern() in intel-topic-matching.ts. \b would break on every term
 * ending in punctuation - "Rule 506(c)" ends in ")", and \b between ")" and a
 * following space is not a boundary at all, so the term could never match and
 * the bare "Rule 506" alternative would win instead.
 *
 * Multi-word terms accept the same separator set as the real matcher, and
 * alternatives are sorted longest-first so "Rule 506(c)" beats "Rule 506" and
 * "Rule 144A" beats "Rule 144". */
export function capitalFormationKeywordRegex(): RegExp {
  const alternatives = [...CAPITAL_FORMATION_KEYWORDS]
    .sort((a, b) => b.length - a.length)
    .map((keyword) =>
      keyword
        .toLowerCase()
        .trim()
        .split(/\s+/)
        .map(escapeRegExp)
        .join("[\\s\\-\\u2013\\u2014_/]+")
    );
  return new RegExp(`(?<![a-z0-9])(?:${alternatives.join("|")})(?![a-z0-9])`);
}
