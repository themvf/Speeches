// Entity normalization / alias map (see CLAUDE.md). The alias data lives in
// entity-aliases.json - one shared config consumed by this module (TS path:
// neon.ts's prepareMentionBatch) and by entity_aliases.py (Python path:
// _normalize_enrichment_payload in run_financial_news_pipeline.py / app.py).
// Never fork per-language copies of the data; both sides must collapse the
// same alias pairs to the same normalized_value or watchlist/attention/trend
// counts fragment across name variants.
//
// The JSON is imported statically (webpack inlines it at build time, so
// there's no runtime fs read to break under Vercel's file tracing); the
// import attribute keeps the same file loadable under plain `node --test`.
import entityAliasConfig from "./entity-aliases.json" with { type: "json" };

// Mention-value normalization shared by every mention type. This is the
// exact normalization the intelligence_mentions.normalized_value column has
// always used (it lived in neon.ts as normalizeMention before this module
// existed); entity_aliases.py ports it byte-for-byte, so alias lookups on
// both sides happen in the same normalized space.
export function normalizeMentionValue(value: string): string {
  return String(value || "")
    .toLowerCase()
    .replace(/['"“”‘’]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

type EntityAliasEntry = { canonical: string; aliases: string[] };

function buildAliasLookup(entries: EntityAliasEntry[]): Map<string, string> {
  const lookup = new Map<string, string>();
  for (const entry of entries) {
    const canonical = String(entry.canonical || "").trim();
    if (!canonical) continue;
    // Canonical labels map to themselves so already-canonical mentions are
    // stable no matter how they're cased/punctuated in the source text.
    const keys = [canonical, ...(entry.aliases || [])];
    for (const alias of keys) {
      const key = normalizeMentionValue(String(alias || ""));
      if (!key) continue;
      // First entry wins on a duplicate alias; the test suite asserts the
      // config has no cross-entity duplicates so this branch stays dead.
      if (!lookup.has(key)) lookup.set(key, canonical);
    }
  }
  return lookup;
}

const ALIAS_LOOKUP = buildAliasLookup(entityAliasConfig.entities as EntityAliasEntry[]);

// Raw entity text -> canonical display label ("Securities and Exchange
// Commission" -> "SEC"), or the trimmed input unchanged when no alias
// matches. Callers should normalize the *returned* label to get the
// canonical normalized_value - that single path guarantees the stored
// normalized form is always normalize(canonical label).
export function canonicalEntityLabel(value: string): string {
  const trimmed = String(value || "").trim();
  if (!trimmed) return trimmed;
  return ALIAS_LOOKUP.get(normalizeMentionValue(trimmed)) ?? trimmed;
}

// Convenience for callers that only need the canonical normalized_value
// (e.g. the future watchlist lookup and the backfill script's SQL both key
// on this).
export function canonicalNormalizedEntityValue(value: string): string {
  return normalizeMentionValue(canonicalEntityLabel(value));
}

// Exposed for the backfill script / tests: every (alias normalized value ->
// {canonical label, canonical normalized value}) pair where the alias
// actually differs from its canonical form.
export function entityAliasPairs(): Array<{ aliasNormalized: string; canonicalLabel: string; canonicalNormalized: string }> {
  const pairs: Array<{ aliasNormalized: string; canonicalLabel: string; canonicalNormalized: string }> = [];
  for (const [aliasNormalized, canonicalLabel] of ALIAS_LOOKUP.entries()) {
    const canonicalNormalized = normalizeMentionValue(canonicalLabel);
    if (aliasNormalized !== canonicalNormalized) {
      pairs.push({ aliasNormalized, canonicalLabel, canonicalNormalized });
    }
  }
  return pairs;
}
