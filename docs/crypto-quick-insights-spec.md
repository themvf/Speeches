# Crypto quick insights: reusable briefing specification

Status: **implementation specification; the crypto integration below is not yet implemented.**

Date: 2026-09-20. Baseline: mobile X quick insights, commit `df99df5`.

## 1. Outcome

A mobile user selects Backpack, or any other tracked crypto, and can answer these questions in about 15 seconds:

1. What is happening in the saved conversation?
2. Why might it matter, and what remains uncertain?
3. Which posts support that interpretation?
4. How recent and complete is the evidence?

Adding another tracked asset must reuse the same API, matching pipeline, briefing builder, and components. Its identity and enabled capabilities come from configuration. No separate Backpack, ZCAT, or other coin dashboard is required.

This is an evidence briefing, not a trade recommendation, price forecast, or claim that X activity caused a market move.

## 2. What exists and what this spec adds

| Existing module | Reuse | Boundary |
| --- | --- | --- |
| `apps/web/components/x-signals-workspace.tsx` | Mobile search, short takeaways, publication windows, evidence expansion, retry states | Currently reads the topic-filtered RSS/X feed, not the crypto archive |
| `apps/web/lib/x-signal-insights.ts` | Deduplication, date handling, query-aware excerpts, valid-analysis gating | Current RSS IDs are numeric; crypto post IDs must remain strings |
| `GET /api/intel/feed?source=X` | Scoped read of up to 500 saved X timeline rows before topic filtering | Not a complete coin search and not a replacement for crypto collectors |
| `apps/web/lib/crypto-coins.json`, `crypto-coins.ts`, `crypto_coins.py` | Shared identity registry and matching behavior | Keep one registry shared by TypeScript and Python |
| `apps/web/lib/crypto-voices.ts` | Existing eligibility and contract-anchor rules | Preserve the distinction between exact contract evidence and provisional contextual evidence |
| `apps/web/lib/crypto-signals.ts` | Named deterministic rules and their explanations | Rule output alone does not establish causation or consensus |
| `/api/market/crypto/run`, `/history`, `/signals`, `/status` | Existing stored posts, market context, rules, coverage | Existing response limits and time semantics are not sufficient for every briefing metric |
| `apps/web/components/market/crypto-workbench.tsx`, `apps/web/lib/crypto-workbench.ts` | Existing workbench and URL state | Add a briefing surface without replacing research views or breaking old links |

The shipped X view answers a free-text question over tracked, topic-mapped timeline posts. The planned crypto view answers a registry-resolved asset question over the crypto archive. Keep those scopes explicit; a generic Backpack word match is not proof that a post concerns the configured Backpack token.

## 3. Mobile experience

Proposed entry: `/market/crypto?coin=BACKPACK&tab=brief&window=24h`.

- Add **Quick brief** to the existing workbench tabs. Make it the default for new mobile visits; preserve explicit tab selections, existing bookmarks, and desktop research workflows.
- Search by configured name, symbol, or chain-qualified contract. Select a result showing name, network, and shortened contract when present. Unknown or ambiguous input opens a choice or an untracked state; it never silently selects the first coin.
- Default to **24 hours**, with **7 days** and **30 days** alternatives. Keep the selected window in the URL. Sparse results offer a wider window instead of widening it silently.
- Show selected identity, latest evidence time, and coverage status next to the briefing. Loading time is not evidence time.
- Lead with a short overview and at most three distinct takeaways. Each has a concise statement, optional **Why it matters**, optional **Watch next**, and a source count that opens its supporting posts.
- A source drawer shows author, publication time, original text, source link, and identity-match reason. Users can reach evidence in one tap.
- Put account rankings, charts, network graphs, raw rule names, and advanced filters below the brief or in existing research tabs.
- Preserve the chosen asset, time window, and search when changing tabs or returning from evidence. Back/forward navigation and shared links restore the same scope.
- On 390 × 844, the first takeaway must be visible without scrolling past controls. At 320px, no page-level horizontal scrolling. Use at least 44px tap targets and 16px search input text; never require hover.
- A screen reader receives the result count and loading/error state. Opening and closing evidence preserves focus. Color cannot be the only indicator of coverage or importance.

Overview example structure, using variables rather than fabricated observations:

> {Asset} — {window}. {Leading supported development}. Based on {distinct posts} saved posts from {distinct authors} accounts. Latest evidence: {time}. {Coverage limitation, if any}.

When no defensible development can be extracted, say **No clear development in the saved posts** and show relevant excerpts. Do not manufacture a narrative from a count.

## 4. Configuration and identity

### Single source of identity

Continue using `crypto-coins.json` for symbol, name, network, address, matching patterns, official-account declarations, archive start, and collection configuration. Do not copy contract addresses into UI components, prompts, or a second registry.

Introduce an optional presentation configuration keyed by the existing registry symbol. Suggested location: `apps/web/lib/crypto-brief-config.ts`. It controls defaults and capability preferences, not whether data actually exists.

```ts
type CryptoBriefConfig = {
  coin: string; // must resolve in COINS
  defaultWindow: "24h" | "7d" | "30d";
  maxTakeaways: 1 | 2 | 3;
  categories: Array<
    "product" | "governance" | "listing" | "security" |
    "market_activity" | "community" | "other"
  >;
  modules: {
    savedPosts: true;
    marketContext: boolean;
    accountContext: boolean;
    namedSignals: boolean;
  };
};

const example: CryptoBriefConfig = {
  coin: "BACKPACK", // identity is read from the existing registry
  defaultWindow: "24h",
  maxTakeaways: 3,
  categories: ["product", "listing", "security", "market_activity", "community", "other"],
  modules: { savedPosts: true, marketContext: true, accountContext: true, namedSignals: true },
};
```

One default configuration applies to every registry entry; overrides are optional. Missing prices or account history disable those outputs at runtime rather than preventing a post-only brief.

### Identity requirements

1. Resolve a stable registry identity before generating a coin brief. For token contracts, identity includes network and address; a ticker alone is not globally unique.
2. Reuse `matchesCoin` and the existing eligibility policy. Preserve case for case-sensitive chain addresses; do not apply EVM lowercasing to every chain.
3. Retain match provenance: `contract`, `cashtag`, `name_context`, or `declared_account`. A declared account's unrelated posts do not become token developments automatically.
4. Keep exact token evidence separate from exchange, wallet, company, native-asset, and ordinary-word mentions. Related entities may be configured later; do not merge them by name.
5. Expose identity uncertainty from the registry. A supplied address or an entry in `official` does not become independently verified because it is displayed.
6. A native asset may legitimately have no contract. Null address is not missing data to invent.
7. New identities receive positive, negative, ambiguous-name, cross-chain, and address-case fixtures shared by Python and TypeScript.

**Backpack acceptance case:** the current `BACKPACK` registry entry describes a Solana token and labels its supplied address as unverified. Bare “backpack” needs token context. A luggage post must fail matching. A Backpack exchange announcement must not be attributed to that token without additional evidence. The user's selected identity must remain visible throughout the brief.

## 5. Modules and dependencies

```text
Shared coin registry + optional brief configuration
                   |
     Identity resolver and eligibility policy
                   |
  Stored-data adapters: crypto archive / RSS X / market / accounts
                   |
  Normalized evidence + explicit coverage + identity provenance
                   |
     Deduplicate -> select window -> group developments
                   |
   Pure briefing builder + optional persisted synthesis
                   |
      Versioned read API -> shared mobile brief components
```

| Module | Responsibility | Must not do |
| --- | --- | --- |
| Identity resolver | Resolve configured asset and match reasons | Guess a contract or silently merge namesakes |
| Source adapters | Read stored rows; normalize IDs, dates, text, URLs, coverage | Start paid collection during a page read |
| Evidence selector | Apply identity policy, publication window, dedupe, and caps | Treat fetched time as publication time |
| Development grouper | Combine duplicate or same-event evidence with traceable members | Treat repeated copies as independent confirmation |
| Brief builder | Produce up to three supported takeaways and limitations | Invent a baseline, consensus, or market explanation |
| Optional synthesis worker | Persist evidence-bound text with version/hash | Generate on every search keystroke or read |
| API layer | Validate parameters; cache and serialize one response contract | Hide unavailable sources behind successful empty arrays |
| Presentation | Render brief and evidence consistently on mobile and desktop | Contain asset-specific matching or scoring logic |

Suggested new files: `lib/crypto-brief-types.ts`, `lib/crypto-brief.ts`, `lib/crypto-brief-config.ts`, `lib/server/crypto-brief-store.ts`, and `components/market/crypto-brief.tsx`. Extract reusable presentation primitives from the shipped X component only as they gain a second caller. Do not create a separate package or framework for each asset.

## 6. Normalized contracts

These are proposed contracts, not descriptions of current endpoint payloads.

```ts
type BriefEvidence = {
  id: string; // canonical x:<snowflake>; never Number(tweetId)
  source: "crypto_archive" | "rss_x";
  sourceRecordIds: string[]; // retain all ingested copies
  authorId: string | null;
  handle: string | null;
  text: string;
  url: string | null;
  publishedAt: string | null;
  observedAt: string | null;
  kind: "original" | "reply" | "quote" | "repost" | "unknown";
  match: {
    coin: string;
    method: "contract" | "cashtag" | "name_context" | "declared_account";
    provisional: boolean;
    reason: string;
  };
};

type BriefCoverage = {
  status: "available" | "partial" | "unavailable";
  windowStart: string;
  windowEnd: string;
  latestPublishedAt: string | null;
  lastCollectedAt: string | null;
  matchingPostCount: number | null; // before evidence cap, when known
  includedPostCount: number;
  includedAuthorCount: number | null;
  evidenceLimit: number;
  truncated: boolean;
  excludedUndatedCount: number;
  collectionComplete: boolean | null; // query exhaustion is not all-of-X coverage
  comparableBaseline: boolean;
  reasons: string[];
};

type BriefTakeaway = {
  id: string;
  category: string;
  text: string;
  basis: "post_excerpt" | "stored_analysis" | "named_rule";
  whyItMatters: string | null;
  watchNext: string | null;
  evidenceIds: string[];
  ruleId: string | null;
  limitations: string[];
};
```

Every displayed factual claim must resolve to evidence IDs or a named metric/rule with timestamped inputs. Stored analysis additionally retains status, model, generation time, evidence hash, and version. A content or identity-policy change invalidates incompatible cached analysis. Stale, failed, or fallback analysis uses source excerpts instead.

Market observations have their own source, interval, observed time, currency, and pool/venue identity. They cannot inherit the post feed's freshness or imply exchange-wide prices from a single pool. Keep unavailable numbers null throughout storage, API, and UI.

## 7. Brief construction

1. Capture one `asOf` instant. Define the requested UTC interval as `[start, end)` and apply it consistently to posts and comparable metric windows.
2. Resolve identity and read matching stored evidence **before** applying the response limit. The existing chronological `/run` limit can retain old posts and omit recent ones; do not download that archive and slice it in the browser for a fresh brief.
3. Exclude future-dated and undated posts from timed takeaways. Count excluded undated rows where practical; expose them in the archive rather than making them appear fresh.
4. Canonicalize X/Twitter hosts and tweet IDs. Preserve string IDs across both stores and all UI keys. Count a post seen through two sources once.
5. Exclude reposts from original-post metrics. Retain replies and quotes as explicitly labeled evidence; embedded quotes and reprinted text are not independent confirmation.
6. Reuse the existing crypto eligibility/contract-anchor rules before grouping. Keep excluded/provisional reasons inspectable.
7. Group exact copies first. Group different posts into a development only when a stored event reference, shared source URL, or conservative versioned rule supports it. Otherwise keep them as separate excerpts.
8. Select at most three distinct developments, prioritizing eligible direct evidence and recency, with stable ID tie-breaks. Prefer one development per author when alternatives exist. Repeated alert feeds must not fill the entire brief. Do not use a follower count as evidence of correctness.
9. Use current persisted analysis when compatible with the selected evidence. Otherwise use an attributed, query-aware excerpt capped near 260 characters. Missing interpretation omits “Why it matters”; it does not generate boilerplate.
10. “Watch next” is a sourced follow-up question or a registered measurable condition. It is not a price target or an unsupported prediction.
11. Preserve material disagreement. Conflicting posts remain attributed and visible; do not collapse disagreement into positive/negative sentiment.

### Comparisons and signals

- Attention change requires matching collection policy, window length, identity version, and usable coverage in both periods. Otherwise show counts for the observed sample and label the comparison unavailable.
- Missing baseline is unknown, never zero. A real zero baseline may be described as “first saved activity in this comparison,” not infinite growth or first activity on X.
- Existing `*_24h` fields in `/signals` can reflect its requested `hours`. The adapter must use the actual interval and cannot reuse labels blindly for 7- or 30-day briefs.
- Existing market comparisons anchored to the latest archived candle need timestamp alignment before being combined with a wall-clock post window. Stale or nonoverlapping market data cannot produce a fresh price/attention comparison.
- Preserve named-rule definitions and inputs. A rule firing can be included as an observation, but severity is not confidence and price adjacency is not causation.
- With partial collection, qualified wording is “in saved posts,” not “X is trending,” “the market believes,” or “everyone is buying.”

## 8. Read API, caching, and failures

Proposed endpoint:

`GET /api/market/crypto/brief?coin=BACKPACK&window=24h`

Parameters are a registry symbol and one of `24h`, `7d`, `30d`. Validate before querying; unknown coins/windows return a structured 400. Do not silently fall back to ZCAT or the first registry entry.

The response uses the existing `{ ok, data }` envelope. `data` contains `schemaVersion`, resolved asset identity, selected window, `asOf`, `coverage`, `overview`, `takeaways`, bounded `evidence`, and independently timestamped optional market/account modules. Proposed evidence cap: 100 unique posts per response. Load additional evidence through a cursor ordered by `(publishedAt, canonicalId)` and pinned to the same `asOf`.

Use `matchingPostCount` from the same predicate before the cap when affordable; otherwise return null and label the count as included posts. Never present 100 loaded posts as the full count when `truncated` is true. Totals, author counts, and takeaway source counts must declare whether they describe all matched rows or only the included sample.

| State | API/UI behavior |
| --- | --- |
| No database or archive not started | Explicit unavailable/not-started state; not “no activity” |
| Valid read, no eligible posts | Empty brief scoped to the selected window; offer a wider window |
| Partial collection or bounded sample | Render supported takeaways with visible coverage limitation |
| No price data | Render post brief; omit price interpretation |
| No usable analysis | Render labeled excerpts with original source links |
| Read failure | Structured error and retry; retain the last successful brief with a stale label |
| Unknown identity | Explain that the asset is untracked; no automatic ingestion |

Initial crypto brief cache: reuse the existing shared cache pattern with a maximum 120-second fresh TTL. Include asset, interval, schema, matching-policy version, and evidence revision in any persisted snapshot key. Return actual computation and source times so cached data is not relabeled as newly collected. The manual refresh action revalidates the saved-data response, subject to existing rate limits; it never calls a paid collector. Do not cache an error as a valid empty brief.

The shipped generic X feed remains `no-store`; this proposal does not silently alter its contract.

## 9. Collection, storage, and cost

- Reuse `crypto_social_*`, existing market archives, current collector ledgers, and health reporting. Brief reads add no external-provider calls, no recurring jobs, and no LLM calls.
- A new configured asset can show “not collected yet.” Merely opening it does not activate collection or increase a budget.
- Adding or changing a collection campaign is a separate operation under the existing bounded collection workflow. Do not mutate a live campaign's saved `searchQuery`, reset its ledger, or extend its lifetime to implement this UI.
- Avoid a second post store. If a persisted brief is later needed, store a derived snapshot with its source IDs, evidence hash, identity/config versions, and generation time.
- Optional cross-post synthesis is a later capability. It requires an explicit per-run and daily budget, idempotency key, output validation, persisted citations, and deterministic excerpt fallback. Keep provider/model selection behind an adapter.
- Instrument response duration, cache hits, row counts, excluded/undated counts, truncation, analysis fallback, stale source age, and failed reads. Use the existing source-health pattern for collection failures.

## 10. Add another crypto: repeatable procedure

1. Resolve the intended asset: name, symbol, network, optional contract, and namesakes. Record uncertainty; do not guess identities from a cashtag.
2. Reuse or add one entry in the shared registry. Add identity fixtures in `tests/fixtures/crypto-coin-matches.json` and run both language matchers.
3. Use the default brief configuration, adding an override only for a real product difference. No new page, route, or asset-specific component.
4. Confirm which stored sources and intervals are available. A post-only asset and a native asset without a contract must work.
5. Run the contract tests against that asset, including negative examples and partial/unavailable data states.
6. Review the mobile brief and its source links. If new collection is needed, plan and activate it separately under the existing budget controls.

Reusable implementation ticket:

```text
Asset / registry key:
Identity and provenance:
Namesakes / exclusions:
Existing stored sources and coverage:
Default configuration or required override:
Enabled optional modules:
Positive / negative / ambiguous matching fixtures:
Missing-data behavior:
Mobile evidence reviewed:
Collection changes needed (separate task, if any):
Validation results:
```

## 11. Delivery order and acceptance

| Phase | Deliverable | Exit criteria |
| --- | --- | --- |
| A: normalized evidence | Types, crypto adapter, identity provenance, shared deduplication | String tweet IDs survive; mixed-store copies dedupe; Backpack namesake cases pass; existing RSS X tests still pass |
| B: deterministic brief API | Bounded read, coverage, grouping, excerpt/analysis fallback | Recent matches are selected before LIMIT; every takeaway resolves to evidence; no provider calls on GET |
| C: mobile workbench integration | Quick brief tab and evidence drawer; URL state | 320/390/768/1440px checks pass; first takeaway visible on phone; Back restores scope; old research links still work |
| D: reusable onboarding | Default config plus second and third registry assets | Backpack plus an existing token on another network and a native asset work without new routes/components |
| E: optional synthesis | Persisted evidence-bound overview and contradictions | Budget/idempotency tests pass; stale or failed output falls back; unsupported claims are rejected |

Required regression cases:

- Same tweet under X/Twitter/mobile URLs, two feeds, and large snowflake IDs above JavaScript's safe integer range.
- Missing, malformed, future, and exact-window-boundary dates; daylight-saving transitions do not change UTC counts.
- Stale/failed/fallback analysis; a late-in-post search match; contradictory source claims.
- Backpack luggage versus configured token versus exchange/company; same ticker on different chains; native asset without contract.
- Empty, unavailable, truncated, stale, and partially collected periods; no baseline and a real zero baseline.
- A signal post mentioning multiple assets does not assign a contract event to the wrong asset.
- A heavily repeated alert does not crowd out all other developments; missing author identity is not invented.
- Stale market candles, differing metric windows, and missing optional modules cannot create unsupported comparisons.
- Failed refresh preserves the previous brief; changing asset cancels obsolete responses; keyboard and screen-reader evidence access work.
- Page reads do not change collection ledgers, call external collectors, or trigger model inference.

Validation commands should include the existing coin/workbench/signal suites, the shipped `x-signal-insights.test.ts`, new pure briefing tests, targeted API tests, mobile browser checks, `npm run typecheck`, and `npm run build`. Run Python registry fixtures whenever identity configuration or matching changes.

Completion means a second eligible crypto can be added through registry/configuration and fixtures alone, and a mobile user can trace each displayed takeaway back to the correct asset's dated evidence.
