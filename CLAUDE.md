# CLAUDE.md

## Process Pipeline Review (Enrichment/Analysis/Topics/Trends) - July 2026

Use this note when debugging missing corpus content, wrong trend counts, stale sentiment, or bad topic/keyword matches. Findings below are from a review of the *processing intelligence* (enrichment, sentiment, RSS analysis, topic matching, trend aggregation) as opposed to the ingestion plumbing (see the Ingestion Pipeline Review section below). Status: **all critical and high-value findings below have been implemented and are covered by tests** (`tests/test_workflow_connectors_valid.py`, `test_trend_aggregation.py`, `test_connector_topic_keywords.py`, `test_enforcement_inference.py`, `test_evidence_verification.py`, `test_sentiment_pipeline.py`). The "good to have" items remain open. Re-verify against current code/git history before assuming anything below is still true, since this note isn't auto-updated.

### Critical findings

- **A bad merge on 2026-07-07 silently deleted 16 connectors that 3 workflows still schedule.** Merge commit `43cf3b6` resolved `run_connector_extraction_pipeline.py` and `trade_media_scraper.py` to the side without the cyber/crypto-media connectors (~530 lines removed from each, plus their tests), but the workflows that reference them were kept. Confirmed via `git show 43cf3b6^2:run_connector_extraction_pipeline.py` (has all 16) vs `HEAD` (has none). Affected and failing every run since:
  - `rss-full-ingestion-3hour.yml` (all 7 legs): `prnewswire_article`, `google_news_ponzi_investor_fraud_article`, `google_news_senate_committee_article`, `coindesk_article`, `cointelegraph_article`, `decrypt_article`, `the_block_article`
  - `cyber-sources-3hour.yml` + `connector-gap-6hour.yml` (9 legs): `krebs_on_security_article`, `the_hacker_news_article`, `welivesecurity_article`, `sophos_security_operations_article`, `flashpoint_blog_article`, `recorded_future_article`, `intel471_blog_article`, `securityweek_article`, `dark_reading_article`
  - Each run fails fast on an argparse "invalid choice" error before any extraction happens, and before `record_source_health` runs — so the health monitor cannot see this failure class (see finding below). This is the likely real cause behind zero-output sources flagged in `tmp_artifacts/source_zero_audit/`.
  - **DONE (commit `52292f9`)** — All 16 configs restored into `TRADE_MEDIA_SOURCES` and `TRADE_MEDIA_CONNECTORS`; the current scraper kept the discovery machinery, so this was purely additive (the deleted tests only covered the removed `clean_article_text`, which the current code intentionally replaced, so they were not restored). Verified: all 16 in `SUPPORTED_CONNECTORS`, all connectors in the 3 workflows resolve, and `krebs_on_security_article` runs end-to-end at the argparse level. Note: actual *production* run success (network + GCS creds) was not independently verified — only the argparse "invalid choice" failure mode that broke every run.
- **Reviewed documents silently disappear from trend aggregation.** `trend_aggregation.py`'s `build_trends` only counts enrichment entries with `status == "enriched"`. But accepting a document's enrichment in the review UI sets `status = "reviewed"`, so curating a document removes it from every trend count, sparkline, and growth calculation.
  - **DONE (commit `52292f9`)** — `TREND_COUNTED_STATUSES = {"enriched", "reviewed"}` now gates counting; `"fallback_enriched"` deliberately stays excluded (heuristic tags would dilute the signal). Covered by `tests/test_trend_aggregation.py`.

### High-value improvements

- **The sentiment/tone pipeline (`run_sentiment_pipeline.py`) is orphaned.** No GitHub Actions workflow or web route references it — confirmed via repo-wide search. The UI (`policy-research-hub.tsx`) renders tone chips that never update for new documents unless someone runs the script by hand. Compounding issues once scheduled: `only_missing` mode skips docs that already have *any* `sentiment` dict, including `fallback_scored` (heuristic/neutral) results — so one OpenAI quota outage permanently mislabels every doc scored during it as neutral (contrast with enrichment/RSS-analysis, which both retry weak/fallback results). It's also OpenAI-only while the rest of the stack has moved to DeepSeek, and it has its own private, unguarded copy of the GCS load/save code (no read-failure guard, no generation-match) — a third unprotected writer to `document_enrichment_state.json`.
  - **DONE (commit `9e1e244`)** — `only_missing` now retries any status not in `{"scored","reviewed"}` (i.e. fallback/failed/missing get re-scored); added `--provider` with DeepSeek default + model fallback; rewrote the script as a thin wrapper over `run_financial_news_pipeline` (`core._load_custom_documents`/`_load_enrichment_state`/`_save_enrichment_state`), inheriting the read-guard and generation-match and deleting ~120 lines of duplicate storage code; added `.github/workflows/sentiment-scoring-daily.yml` (cron `30 21 * * *`, matrix over newsapi/bloomberg/wsj/substack, serialized). Covered by `tests/test_sentiment_pipeline.py`.
- **No CI check validates that scheduled workflows reference real connectors.** This is exactly how finding #1 went unnoticed for days.
  - **DONE (commit `52292f9`)** — `tests/test_workflow_connectors_valid.py` parses all workflow YAMLs and asserts every `connector`/`--connector` is in `SUPPORTED_CONNECTORS` and every `source_kind`/`--source-kind` is a connector or a known non-connector kind (`newsapi_article`); `python-tests.yml` now triggers on `.github/workflows/**`. Negative-tested that dropping a connector makes it fail.
- **Evidence spans are never verified against the source document.** Enrichment asks the model for verbatim quotes (`evidence_spans`) and `_compute_reward` scores by *count*, but nothing checks the snippets actually appear in the text — hallucinated quotes score identically to real ones and skew review-priority ranking.
  - **DONE (commit `6971922`)** — `_normalize_enrichment_payload` now tags each span `verified: true/false` via `_evidence_snippet_verified` (whitespace/quote-normalized substring check), and `_compute_reward` uses `_evidence_quality_score` = coverage × verified-fraction. app.py's duplicate copies updated in lockstep; spans without a `verified` key are grandfathered as verified. Covered by `tests/test_evidence_verification.py`.
- **Topic keyword matching has two different, inconsistent implementations.** TS (`apps/web/lib/intel-topic-matching.ts`) uses word-boundary regexes; Python (`_annotate_topic_matches`) used plain substring matching, so short keywords like `"ai"`/`"sec"` over-matched.
  - **DONE (commit `6971922`)** — `_topic_term_pattern`/`_topic_term_matches` port the TS `keywordPattern` word-boundary semantics (flexible separators for multi-word terms). Covered by `tests/test_connector_topic_keywords.py`. STILL OPEN (not done): auditing the overly-generic `DEFAULT_TOPIC_RULES` keywords (`"market"`, `"credit"`, `"economy"`, `"stock"`) that gate keyword-filtered RSS ingestion. NOTE: `_match_filter_terms` (DOJ exclude terms) is still substring by design — that's intentional, not the same bug.
- **Enforcement-metadata heuristics substring-match on generic words.** `_infer_enforcement_metadata` matched "order"/"charges" as bare substrings, so "in order to..." / "supercharges" got misclassified — on every heuristic-enrichment fallback, not just enforcement docs.
  - **DONE (commit `6971922`)** — added a `_contains_word` boundary matcher and an `is_enforcement_context` gate (`_ENFORCEMENT_SOURCE_KINDS` or `_ENFORCEMENT_CONTEXT_TERMS`); action_type/forum/outcome/violations/parties only classify inside enforcement context, and bare "order" was dropped as a signal. Covered by `tests/test_enforcement_inference.py`.
- **Padded RSS analysis defeats its own retry logic.** `strengthenFeedAnalysis` (`feed-analysis.ts`) padded lists with boilerplate to hit minimum counts, but the retry checks judged weakness by counting items, so padded analyses looked healthy and never regenerated.
  - **DONE (commit `0f4bd04`)** — `FeedAnalysis` now carries `strengthened` (set only when a *non-sparse* input was padded, so genuinely sparse items don't thrash); persisted on `rss_article_analysis` (new column + idempotent ALTER) and re-queued by both `shouldRefreshFeedAnalysisForDeepSeek` and the `getRssArticlesNeedingAnalysis` SQL. Verified by `tsc`/`eslint`/`next build`.

### Good to have, not urgent — ALL STILL OPEN (none of these were done)

- Unify the three separate tone/sentiment systems (RSS keyword tone in `neon.ts`'s `inferToneLabel`, LLM sentiment in `run_sentiment_pipeline.py`, heuristic fallback) — they currently disagree, and feed analysis consumes `tone_label` as prompt input so keyword misfires propagate downstream.
- Entity normalization / alias map for recurring regulators ("SEC" / "Securities and Exchange Commission" / "the Commission") — currently fragments `intelligence_mentions` and weakens the knowledge graph.
- Trends: add an "uncategorized/emerging" bucket instead of silently dropping tags that match no fixed taxonomy category (`_map_to_taxonomy`); `min_mentions` qualification is all-time rather than windowed; `baseline == 0 → +100%` growth is an arbitrary convention.
- Three near-identical stopword lists (`_heuristic_enrichment` in Python, `heuristicKeywords` in TS, trend aggregation) will drift — consider one shared config.
- Long-document truncation (14k chars for RSS analysis, 90k for enrichment, both cut-the-middle) loses mid-document content on long testimony/transcripts; consider smarter selection if that content matters.

### Implementation record

All critical + high-value items above were implemented across commits
`52292f9` (connectors, CI guard, trends), `6971922` (topic matching,
enforcement inference, evidence verification), `0f4bd04` (padded-analysis
retry), and `9e1e244` (sentiment pipeline), then pushed to `main`. Full
Python suite (186 tests) plus web `tsc`/`eslint`/`next build` all pass. The
"good to have" items remain open.

## Ingestion Pipeline Review - July 2026

Use this note when debugging missing/duplicated documents, corpus data loss, or scheduled ingestion job conflicts. Findings below are from a full review of the connector extraction, enrichment, and RSS ingestion pipeline. Status: all findings except the Neon migration and the admin-route half of the concurrency fix have been implemented and are covered by tests (`tests/test_url_match_key.py`, `tests/test_json_store_concurrency.py`, updates to `tests/test_financial_news_enrichment_checkpoint.py` and `tests/test_run_connector_extraction_pipeline.py`) — re-verify against current code/git history before assuming a finding is still open, since this note isn't auto-updated.

### Critical findings

- **A transient GCS read failure can silently roll back or wipe the corpus.** Both `_load_json_store` (`run_financial_news_pipeline.py`) and `POST /api/admin/documents` (`apps/web/app/api/admin/documents/route.ts`) treated a failed remote read the same as "no data exists," then saved that empty/stale payload back over the real corpus in `custom_documents.json`. `--require-remote-persistence` only guarded writes, not reads. **Fixed**: `_load_json_store`/`_save_json_store` now track per-blob whether the remote read errored (`_REMOTE_LOAD_ERRORED_BLOBS`) and refuse to save under `require_remote_persistence` if so; `app.py`'s independent duplicate of this load/save logic got the same guard (`_CUSTOM_DOCS_REMOTE_LOAD_ERRORED`). `gcs-loader.ts` gained `downloadGcsJsonSafe`, which distinguishes not-found from error; both admin document routes (`POST`/`DELETE`) now return 503 on a read error instead of proceeding.
- **All YouTube videos collapse to one dedup key.** `_url_match_key` (`run_financial_news_pipeline.py`, and a duplicate copy in `app.py`) stripped query strings, so every `youtube.com/watch?v=<id>` URL normalized to the same key — each new video ingested overwrote the previous one in `custom_documents.json`. Also collapsed `http://` vs `https://` variants of any URL. **Fixed**: scheme is normalized to `https`, the query string is preserved (sorted, for stable ordering) except for known tracking params (`utm_*`, `fbclid`, `gclid` — stripped so RSS re-fetches with different tracking tags still dedupe, matching the convention already used in `trade_media_scraper.py`). Both `app.py` and `run_financial_news_pipeline.py` were updated identically since they must compute the same doc-identity keys. Note: the local committed `data/custom_documents.json` had zero YouTube records at fix time (predates the connector), so no backfill was needed there — the real corpus lives in GCS and wasn't independently audited.
- **Lost-update races between concurrent workflows writing the same JSON blob.** ~15 scheduled GitHub Actions workflows plus Vercel admin routes all do unguarded read-modify-write of `custom_documents.json` / `document_enrichment_state.json` / `rule_summaries.json` with only per-workflow concurrency groups. **Partially fixed**: the Python side (`_load_json_store`/`_save_json_store`) now tracks the GCS object `generation` observed at load (`_BLOB_GENERATIONS`) and passes it as an `if_generation_match` precondition on save, raising a clear `RuntimeError` on conflict instead of silently overwriting a concurrent writer's changes (chained correctly across enrichment's checkpointed saves). This is fail-loud, not retry-and-reapply — a conflict fails the run rather than auto-merging, which is judged safer than reapplying arbitrary in-flight mutations. The Vercel admin routes (`apps/web/app/api/admin/documents/*`) still lack this guard — see spawned follow-up task for extending `downloadGcsJsonSafe`/`uploadGcsJson` with the same generation-match pattern (couldn't be verified against a live GCS bucket in the original session). Longer-term, migrate documents to normalized Neon storage instead of one JSON blob.

### High/medium findings

- Single-JSON-blob storage for the whole corpus doesn't scale (O(N) linear scans, full re-upload per ~25 new docs) and is the root cause of the race above — plan a Neon-backed schema migration (not yet started).
- **Fixed**: `/api/intel/rss-refresh/route.ts` and `/api/intel/x-refresh/route.ts` were both fail-open (skipped auth entirely) if `CRON_SECRET`/`RSS_REENRICH_SECRET` were both unset. Both now use a shared `checkCronAuth` helper (`apps/web/lib/server/api-utils.ts`) that fails closed (503) like the admin middleware does.
- **Fixed**: `neon_feeds.py add_feed()` inserted with `ON CONFLICT (feed_url)` instead of `feed_key`, reintroducing the exact `rss_feeds_feed_key_key` duplicate-key bug already fixed in `apps/web/lib/server/neon.ts` (see smoke-fix section below). Now conflicts on `feed_key` and updates `feed_url`.
- **Fixed**: `_should_fail_for_item_failures` (`run_connector_extraction_pipeline.py`) failed the whole workflow run on any single item-level extraction failure for most connectors, even at 29/30 success. Now uses a failure-rate threshold (fails only if `processed_count == 0` or failure rate exceeds 50%).
- **Fixed**: `markFeedRefreshed` ran in a `finally`/masked-outcome path in `rss-refresh/route.ts`, so persistently-failing feeds still looked "recently refreshed." `rss_feeds` gained `last_error`/`consecutive_failures` columns; `markFeedRefreshed(feedKey, error?)` now records failures instead of resetting them, and the admin UI (`apps/web/app/admin/page.tsx`) surfaces them per feed. Note: `apps/web/lib/server/x-timeline-ingestion.ts` and `apps/web/app/api/intel/feed/route.ts` call `markFeedRefreshed` with only `feedKey` (no error tracking) — not touched, since the finding was scoped to the RSS refresh path.
- **Fixed**: metadata-fallback stub records (`_build_short_text_fallback`) flowed into DeepSeek enrichment unfiltered. `_build_news_enrichment_candidates` now skips docs whose `extraction_mode == "metadata_fallback"` or whose text contains the shared `METADATA_FALLBACK_TEXT_MARKER` (detection by text marker matters because `extraction_mode` isn't reliably set to `"metadata_fallback"` for every connector that can produce one of these stubs via the pipeline-level fallback helper, as opposed to scraper-level fallbacks).

## Production Endpoint Smoke Fix - July 2026

Use this note when debugging app endpoint smoke failures or future regressions in the production dashboard APIs.

### What broke

- The production smoke suite was failing on `/api/metrics` and `/api/search` with HTTP 500s.
- `/api/search` depended entirely on OpenAI vector/file-search. When the OpenAI account was over quota or the provider was slow, the route returned `SEARCH_FAILED` instead of keeping the app usable.
- `/api/metrics` loaded large GCS-backed corpus and settings payloads without a bounded response budget. Slow corpus reads could make the endpoint time out or fail even though partial metrics were still useful.
- While verifying the production fix, `/api/intel/feed` and `/api/intel/recap` exposed a separate Neon schema issue: `ensureSchema()` seeded `rss_feeds` with `ON CONFLICT (feed_url)`, but production already had rows that collided on the table's unique `feed_key`. That duplicate-key exception happened before feed or recap responses could be returned.

### How it was fixed

- `apps/web/app/api/search/route.ts` now wraps semantic search in a route-level time budget and returns a corpus keyword fallback with `mode: "keyword_fallback"` plus a warning when vector search is unavailable, over quota, or slow.
- `apps/web/app/api/metrics/route.ts` now loads corpus, custom documents, enrichment state, and connector settings through bounded helpers. If one source is slow or unavailable, the endpoint returns partial metrics with a `warnings` array instead of failing the whole request.
- `apps/web/app/api/intel/recap/route.ts` now returns an empty recap payload with a warning when the recap store is unavailable. Invalid date input still returns 400.
- `apps/web/lib/server/neon.ts` now seeds default RSS feeds with `ON CONFLICT (feed_key)` and updates `feed_url`, preventing duplicate `rss_feeds_feed_key_key` errors during `ensureSchema()`.

### Verification

- Local validation: `npm run typecheck`, `npm run lint`, `npm run build`, and `APP_SMOKE_BASE_URL=http://127.0.0.1:<port> npm run test:app-endpoints`.
- Production validation after merge: `npm run test:app-endpoints` against `https://speeches-zeta.vercel.app` passed `7/7`.

## Stock-Specific News Connectors

Use this guidance when adding connectors that support stock/ticker-specific news, market commentary, or "stocks getting attention" workflows.

### Feasibility By Source

- RSS, blogs, newsletters: high feasibility. Prefer RSS/Atom feeds and normal article extraction. This matches the existing `trade_media_scraper.py`, `wsj_rss_scraper.py`, and `substack_public_scraper.py` patterns.
- Public Substack posts: high feasibility for public posts. Prefer publication `/feed` URLs and public post extraction. Do not attempt to bypass paid/private posts; mark them as access-limited and preserve metadata/preview only when available.
- Reddit: medium feasibility. Prefer OAuth/PRAW for reliable access. The existing `reddit_scraper.py` supports this shape. Treat commercial use as requiring Reddit API compliance review/permission, and expect unauthenticated JSON access to fail from cloud IPs.
- X/Twitter: medium feasibility but potentially costly. Use official X API v2 for recent search, user timelines, and streams. Do not scrape X pages. Add strict budget caps, source allowlists, and rate-limit handling before enabling production jobs.
- YouTube: medium feasibility. Use the YouTube Data API for channel/video discovery, metadata, comments, and stats. Treat transcripts/captions as conditional on availability and authorization.
- Telegram: medium/low feasibility. Bot API works for channels where the bot is authorized and receives channel posts. Do not assume arbitrary public-channel history can be ingested reliably.
- TradingView: low feasibility for ingestion. TradingView widgets are display tools, not a market-data export API. For prices, candles, fundamentals, or indicators, use licensed market-data providers instead.

### Preferred Architecture

Build stock-news ingestion as connector modules feeding a normalized schema, not as a single scraper.

Core records should include:

- `sources`: platform, handle/url, source owner, access mode, priority, compliance status.
- `raw_items`: platform id, source id, author, published timestamp, canonical URL, raw text/metadata, engagement metrics, fetched timestamp.
- `mentions`: raw item id, ticker/entity, confidence, context snippet.
- `daily_stock_attention`: ticker, mention count, weighted score, mood, source count, top source ids.
- `analyst_profiles`: curated identity, handles, channels, notes, credibility metadata.
- `daily_briefs`: generated summaries with source citations.

Pipeline order:

1. Ingest new posts/articles from each connector.
2. Normalize records into one internal schema.
3. Deduplicate by platform id, canonical URL, and content hash.
4. Extract tickers, cashtags, companies, sectors, and themes.
5. Classify mood as bullish, bearish, mixed, neutral, or informational.
6. Aggregate "stocks getting attention" using mention count, source diversity, source weight, freshness decay, and engagement.
7. Generate source-backed daily briefs with links/citations to originals.
8. Label output as research context only, not investment advice.

### Implementation Order

Start with sources that are legally and operationally stable:

1. Expand RSS/blog/Substack connectors.
2. Add Reddit OAuth/PRAW configuration and commercial-use checks.
3. Add X API connector with explicit budget controls and curated handles.
4. Add ticker/entity extraction and mood aggregation.
5. Add daily brief generation and source-backed UI surfaces.

Avoid starting with Telegram history or TradingView data ingestion. Those should only be added after the reliable source pipeline is working and the access model is clear.

### Compliance Defaults

- Prefer official APIs and public RSS feeds over page scraping.
- Preserve canonical URLs and source attribution for every item.
- Do not bypass paywalls, login walls, CAPTCHAs, private groups, or anti-bot controls.
- Store enough raw metadata to audit why a ticker appeared in a summary.
- Separate source-backed facts from model-generated summaries.
- Add per-connector rate limits, retries, and failure reporting before production scheduling.

## Deferred X Account Monitoring

The user is interested in tracking a small curated list of X accounts, but this work is paused for now. Do not enable scheduled X ingestion, purchase credits, add proxy-based scraping, or expand X monitoring unless the user explicitly asks to resume it.

Current budget estimate for official X API v2 pay-per-use:

- X charges roughly `$0.005` per post read and `$0.010` per user read.
- Tracking 25 accounts once per day costs about `$3.75/month` for 1 post per account per day, `$18.75/month` for 5 posts each, `$37.50/month` for 10 posts each, and `$75/month` for 20 posts each.
- User lookups add about `$7.50/month` if done daily, but should be reduced to a one-time roughly `$0.25` cost by caching X user IDs after the first lookup.
- DeepSeek enrichment on this volume should be minor compared with X API reads when using `deepseek-v4-flash`.

If resumed, prefer this low-budget architecture:

1. Use official X API v2 with `X_BEARER_TOKEN` as the primary provider.
2. Cache X user IDs for configured handles and avoid repeated user lookup charges.
3. Poll no more than once daily initially, with a hard per-run post limit.
4. Store posts in the existing feed/article pipeline with source chips like `X: @SECGov`.
5. Run DeepSeek enrichment only on stored posts that pass topic/source gates.
6. Treat unauthenticated syndication endpoints as opportunistic fallback only; do not rely on them for production freshness.
