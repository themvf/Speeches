# CLAUDE.md

## Enrichment/Analysis/DeepSeek Review - July 2026

Use this note when debugging DeepSeek spend, stale/missing enrichment for a source kind, or why an article keeps getting re-analyzed. A review of the enrichment (`run_financial_news_pipeline.py`), RSS analysis (`apps/web/lib/server/feed-analysis.ts`), and sentiment (`run_sentiment_pipeline.py`) pipelines found the items below. Status: **all 7 items implemented and tested** (Python: 217 tests passing including 10 new; TS: `tsc`/`eslint`/`next build` all pass, plus a pglite smoke test for the new SQL).

1. **`deepseek-chat` is deprecated 2026-07-24 (verified live via WebFetch against DeepSeek's pricing docs)** — it was still a fallback model in `_candidate_deepseek_models()` (`run_financial_news_pipeline.py`) and `CHAT_MODEL_FALLBACKS` (`trend_aggregation.py`). **Fixed**: removed it from both; DeepSeek's docs confirm it maps to `deepseek-v4-flash`'s non-thinking mode, which is already the primary/first candidate in both lists, so nothing is lost. Trend aggregation's fallback (behind v4-flash) is now `deepseek-v4-pro` instead of the deprecated model.
2. **`connector-enrichment-6hour.yml` — the workflow enriching 49 source kinds including `sec_speech`, `sec_enforcement_litigation`, `finra_awc`, `finra_regulatory_notice`, `pcaob_update`, `msrb_press_release`, and the four banking regulators added this session — had no schedule trigger at all**, only `workflow_dispatch`. Confirmed no other scheduled workflow enriches these kinds (their extraction workflows have no enrich step). This meant primary regulator content only got DeepSeek enrichment when someone manually dispatched it. **Fixed**: added `schedule: cron: "30 */6 * * *"` (offset from other workflows' `:00`/`:15`/`:20` crons to avoid pileup).
3. **Sentiment scoring duplicated a full-text DeepSeek call that enrichment could answer in the same request.** `run_sentiment_pipeline.py` sent the full document text to DeepSeek a second time, after enrichment already sent it once, for every `newsapi_article`/`bloomberg_public_article`/`wsj_dow_jones`/`substack_public_article` document — roughly doubling spend on those four kinds. **Fixed**: added a `sentiment: {score, label, rationale}` key to the enrichment prompt/schema in `_run_enrichment_agent`/`_normalize_enrichment_payload`/`_heuristic_enrichment` (`run_financial_news_pipeline.py`), validated by a shared `_normalize_sentiment_fields()` helper now used by both enrichment and `run_sentiment_pipeline.py`'s own `_normalize_sentiment` (previously two separate, driftable copies of the same validation logic). `_run_news_enrichment` mirrors the enrichment result's sentiment into `entries[doc_id]["sentiment"]` with the exact shape `run_sentiment_pipeline.py` already produces (`score, label, rationale, model, provider, status, error, updated_at`) and a `status` of `"scored"`/`"fallback_scored"` matching its own convention. Because `run_sentiment_pipeline.py`'s `_needs_scoring()` already treats `status in {"scored","reviewed"}` as done, this required **zero changes to the sentiment script's selection logic** — newly-enriched docs are simply already "done" by the time the sentiment cron runs, and it naturally falls back to a real LLM call only for docs enrichment couldn't reach or which predate this change. **Deliberately not touched**: `app.py`'s separate, simpler, OpenAI-`responses`-only `_run_enrichment_agent`/`_normalize_enrichment_payload` (used only for interactive single-document manual re-enrichment from the Streamlit admin UI, not the cron-scale cost driver) — adding sentiment there wasn't worth the risk/effort for a low-volume manual path. Sentiment for docs enriched that way still falls back to the standalone pipeline as before, so nothing regresses.
4. **Every enrichment call used `deepseek-v4-pro` ($0.435/M in, $0.87/M out — verified live) even for lower-stakes commodity content**, where `deepseek-v4-flash` (~3x cheaper) is a reasonable quality tradeoff. **Fixed**: changed the `ENRICH_MODEL` default to `deepseek-v4-flash` in `cyber-sources-3hour.yml` (Krebs, Hacker News, etc.) and `rss-full-ingestion-3hour.yml` (PR Newswire, Google News aggregator queries, CoinDesk/Cointelegraph/Decrypt/The Block); and the hardcoded DeepSeek model fallback in `apps/web/lib/server/feed-analysis.ts` (RSS feed analysis is title+description only, a simpler task than full-document enrichment). Left at v4-pro: primary regulator content (`connector-enrichment-6hour.yml`, agency/senate-committee sites), curated financial news (Bloomberg, NewsAPI, Substack), rule comments, and YouTube — where tag/summary quality feeds the review UI directly.
5. **No token usage telemetry existed anywhere**, so there was no way to attribute spend per source kind or notice a cost regression. **Fixed**: `_create_enrichment_completion` now returns `(text, usage)` via a new `_extract_usage()` normalizer (handles both the chat-completions `prompt_tokens`/`completion_tokens` shape and the responses-API `input_tokens`/`output_tokens` shape), threaded through `_run_enrichment_agent` and `run_sentiment_pipeline.py`'s `_score_with_model`; both run summaries now include aggregate `tokens: {prompt_tokens, completion_tokens, total_tokens}`. On the TS side, `generateFeedAnalysis` (`feed-analysis.ts`) now `console.info`s per-call token usage for both the DeepSeek and OpenAI branches, matching the existing logging convention already used in `recap/generate/route.ts` — there's no per-run summary object to aggregate into here (each call is one request/response cycle inside a web request), so structured console logging (picked up by Vercel's log pipeline) is the right mechanism rather than a port of the Python approach. Known gap: if all attempts in `_run_enrichment_agent`/`_score_with_model` fail outright (raising `RuntimeError`), the tokens spent on those failed attempts aren't captured in the run summary — accepted as a minor telemetry gap, not a correctness issue.
6. **Two unbounded retry loops could re-spend indefinitely on the same doc/article with no forward progress:**
   - Python: `only_missing_or_failed` re-selected any `fallback_enriched`/`failed` doc on every single workflow leg run, forever, with no cap. **Fixed**: added an `attempt_count` field to the stored enrichment entry (incremented every attempt, reset to 0 on a clean success so a later reprocess starts fresh); `only_missing_or_failed` selection now also excludes docs at or above `MAX_ENRICHMENT_ATTEMPTS = 3` (still visible/fixable via the admin review UI, just no longer auto-retried).
   - TS: `shouldRefreshFeedAnalysisForDeepSeek`'s `strengthened` condition (an analysis the model under-delivered on, padded with boilerplate) had no attempt counter, so a chronically-underperforming article could be re-queued and re-billed forever. **Fixed**: added a `strengthen_attempts` column to `rss_article_analysis` (incremented on save when `strengthened` is true, reset to 0 otherwise), exported `MAX_STRENGTHEN_ATTEMPTS = 3` from `neon.ts` so both `getRssArticlesNeedingAnalysis`'s SQL (the actual automatic-reselection gate) and `shouldRefreshFeedAnalysisForDeepSeek` (used by the two manual/on-demand analyze paths) cap that specific retry at the same value without drifting apart. The other refresh conditions (wrong model, weak fields, fallback) were left uncapped since they resolve automatically once DeepSeek is reachable/correctly configured, unlike the strengthened case which was specifically identified as looping with no forward-progress signal.
7. **No timeout guarded the TS DeepSeek/OpenAI fetch calls** in `generateFeedAnalysis` (`feed-analysis.ts`) — a hung provider call could ride out the whole route's request budget (55-60s). **Fixed**: added a `fetchWithTimeout()` helper wrapping both fetch calls in an `AbortController` with a 25-second timeout. A timeout simply makes `fetch()` throw, which is left uncaught here exactly as an ordinary network failure already was before this change — callers (e.g. `rss-analysis-runner.ts`) already catch that and record it as a visible `"failed"` status via `saveRssArticleAnalysisFailure`, rather than a silently-successful-looking heuristic fallback. This preserves existing error-visibility behavior; it only bounds the wait.

**Incidental fix found while editing `neon.ts` for item 6**: a single stray literal NUL byte (`\x00`) had somehow ended up embedded in a template-literal string inside `prepareMentionBatch` (from the Neon cost-fix session), where a plain space was intended. It didn't break `tsc`/`eslint`/`next build` (all tolerated it silently), but it made the file appear binary to some tooling. Replaced with the intended space; verified via byte-level diff that nothing else in the file was affected.

**Verification**: Python — full `pytest` suite passes, including new/updated tests for sentiment mirroring, attempt-count capping, and the model-list changes. TS — `tsc --noEmit`, `eslint --max-warnings 0`, `next build` all pass; the new `strengthen_attempts` SQL was smoke-tested the same way as the Neon cost-fix session above (pglite, since there's still no live `DATABASE_URL` in this sandboxed session).

## Database Cost Audit (Neon/GCS) - July 2026

Use this note when debugging Neon compute-time spend, query latency on `rss_articles`/`intelligence_mentions`, or unbounded storage growth. An audit of Neon and GCS costs found several architectural drivers, all in `apps/web/lib/server/neon.ts`. Status: **all four proposed Neon fixes below are implemented.** The GCS-side findings (bucket versioning/lifecycle status, and whether the ~23 scheduled workflows' full-blob rewrites should be reduced) were **not implemented** — they require direct GCS console verification and/or the already-scoped Neon `documents` migration (see Ingestion Pipeline Review below), not a quick code change.

### Neon findings and fixes

- **`ensureSchema()` ran unmemoized on nearly every call** — ~24 DDL/migration statements, called from ~16 places inside `neon.ts` itself with no caching, meaning a single request (e.g. one 10-minute rss-refresh cron tick) could re-run the full check 6-10+ times, and every real page load touching feeds/articles/topic-rules paid the same cost. **Fixed**: `ensureSchema()` now caches the in-flight/completed promise in a module-level `_schemaEnsured` (cleared on failure so the next call retries instead of wedging permanently), so the DDL block runs once per process lifetime instead of once per query.
- **`saveIntelligenceMentions` did one DELETE + one round-trip INSERT per mention** — called once per analyzed RSS article with up to ~90 potential mentions (keywords + individuals + entities + topics combined), each a separate network round trip to Neon's HTTP driver. **Fixed**: extracted a pure `prepareMentionBatch()` helper that dedupes by `(mention_type, normalized_value)` — the only possible collision given the table's unique constraint — keeping the last occurrence (matching the old loop's overwrite-on-conflict semantics), then a single `INSERT ... SELECT ... FROM unnest(...)` multi-row statement replaces the per-mention loop. One save call is now ~2 round trips (delete + insert) instead of up to ~91.
- **No age-based retention/pruning existed anywhere** — `rss_articles`, `rss_article_analysis`, and `intelligence_mentions` grew forever; the only DELETEs were content-quality filters (coupon/gambling/non-English), never age-based cleanup. **Fixed**: added `pruneOldRssData(retentionDays = 180)`, which deletes `rss_articles` older than the retention window (`rss_article_analysis` cascades via its FK; `intelligence_mentions` needs an explicit sweep since it's keyed generically by `source_type`/`source_id` with no FK). Wired into `/api/intel/rss-refresh` gated to run on only one cron tick per day (03:00 UTC) rather than all 144, since it's a full-table sweep. Override window via `RSS_ARTICLE_RETENTION_DAYS` env var.
- **Three cleanup queries did full-table scans on every call, against an ever-growing, unpruned table** — `deleteInvalidCouponArticles` (15 leading-wildcard `ILIKE '%word%'` conditions), `deleteNonEnglishPrNewswireArticles`, and `deleteBlockedRssArticles` (regex `~*` conditions `OR`-combined with a `feed_key LIKE` clause, which defeats index-only planning for the whole query) — none of these can use a standard index, and they're called far more often than just the 10-minute cron: also from `rss-analysis-runner.ts`, `/api/intel/feed` (twice, on every feed page load), and `/api/intel/recap/generate`. **Fixed**: all three now filter on `fetched_at >= now() - INTERVAL 'N hours'` first (default 14-day window, configurable per-call), using the existing `rss_articles_fetched_at` index to turn an O(N) sequential scan into a bounded index range scan. Trade-off: if a topic rule is edited, `deleteBlockedRssArticles` will only retroactively reclassify articles fetched within that window, not the full history — acceptable given the new retention policy above will eventually prune anything older anyway.

**Verification**: `tsc --noEmit`, `eslint --max-warnings 0`, and `next build` all pass. No `DATABASE_URL`/live Neon access exists in this sandboxed session, and there's no existing TS-side DB-mocking harness in this repo (unlike the Python side's mocked-psycopg2 pattern) — and `@neondatabase/serverless`'s `neon()` client only speaks Neon's proprietary SQL-over-HTTP protocol, so it can't be pointed at a generic local Postgres either. Docker Desktop wasn't running and didn't come up after ~5 minutes of waiting (likely needs one-time interactive WSL2 setup), so instead the exact SQL from every changed function was executed against `@electric-sql/pglite` (real Postgres compiled to WASM, in-process, no daemon needed) with the same schema from `ensureSchema()`. Verified: the batched `unnest` multi-row insert (including re-save/upsert semantics and an empty-batch no-op), all three windowed cleanup queries (confirmed a 1-day window excludes and a 30-day window includes the same 20-day-old matching row), and `pruneOldRssData` (confirmed the FK cascade into `rss_article_analysis`, the explicit `intelligence_mentions` sweep, and that recent unrelated rows survive) — all behaved exactly as intended. Also confirmed, as a negative control, that skipping the in-app `prepareMentionBatch()` dedup step and sending a raw colliding batch straight to the same SQL throws Postgres's real `ON CONFLICT DO UPDATE command cannot affect row a second time` error, proving the dedup step is load-bearing rather than defensive-only. This validates the SQL and application logic thoroughly; the one thing it does not cover is the Neon-specific HTTP driver's own parameter/array serialization, which should still get a quick real-Neon smoke test after deploy.

### GCS findings — not implemented, need direct verification or the Neon migration

- ~23 scheduled workflows read-modify-write the full `custom_documents.json` blob (36+ MB and growing) at frequencies from hourly to every 3 hours; likely 300+ full download/upload cycles/day, which is probably the single largest cost line item (egress, not storage). Reducing this requires either fewer/less-frequent full-blob rewrites or cutting readers over to the already-scoped-but-dormant Neon `documents` table (see Ingestion Pipeline Review's Neon migration Phase 2/3).
- GCS bucket-level Object Versioning/lifecycle rules are not set in application code (`gcs_storage.py`, `gcs-loader.ts`) — verify directly in the GCS console. If versioning is on, every one of those 300+ daily rewrites of a 30-40 MB object retains a full old copy forever, which could silently dominate the bill. This is the single highest-priority manual check, independent of any code change.

## Source Coverage Quick Wins - July 2026

Seven quick wins were identified to improve source coverage/quality. Reddit and X were explicitly excluded (Reddit: user hasn't picked it up as a source yet; X: already deferred per the "Deferred X Account Monitoring" section below). Two items (#5, #6) are code changes implemented and tested in this session. The other four (#1, #3, #4, #7) are operational/runbook steps that need either a production trigger (external network calls + real GCS/Neon writes, which needs your go-ahead, not something to do unilaterally) or live credentials this sandboxed session doesn't have — they're specified precisely below so they're a checklist, not vague TODOs.

### Verified live via WebFetch this session (do not re-guess these later)

Government agency press-release/RSS URLs change structure often and a wrong guess just recreates the exact "silent failure" bug fixed earlier (see Process Pipeline Review below). Before adding any new gov-source connector, verify the URL live rather than trusting training-data memory. Verified in this session (all returned live, current — June/July 2026 — items):

- **OCC**: RSS at `https://www.occ.gov/rss/occ_news.xml` (also has separate bulletins/speeches/testimony feeds at `/rss/occ_bulletins.xml`, `/rss/occ-speeches.xml`, `/rss/occ-congressional-testimony.xml` if wanted later).
- **FDIC**: RSS at `https://public.govdelivery.com/topics/USFDIC_26/feed.rss`.
- **CFPB**: RSS at `https://www.consumerfinance.gov/about-us/newsroom/feed/`.
- **NYDFS**: no RSS feed exists. HTML listing at `https://www.dfs.ny.gov/reports_and_publications/press_releases`, individual releases at predictable paths like `/reports_and_publications/press_releases/pr20260701` — fits the existing `html_links` discovery mode.
- **FinCEN**: **could not verify** — `fincen.gov` returned repeated timeouts/socket-hangs (5 attempts, different paths), which reads as active bot-blocking rather than a slow page (consistent with why this codebase already has `webshare_proxy.py` for exactly this class of site). No FinCEN connector was added directly; a Google News query connector (`google_news_fincen_enforcement_article`, see below) covers FinCEN enforcement news as a workaround that doesn't depend on scraping fincen.gov directly. If a direct FinCEN connector is wanted later, verify the URL from a proxied/residential-IP context, not a repeat of this session's approach.

### 5. Add banking/consumer-finance regulator feeds — DONE (OCC, FDIC, CFPB, NYDFS)

Prior coverage was SEC (5 feeds) + FINRA (5) + CFTC (5) with zero banking/AML-regulator coverage, despite AML being a first-class topic (`aml-news-ingest.yml`) and topic rule. Implemented by extending `securities_market_sources_scraper.py` (already hosts non-strictly-securities official sources like MSRB and PCAOB, so this reuses tested, working generic scraper machinery instead of a new near-duplicate file) with four new source configs and registering them in `SECURITIES_MARKET_CONNECTORS`:

- `occ_news_release` (RSS)
- `fdic_press_release` (RSS)
- `cfpb_newsroom` (RSS)
- `nydfs_press_release` (`html_links` mode, path filter `/reports_and_publications/press_releases/pr`)

Wired into `connector-enrichment-6hour.yml`'s enrichment matrix alongside the existing securities-market connectors. Not added to a new extraction-schedule workflow in this pass — see whichever workflow currently runs `SECURITIES_MARKET_CONNECTORS` extraction and add these `--connector` values there too if a schedule is wanted (check `connector-gap-6hour.yml` first, since PCAOB/MSRB already run there).

### 6. Google News query connector for FinCEN coverage — DONE (uncovered and fixed a bigger pre-existing bug)

Added `google_news_fincen_article` to `trade_media_scraper.py`'s `TRADE_MEDIA_SOURCES`/`TRADE_MEDIA_CONNECTORS` (query: `"FinCEN" OR "Financial Crimes Enforcement Network" when:7d`) as the practical substitute for the direct FinCEN connector that couldn't be verified (see above).

**Bigger finding while verifying it live**: copying the existing `google_news_ponzi_investor_fraud_article`/`google_news_senate_committee_article` pattern reproduced their behavior exactly — and live-testing exposed that **both already-shipped Google News connectors have never actually returned their intended search results in production.** Two compounding, pre-existing bugs, neither introduced by this session:

1. **Wrong discovery path chosen.** `discover_documents()`'s fallback order is RSS-feed-parse → generic-listing-scrape → `_discover_from_google_news_search` (the dedicated, decode-aware path). For these cross-site topic aggregators (`default_url` is `news.google.com` itself, so there's no "home site" to scrape), `_discover_from_feed` always returned zero items — Google's redirect links contain `/rss/articles/`, and `_looks_like_article_url`'s generic feed-link skip filter rejects any path containing `/rss`. That silently fell through to scraping the **unrelated `https://news.google.com/` homepage** as a generic listing page, which "succeeded" (so the correct `_discover_from_google_news_search` fallback was never reached) and returned random, off-topic homepage links with garbled titles (`_title_from_url()` applied to the base64 redirect slug, e.g. `"Cbmijgfbvv95Cuxq..."`).
   - **Fixed**: `discover_documents()` now detects `_canonical_host(source_url) == "news.google.com"` and routes straight to `_discover_from_google_news_search()`, skipping the RSS/listing attempts entirely for these sources.
2. **Even when reached, the domain filter would have rejected every real result.** Both `_build_google_news_query` and `_discover_from_google_news_search` computed `search_domain` as `cfg.get("search_domain") or _canonical_host(source_url)` — for these 3 connectors that fallback resolves to `"news.google.com"` (since `source_url` is literally the Google News homepage), and the existing configs also set `"search_domain": "news.google.com"` explicitly. After decoding a redirect link to its real publisher URL (e.g. `consumerfinancemonitor.com`), the post-decode filter (`_host_matches(decoded_url, search_domain)`) would reject it for not matching `news.google.com` — so this path would have returned zero results too, or only undecoded raw redirect links.
   - **Fixed**: added `_google_news_site_scope_domain()`, which only uses an *explicit* `search_domain` config value or a *non*-`news.google.com` canonical host; removed the now-incorrect `"search_domain": "news.google.com"` from all 3 configs. Also removed `rss_candidates`/`article_path_keywords`/`minimum_path_segments`/`required_url_substrings` from the 3 configs, since they belonged to the generic feed/listing path these connectors never use — `required_url_substrings` in particular would reject every successfully-decoded (real, non-Google) URL if left in.
   - The recency scoping previously baked into the now-removed `rss_candidates` URLs (`when:7d`) is preserved by moving it into `default_search_query` directly, since that's the text `_build_google_news_query` actually sends.

Verified live end-to-end after the fix: all 3 connectors now discover real, correctly-decoded, on-topic articles from real (varied) publisher domains with real titles, and extraction succeeds. Covered by new tests in `tests/test_trade_media_scraper.py` (`test_build_google_news_query_does_not_site_scope_cross_site_topic_aggregators`, `test_google_news_search_domain_filter_is_skipped_for_cross_site_topic_aggregators`, `test_discover_documents_routes_google_news_primary_sources_directly`). The Senate-committee connector's per-committee `site:X.senate.gov` sub-query scoping (previously 6 separate `rss_candidates` fetches merged) was consolidated into one phrase-based OR query without site-scoping, since `_discover_from_google_news_search` only supports a single query per run — verified this still returns genuinely on-topic committee-related results, but it's a real (believed net-positive) behavior change worth knowing about, not just a bug fix.

### 1. Backfill the 16 restored connectors — NOT EXECUTED, needs your go-ahead

The connectors restored in the Process Pipeline Review fix (commit `52292f9`) were dead for ~2-3 days before the fix shipped. The regular schedule's `selection: new_or_updated` with modest `--limit`/`--max-pages` may not reach back through that gap on high-volume sources (PR Newswire, CoinDesk, The Hacker News). Runbook: trigger `workflow_dispatch` on `rss-full-ingestion-3hour.yml` and `cyber-sources-3hour.yml` with `extraction_limit: 50` and `max_pages: 4`. Not run in this session because it's a production action (real external fetches + real writes to the production GCS bucket), not a code change — needs explicit confirmation before triggering.

### 3. Audit generic topic-rule keywords with the new backtest tool — NOT EXECUTED, needs live DB access

Still-open item from the Process Pipeline Review: `DEFAULT_TOPIC_RULES` keywords like `"market"`, `"credit"`, `"economy"`, `"stock"` are broad enough to nearly always match, and they gate ingestion for keyword-filtered feeds. The backtest tool built in the previous session (`POST /api/admin/topic-rules/backtest`, live in the admin UI) makes this a live, empirical, ten-minute check — but it requires a live Neon `DATABASE_URL` connection this sandboxed session doesn't have. Runbook: open `/admin` → Topic Rules → expand each rule with a broad keyword → "Preview Matches" → drop or narrow any keyword with a disproportionate per-keyword hit count relative to the others in its rule.

### 4. Add YouTube channels via the managed-channels admin — NOT EXECUTED, needs live admin access + channel refs

Infrastructure already exists (config, scheduling, transcripts, enrichment) and takes new channels with zero code — see `scripts/run_youtube_channel_sources.py` and the admin YouTube-channels UI. Candidates worth adding: Senate Banking Committee, House Financial Services Committee, CFTC, and the Federal Reserve's channel (FOMC press conferences) — verify each channel's current handle/ID live before adding, same caveat as the RSS URLs above. Not done in this session: requires live admin login with real GCS/YouTube credentials, and the actual channel refs weren't confirmed live.

### 7. Monitor the new per-feed failure tracking — NOT EXECUTED, inherently a future check

`rss_feeds.last_error`/`consecutive_failures` (added in the Ingestion Pipeline Review fixes) needs a few days of real traffic to accumulate signal. Runbook: check the admin Feeds panel in ~1 week; prune or fix any feed with a persistent failure streak.

## Enhancement Scope: Storage, Health-Monitoring Blind Spot, Topic Backtesting - July 2026

Three enhancements were proposed after the ingestion/process reviews below. Status: **items 2 and 3 fully implemented; item 1 implemented as a deliberately conservative Phase 1 only** (schema + non-blocking dual-write, no reader cutover, and the mirror is currently dormant in production since no workflow has `DATABASE_URL` configured yet) — see the caveats under item 1 before assuming more was done.

### 1. Migrate the corpus from one GCS JSON blob to Neon (plumbing) — PHASE 1 ONLY

The full case for this is already documented in the Ingestion Pipeline Review section below: almost every serious bug found across both reviews (silent wipes, lost-update races, O(N) upsert scans, 240-minute workflow timeouts) traces back to `custom_documents.json` being one JSON blob that ~15 workflows read-modify-write in full. `rss_articles` already proves the alternative works (per-row upserts, no contention).

**DONE — what was actually implemented (Phase 1 of an incremental migration):**
- A `documents` table (schema lives in `neon_feeds.py`'s `_ensure_documents_schema()`, called lazily on first write — **not** in `apps/web/lib/server/neon.ts`'s `ensureSchema()`, since the only writer so far is the Python side). Typed columns for the commonly-filtered fields (`document_id` PK, `title`, `speaker`, `organization`, `doc_type`, `source_kind`, `url`, `published_date`, `word_count`, `full_text`) plus a `metadata JSONB` column (GIN-indexed) for everything else, mirroring the hybrid typed-columns-plus-JSONB pattern already used by `rss_article_analysis`.
- `neon_feeds.mirror_document(record)` does the actual upsert (`ON CONFLICT (document_id) DO UPDATE`).
- A best-effort, non-blocking call site in `_upsert_custom_document_record` (`run_financial_news_pipeline.py`, the single choke point both pipeline scripts funnel through) and in `app.py`'s duplicate copy of the same function. **Cheaply skips entirely (no log noise) when `DATABASE_URL` isn't set** — confirmed only 1 of the ~15 extraction/ingest workflows currently sets it, so this is a true no-op almost everywhere today. Only logs (and never raises) if `DATABASE_URL` *is* set but the write still fails. Verified with mocked psycopg2 (`tests/test_neon_document_mirror.py`, 6 tests) that the primary in-memory upsert's return value and corpus state are byte-for-byte identical whether the mirror is skipped, succeeds, or throws.
- **No reader was cut over, and the mirror is currently dormant in production** (see above — no workflow has `DATABASE_URL`). The blob remains the sole source of truth everywhere. This table is write-only and, right now, mostly not even being written to.

**Deliberately NOT done, and left as a decision for you (not a technical gap):** wiring `DATABASE_URL` into the ~15 extraction/ingest workflow YAML files so the mirror actually starts populating in production. That's a credential/security decision (granting live DB write access to more scheduled CI jobs) that shouldn't be made unilaterally. Once you decide to do that, the mirror will start populating with no further code changes needed.

**Remaining phases (not started):**
- Phase 1.5: decide on and wire `DATABASE_URL` into the relevant workflows (see above) so the mirror actually starts running.
- Phase 2: one-time backfill script (`sync_knowledge_index.py`-style) reading the full blob and writing every document into the new Neon table; verify row counts and spot-check content match.
- Phase 3: cut over readers one at a time, cheapest/lowest-risk first — likely `/api/metrics` (already has a bounded/partial-failure pattern per the smoke-fix section below) or enrichment-candidate selection (`_build_news_enrichment_candidates`), which would immediately benefit from an indexed "unenriched docs of kind X" query instead of a full-corpus scan.
- Phase 4: once all readers are cut over and stable for an observation period, stop writing `custom_documents.json` entirely and keep it only as a periodic read-only export/backup.

### 2. Close the "invisible failure" blind spot in source health monitoring — DONE

**Important correction to the original pitch:** the "ops dashboard with alerting" enhancement was pitched as if it didn't exist. It substantially already does — `source_health.py`'s `build_source_health_report` already computes `failing_sources` (status/consecutive-failure based), `stale_sources` (>36h since last run), and `quiet_sources` (zero discovered/processed), all rendered in `apps/web/app/admin/page.tsx`'s `SourceHealthSection` and auto-posted as a GitHub issue by `daily-health-check.yml` with a DeepSeek review. Do not rebuild any of this.

The actual, narrow, verified gap: **argparse validation failures never reach `record_source_health()`.** Confirmed empirically — `python run_connector_extraction_pipeline.py --connector totally_fake_connector` exits 2 via argparse before the script's own try/except (which is what calls `record_source_health`) ever runs, so it produces zero entries in `source_health_log.json`. This is exactly the blind spot that let the July 7 merge regression (16 dropped connectors, see Process Pipeline Review below) run silently for days despite this otherwise-comprehensive monitoring system.

**Fix**: `source_health.py` gained `RecordingArgumentParser`, an `argparse.ArgumentParser` subclass whose `error()` override logs a `source_key`-bearing failure entry (extracting the attempted `--connector`/`--source-kind` value from `sys.argv` for diagnostics) before deferring to normal argparse error/exit behavior, so it fails exactly the same way to the caller — the only change is that the failure is now visible in `failing_sources`. Wired into `_build_parser()` in both `run_connector_extraction_pipeline.py` and `run_financial_news_pipeline.py` (subparsers inherit the class automatically). Added an `invalid choice` branch to `categorize_error`. The logging call is wrapped so it can never itself raise and mask the real argparse error.

### 3. Topic rule keyword backtest/preview tool — DONE

Distinct from the existing static `apps/web/lib/topic-rule-recommendations.ts` (hand-curated suggested keywords, no live data). This is empirical: a new admin API route runs a candidate keyword set against recently stored `rss_articles` using the same word-boundary matcher as `intel-topic-matching.ts`, and returns total match count, per-keyword hit counts, and sample matched titles. Wired into the topic-rule editor in `apps/web/app/admin/page.tsx` as a live preview so editing keywords shows real impact before saving, instead of finding out days later that a keyword over- or under-matches. This is also the tool to use for the still-open item from the Process Pipeline Review below (auditing generic `DEFAULT_TOPIC_RULES` keywords like `"market"`/`"credit"` that gate keyword-filtered RSS ingestion).

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
