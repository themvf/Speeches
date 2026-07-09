# CLAUDE.md

## Ingestion Pipeline Review - July 2026

Use this note when debugging missing/duplicated documents, corpus data loss, or scheduled ingestion job conflicts. Findings below are from a full review of the connector extraction, enrichment, and RSS ingestion pipeline; fixes are proposed but not all applied yet — check git history/blame on the referenced files to see current status before assuming a finding is still open.

### Critical findings

- **A transient GCS read failure can silently roll back or wipe the corpus.** Both `_load_json_store` (`run_financial_news_pipeline.py`) and `POST /api/admin/documents` (`apps/web/app/api/admin/documents/route.ts`) treat a failed remote read the same as "no data exists," then save that empty/stale payload back over the real corpus in `custom_documents.json`. `--require-remote-persistence` only guards writes, not reads. Fix: track whether the remote load actually succeeded and refuse to save under `require_remote_persistence` if it fell back to local/empty; make `downloadGcsJson` distinguish not-found from error.
- **All YouTube videos collapse to one dedup key.** `_url_match_key` (`run_financial_news_pipeline.py`) strips query strings, so every `youtube.com/watch?v=<id>` URL normalizes to the same key — each new video ingested overwrites the previous one in `custom_documents.json`, meaning the YouTube channel connectors (`youtube_video`, `sec_youtube_video`) are losing almost all transcripts. Also collapses `http://` vs `https://` variants of any URL. Fix: keep the query string (or canonicalize YouTube URLs to `youtu.be/<id>`) and normalize scheme before hashing; add a regression test asserting two `watch?v=` URLs produce distinct keys; backfill/re-key existing records.
- **Lost-update races between concurrent workflows writing the same JSON blob.** ~15 scheduled GitHub Actions workflows plus Vercel admin routes all do unguarded read-modify-write of `custom_documents.json` / `document_enrichment_state.json` / `rule_summaries.json` with only per-workflow concurrency groups; overlapping schedules (e.g. bloomberg-public-hourly and connector-gap-6hour both at minute 0) can silently drop each other's writes. Fix: optimistic concurrency via GCS `generation` match with retry-and-reapply on conflict (first step); longer-term, migrate documents to normalized Neon storage instead of one JSON blob.

### High/medium findings

- Single-JSON-blob storage for the whole corpus doesn't scale (O(N) linear scans, full re-upload per ~25 new docs) and is the root cause of the race above — plan a Neon-backed schema migration.
- `/api/intel/rss-refresh/route.ts` is fail-open (skips auth entirely) if `CRON_SECRET`/`RSS_REENRICH_SECRET` are both unset — should fail closed (503) like the admin middleware does.
- `neon_feeds.py add_feed()` still inserts with `ON CONFLICT (feed_url)` instead of `feed_key`, reintroducing the exact `rss_feeds_feed_key_key` duplicate-key bug already fixed in `apps/web/lib/server/neon.ts` (see smoke-fix section above).
- `_should_fail_for_item_failures` (`run_connector_extraction_pipeline.py`) fails the whole workflow run on any single item-level extraction failure for most connectors, even at 29/30 success — creates alarm fatigue that trains you to ignore real failures.
- `markFeedRefreshed` runs in a `finally` block in `rss-refresh/route.ts`, so persistently-failing feeds still look "recently refreshed" to `dueOnly` scheduling and their failures become invisible over time.
- Metadata-fallback stub records (`_build_short_text_fallback`) flow into DeepSeek enrichment unfiltered — wastes enrichment calls summarizing placeholder text instead of skipping until real body text is recovered.

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
