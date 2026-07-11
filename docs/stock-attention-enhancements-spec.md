# Stock Attention Tracker — Enhancement Spec (10 Items)

Status: **plan only — nothing below is implemented.** Written 2026-07-11, the same day v1 (phases 1-4 of [stock-attention-spec.md](stock-attention-spec.md)) went live in production with its first real day of data (AAPL/NFLX/MSFT leading, 107 tickers, clean gating). This doc is the roadmap for what comes after Phase 5 validation.

Each item is specified to be buildable without re-deriving context: data source (and whether it's verified or needs live verification at build time), schema, pipeline placement, cadence, cost, risks, effort, dependencies, and a concrete done-when. Items are numbered in **recommended build order**, not by size.

**Cross-cutting rules that apply to every item** (from CLAUDE.md's established conventions):
- Verify external URLs/APIs live at build time — never trust training-data memory for gov/vendor endpoints (the FinCEN/OCC lesson).
- New scheduled scripts wire into `record_source_health()` so `daily-health-check.yml` sees failures (the argparse blind-spot lesson).
- All text into Neon passes `_strip_nul_bytes()`/`_sanitize_for_json()` (the batch-72 backfill lesson).
- New tables are Python-owned via `neon_feeds.py` lazy-ensure unless the web tier writes them; web readers degrade to empty-with-warning, never 500 (the pattern `/api/market/attention` already uses).
- Every derived score keeps its raw inputs in columns so formulas can be re-tuned and history re-computed without re-ingesting (the §6.2 principle).
- No new LLM calls in aggregation paths; LLM spend only where text understanding is genuinely needed, and then with the model-tiering rules from the DeepSeek review.
- Label all user-facing output "Research context only — not investment advice."

**Shared prerequisite (small, do first):** regenerate `ticker_config.json` retaining `cik_str` per ticker (build_ticker_config.py currently keeps only ticker/title — verified in code). Items 7 and 8 join EDGAR filings to tickers via CIK; adding the field now avoids a second regeneration later. One-line change + regenerate + commit.

---

## 1. News-channel attention (multi-source counting, not just Reddit)

**What/why:** The original roadmap item 2 scope — ticker mentions from content the app *already ingests* (RSS articles, NewsAPI/Bloomberg/Substack documents, enrichment entities) — was set aside when the live Reddit sweep proved out. Build it now: run `ticker_resolver` over stored article titles/descriptions and over enrichment `entities` (via the alias map for company names), writing `mention_type='ticker'` rows with the existing `source_type='rss_article'` / document source types. The leaderboard then distinguishes *retail chatter* from *news coverage* — the divergence between them is the interesting signal (news-heavy + Reddit-quiet = institutional story; Reddit-heavy + news-quiet = retail-driven or early).

**Data source:** already in Neon/GCS. Zero new external access.
**Schema:** no new tables. `daily_stock_attention` gains `news_count INTEGER DEFAULT 0` and `reddit_count INTEGER DEFAULT 0` (rename-safe: `mention_count` stays the total). Aggregation groups by source channel.
**Pipeline:** hook into `rss-analysis-runner`/`saveRssArticleAnalysis` (TS side already writes entity mentions — add ticker resolution there via a TS port of *only* the cashtag/curated-name tiers, or simpler: a Python pass inside the daily rollup that scans the day's `rss_articles` rows via a new indexed query). Prefer the Python-side pass: no TS resolver port, no webpack bundle concerns, one place to tune.
**Cadence:** inside the existing `stock-attention-daily.yml` rollup — no new workflow.
**Cost:** none. **Effort:** Small-Medium. **Depends on:** nothing.
**Done when:** attention rows show per-channel counts; UI leaderboard gains a small "News" count column; a ticker appearing only in news (zero Reddit) still ranks.

## 2. Price/volume market context + divergence flags

**What/why:** Pair each attention row with same-day market behavior: close-over-close %, volume vs. 20-day average volume (the "unusual volume" signal), and a computed `divergence` flag — `attention_spike_no_price_move` (possible early signal or pump) and `price_move_no_attention` (institutionally driven). This is what turns a chatter list into a research tool, and it's the "what moved + what's talked about" pairing CLAUDE.md item 2 named as the point of putting this on `/market`.

**Data source:** Yahoo chart endpoint (keyless, already used by `fetchYahooQuote`; the same endpoint returns daily OHLCV with `range=1mo` for the volume baseline). Finnhub is the fallback — but note **FINNHUB_API_KEY exists only in Vercel env, not GitHub Actions secrets**; the rollup runs in Actions, so either use Yahoo (recommended, keyless) or add the key as an Actions secret.
**Schema:** `daily_stock_attention` gains `price_close NUMERIC`, `price_pct NUMERIC`, `volume BIGINT`, `volume_vs_20d NUMERIC`, `divergence TEXT DEFAULT ''`. Raw inputs stored, flags derived.
**Pipeline:** in the daily rollup, after ticker aggregation: fetch OHLCV for the rolled-up tickers only (~100-300/day), batched with polite pacing (Yahoo is unofficial — throttle ~5/s, accept nulls). Fetch failures leave columns null, never fail the rollup.
**Cadence:** existing daily rollup. **Cost:** none (Yahoo) with rate care. **Effort:** Medium. **Depends on:** nothing.
**Risks:** Yahoo's endpoint is unofficial and can change/block — isolate behind one fetch function, treat nulls as normal, and health-log the null rate so silent decay is visible.
**Done when:** leaderboard shows price/volume columns from stored data (dropping the request-time Yahoo pairing for past days), and rows carry divergence badges with a tooltip explaining the flag.

## 3. Historical trends: sparklines, ticker detail, and the intraday view

**What/why:** Single-day snapshots hide the story; every reference site's most-used feature is the trend line. Three surfaces: (a) a 14-day sparkline column in the leaderboard (ApeWisdom's 30d trend, shortened), (b) a per-ticker drawer/page with mention + price time series and day-by-day top threads, (c) the deferred "hot right now" intraday view using `reddit_attention_items.created_utc` directly with the freshness decay that §6.2 deliberately kept out of the daily rollup.

**Data source:** already stored — `daily_stock_attention` persists indefinitely; intraday reads `reddit_attention_items` (90-day retention is ample).
**Schema:** none new. New API: `GET /api/market/attention/history?ticker=X&days=30` and `GET /api/market/attention/intraday` (last-24h buckets by hour, decay-weighted).
**Pipeline/UI:** SVG sparkline inline in the table (no chart lib — repo convention is hand-rolled minimal SVG); drawer extends the existing expand-row pattern.
**Cadence:** request-time reads. **Cost:** none. **Effort:** Medium (mostly UI). **Depends on:** a week-plus of accumulated rollups to be worth shipping; item 2's stored prices make the detail chart much better.
**Done when:** leaderboard has sparklines; clicking a ticker shows a 30-day mention/price chart; an "Intraday" toggle shows the decay-weighted last-24h board.

## 4. Subreddit expansion + admin-managed sweep config

**What/why:** Tier 2 (r/pennystocks, r/Shortsqueeze, r/SqueezePlays, r/smallstreetbets, r/ValueInvesting, r/dividends) is already spec'd and gated on Phase 5 validation. Beyond flipping it on: move the subreddit list, per-subreddit weights, bot blocklist, and ambiguous-symbol additions from code/env into a Neon-backed config table with an admin panel — the same feeds/topic-rules CRUD pattern `admin/page.tsx` already has, so tuning stops requiring commits.

**Data source:** n/a (config). **Schema:** `attention_sweep_config` (single JSONB row, versioned) or three small tables (`sweep_subreddits(name, tier, weight, active)`, `sweep_bot_blocklist(name)`, `sweep_symbol_overrides(symbol, disposition)`), read by the sweep at start, env-fallback if unreachable.
**Pipeline:** sweep reads config first; `aggregate_stock_attention.py` applies per-subreddit weights in `weighted_score` (weights default 1.0 — the formula's inputs stay stored raw).
**Cadence:** unchanged. **Cost:** Tier 1+2 ≈ 252 calls/sweep, still trivial under PRAW throttling. **Effort:** Medium (admin UI is most of it). **Depends on:** Phase 5 validation before enabling Tier 2 by default.
**Risks:** Tier 2 boards are where manipulation lives — enable together with item 6's defenses, and weight them < 1.0 initially.
**Done when:** an admin can add a subreddit, adjust its weight, blocklist a bot, and force-gate a symbol without a deploy; Tier 2 enabled with sub-1.0 weights.

## 5. Author-level tracking & credibility weighting

**What/why:** The per-author dedup already neutralizes single-account spam; the next layer is knowing *who* is talking. Aggregate per-author history from data already stored (`reddit_attention_items.author`): distinct tickers mentioned, days active, subreddit spread, account concentration (share of activity on one ticker). Two uses: (a) **discount** low-diversity accounts (a 3-day-old account posting one ticker 40 times should weigh near zero even deduped — currently it still counts as 1 full mention), (b) **surface** consistently-early authors (mentioned tickers that subsequently moved — computable once item 2 stores prices).

**Data source:** already stored; optionally enrich with account age/karma via PRAW (`redditor(name)` — 1 call/author, cache aggressively, only for authors above an activity threshold; ~tens of calls/day, not thousands).
**Schema:** `reddit_author_stats(author PK, first_seen, last_seen, items_total, tickers_distinct, subreddits_distinct, top_ticker_share NUMERIC, account_created TIMESTAMPTZ NULL, link_karma INT NULL, refreshed_at)` — recomputed daily from raw items (a pure rollup, so it inherits the re-runnable property).
**Pipeline:** new section of the daily rollup script (no new workflow). `mention_count` gains a companion `weighted_mention_count` where authors with `top_ticker_share > 0.8 AND tickers_distinct <= 2` count at 0.25 — thresholds stored in the item-4 config table.
**Privacy/compliance:** usernames are public and already retained ≤90 days; author *stats* are derived aggregates — keep the same 90-day inactivity expiry, never display author stats in the UI beyond what's already on the permalink (the drill-down already shows u/name as Reddit does).
**Cost:** near zero. **Effort:** Medium. **Depends on:** item 4's config table for thresholds (soft), item 2 for the "early author" analysis (later phase of this item).
**Done when:** the rollup writes author stats + weighted counts; a known-pattern test account (fixture) is discounted in tests; leaderboard optionally sorts by weighted count.

## 6. Manipulation defense & data-quality review tooling

**What/why:** Institutionalize Phase 5 instead of doing it once: (a) an admin **review queue** — every ticker newly appearing in a day's top 50, with its sample source texts, one-click "legit / false-positive → auto-add to ambiguous list" (writes item 4's config); (b) **coordination signals** — author-overlap clustering (same small author set driving one ticker across days), account-age clustering (many young accounts on one ticker — needs item 5's `account_created`), single-thread concentration; (c) a per-row `quality_flags TEXT[]` surfaced as a muted warning icon, never silently dropping data.

**Data source:** internal. **Schema:** `attention_review_queue(date, ticker, status, reviewed_by, sample_source_ids)`; `quality_flags` column on `daily_stock_attention`.
**Pipeline:** flags computed in the daily rollup (pure functions, unit-testable: overlap coefficient between the ticker's author set and its prior-day author set; share of mentions from accounts < 30 days old; share from one thread). Review queue populated on rollup; admin panel actions write back to config.
**Cadence:** existing rollup + human review as-needed. **Cost:** none. **Effort:** Medium-Large (admin UI + clustering logic). **Depends on:** items 4 (config writes) and 5 (account ages) for full strength; the review queue alone depends on neither.
**Done when:** a pumped fixture dataset trips each flag in tests; a false positive found in review can be killed and history re-run (`--date`) without a deploy.

## 7. SEC filings catalyst overlay (8-K and friends)

**What/why:** Attention spikes usually have a cause. Overlay same-day SEC filings — 8-K (material events) foremost, plus 10-Q/10-K/S-1/13D/13G — so each leaderboard row can show a "catalyst" badge: *this ticker filed an 8-K today* (with item type, e.g. 8-K Item 1.01, and a link). Divergence matters here too: attention spike with **no** filing and no news (item 1) is the "something's brewing on Reddit alone" signal.

**Data source:** SEC EDGAR daily index / submissions API (`data.sec.gov/submissions/CIK##########.json` per company, or the daily full-index for everything). Free; hard rules: max 10 req/s, declared User-Agent, and this repo already knows sec.gov requires `curl_cffi`. **Verify exact endpoints live at build time** per repo convention. Join via CIK — requires the shared prerequisite (CIK retained in `ticker_config.json`).
**Schema:** `sec_filings_daily(accession_no PK, cik, ticker, form_type, item_codes TEXT[], filed_at, url)` — only filings whose CIK maps to a swept-universe ticker, ~hundreds/day.
**Pipeline:** new `fetch_sec_filings_daily.py` + workflow (cron ~22:30 UTC after EDGAR's day closes; `workflow_dispatch --date` for backfill, same idempotent pattern). The attention rollup (00:15 UTC, after it) joins and sets `catalyst` fields on rows.
**Cost:** none. **Effort:** Medium. **Depends on:** CIK prerequisite; item 1/2 for the full divergence story (soft).
**Risks:** EDGAR structure changes; index-vs-submissions API choice needs the live check. Form parsing kept to metadata only (form type + item codes) — no document text extraction in this item.
**Done when:** an 8-K filed by a top-50 attention ticker shows a catalyst badge with a working EDGAR link, and rows carry `catalyst_form`/`catalyst_url` columns.

## 8. Insider trading overlay (Forms 3/4/5)

**What/why:** The sharpest catalyst class gets its own item: insider buys/sells per ticker per day from Form 4 filings. Retail attention + insider *selling* is a materially different picture than attention + insider *buying* — a red/green badge worth more than any sentiment heuristic. (Congressional trading disclosures — the StonkWhisper paid tier — are a possible later extension via the House/Senate disclosure sites, but those need their own access verification and are **not** in scope here.)

**Data source:** EDGAR again — Form 4 XML is standardized (`ownershipDocument`: transaction code P/S, shares, price, insider role). Same rate rules, same CIK join, same curl_cffi. Verify the current XML index path live at build.
**Schema:** `insider_transactions(accession_no PK, cik, ticker, insider_name, insider_role, tx_code, tx_date, shares NUMERIC, price NUMERIC, value NUMERIC, filed_at, url)`; daily per-ticker rollup columns on `daily_stock_attention`: `insider_net_value NUMERIC NULL`, `insider_tx_count INT`.
**Pipeline:** extends item 7's fetch script (same daily index pass discriminates form types) — build 7 first, this rides its plumbing. XML parse via `defusedxml` (already a transitive dep).
**Cadence/cost:** as item 7. **Effort:** Medium (XML parsing + tests are the bulk). **Depends on:** item 7.
**Compliance note:** Form 4 data is public disclosure; presenting it factually with links is standard practice — keep the research-only label and never editorialize ("insider dumping!") in generated text.
**Done when:** a ticker with same-day Form 4 activity shows net insider buy/sell value in its drawer, with per-transaction rows linking to EDGAR.

## 9. StockTwits as a second forum

**What/why:** The highest-value non-Reddit forum: cashtag-native (near-zero false positives), finance-only, and posts carry **user-declared** bullish/bearish labels — ground-truth sentiment that both replaces the keyword mood heuristic for this channel and provides a calibration set for Reddit's heuristic (how often does keyword-mood agree with declared sentiment on similar text?).

**Data source:** StockTwits public API (symbol streams + trending endpoint). **Access model must be verified live before building** — rate limits (~200/hr unauthenticated historically, higher with a free key) and current ToS for aggregate republication; this item is contingent on that check passing, same posture as the Reddit commercial-terms flag.
**Schema:** reuses everything: items → `reddit_attention_items`… no — new `stocktwits_attention_items` mirroring its shape (`source_id` = ST message id, `declared_sentiment TEXT NULL` extra column), mentions → `intelligence_mentions` with `source_type='stocktwits_post'`. Rollup gains `stocktwits_count` alongside item 1's channel columns.
**Pipeline:** new `stocktwits_attention_sweep.py` + hourly workflow (offset :35). Sweep strategy: trending endpoint + symbol streams for the prior day's top ~50 tickers (bounded call budget ~60/hr) rather than firehose.
**Cost:** free tier; budget-capped by design. **Effort:** Medium. **Depends on:** item 1's channel-count columns (soft); ToS verification (hard gate).
**Done when:** leaderboard channel breakdown includes StockTwits; declared-sentiment agreement rate with the keyword heuristic is logged in the rollup summary (the calibration metric).

## 10. Attention alerts & watchlist integration

**What/why:** Close the pull→push gap for this feature the same way the recap review framed it. Triggers computed at rollup time: (a) ticker enters the top 10 for the first time in N days, (b) mention count z-score > 3 vs. its trailing 14-day mean, (c) "new ticker" (never seen before), (d) divergence flags from item 2, (e) catalyst coincidence from items 7/8 (attention spike + 8-K + insider selling in one alert is the headline case). Delivery starts with the zero-credential GitHub-issue digest pattern (`daily-health-check.yml` precedent); per-user delivery arrives with roadmap item 1's watchlists (`kind='ticker'` slots straight into the existing watchlist spec since mentions already live in `intelligence_mentions`).

**Data source:** internal. **Schema:** `attention_alerts(id, alert_date, ticker, kind, detail JSONB, delivered_at NULL)`.
**Pipeline:** pure-function trigger evaluation in the daily rollup (z-scores need only stored rollup history); a small digest step posts one GitHub issue per day when alerts exist, collapsing to nothing on quiet days.
**Cadence:** daily initially; hourly alerts only after the intraday view (item 3) exists to link to. **Cost:** none. **Effort:** Small-Medium. **Depends on:** ~2 weeks of rollup history for meaningful z-scores; items 2/7/8 enrich alert kinds but (a)-(c) work standalone.
**Done when:** a synthetic spike fixture fires each trigger in tests; a daily digest issue appears with ticker links into the history UI (item 3).

---

## Explicitly excluded (decided, not forgotten)

- **X/Twitter** — remains deferred per CLAUDE.md's "Deferred X Account Monitoring" (cost model already documented there). Do not add via scraping; the compliance defaults prohibit it.
- **4chan /biz** (ApeWisdom tracks it) — no stable API, high toxicity/noise, manipulation-first culture; the signal isn't worth the moderation surface.
- **Discord/Telegram groups** — private/semi-private access models conflict with the compliance defaults (no private groups, no ToS bypass).
- **LLM-scored sentiment per post** — volume × cost is exactly the spend pattern the DeepSeek review eliminated elsewhere; StockTwits' declared sentiment (item 9) is free ground truth instead.
- **Congressional trading feeds** — noted inside item 8 as a later extension needing its own verification; not one of the 10.

## Sequencing at a glance

| Order | Item | Size | Hard dependency |
|---|---|---|---|
| 0 | CIK in ticker_config.json | XS | — |
| 1 | News-channel attention | S-M | — |
| 2 | Price/volume context + divergence | M | — |
| 3 | History UI + intraday | M | data accumulation |
| 4 | Subreddit expansion + admin config | M | Phase 5 validation |
| 5 | Author tracking & weighting | M | — (4 soft) |
| 6 | Manipulation defense + review queue | M-L | 4, 5 (partial) |
| 7 | 8-K catalyst overlay | M | item 0 |
| 8 | Insider trading overlay | M | item 7 |
| 9 | StockTwits | M | ToS check (gate) |
| 10 | Alerts & watchlists | S-M | rollup history |

Items 1-3 are the "make the existing board genuinely useful" tier and touch no new external services. Items 4-6 are the trust tier (required before Tier 2 subreddits or any public promotion of the numbers). Items 7-8 are the catalyst tier on shared EDGAR plumbing. Items 9-10 broaden inputs and outputs last, when the core is trustworthy.
