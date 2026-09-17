# ZCAT and Zcash: profile, bio, and influence tracking

This implements the next bounded research phase in Market → Crypto. The existing
GitHub `DATABASE_URL` and `TWITTERAPI_IO_API_KEY` secrets are reused. No provider
key is needed in Vercel, and dashboard reads never call TwitterAPI.io.

## Activation and lifetime

`Crypto Social Tracking (30-day pilot)` runs at 07:17 UTC daily once merged into
main. GitHub scheduled jobs may run later than their nominal time. It can also
be run manually with `execute=true`; the default manual run only prints a plan.
The first executed run creates additive tables/views and fixes a 30-day end
instant. Later executions never extend that end time or reset credit usage.
The schedule checks the end date and exits without paid calls after the pilot.

The first run completes up to one page per original six-hour search window.
It then selects up to 20 accounts per coin, including interaction targets whose
own posts were not collected. This is an automatically selected **provisional**
cohort ranked by distinct amplifiers, distinct incoming participants, and post
volume. Roles default to unreviewed. No account is called an official affiliate
or authenticated influencer merely because it appears in search or a bio.

## Collection allocations

All calls atomically reserve against the existing `zcat-zec-v1` ledger. Outstanding
or uncertain calls stop all endpoints. No automatic retries or reservation refunds.

| Allocation | Maximum credits | Behavior |
| --- | ---: | --- |
| Original coin searches | 16,800 | Includes existing pilot usage; first-page coverage is prioritized |
| Profile snapshots | 21,600 | Up to 40 unique IDs, once per UTC day, 18 credits each |
| Activity enrichment | 9,000 | One 300-credit page per day, up to 30 date slots |
| Optional keyword discovery | 2,600 | Requires verified provider page-size bound; otherwise retained |
| Total | 50,000 | Existing reservations included; never resets on rerun |

Activity slots rotate: five of each seven slots search one coin over the previous
UTC day (alternating coins), one samples an account timeline with replies, and
one attempts a refresh of up to 20 saved posts aged 24–30 hours. A refresh with
no eligible posts costs nothing. This is deliberately sparse: it is not daily
complete coverage of either coin or every account timeline. Initial searches
and these later samples are not equally intensive. Missing collection days
must not be treated as zero volume. Timeline samples are also retained but do
not count as complete coin-search coverage.

Sources checked September 14, 2026:
- https://twitterapi.io/pricing (15 credits/post; 18/profile; 100,000 credits/USD)
- https://docs.twitterapi.io/api-reference/endpoint/batch_get_user_by_userids
- https://docs.twitterapi.io/api-reference/endpoint/get_user_last_tweets (20/page)
- https://docs.twitterapi.io/api-reference/endpoint/get_tweet_by_ids
- https://docs.twitterapi.io/api-reference/endpoint/search_user

## Bio discovery and changes

All returned author/profile descriptions are scanned without additional provider
calls. Matching is case-insensitive with Unicode normalization and word boundaries.
ZCAT/Anonymous Cat and Zcash/ZEC accept hashtags or cashtags; the supplied token
address is matched exactly. Bio, name, and handle matches are stored separately.
The dashboard bio board includes only actual bio matches in the latest observation.

Profile history stores stable ID, handle, name, bio, follower/following count,
availability, source endpoint/request, and observation time. Empty bio differs
from missing bio. Renames keep the same account history. Missing/unavailable
profiles have unknown counts, never fabricated zeros. The bio-change view compares
known bios and says when the change was first observed, not when it was edited.
Older known bios remain in history even if the current profile is unavailable.

The keyword user-search adapter is implemented but **not automatically enabled**.
The public docs do not promise a maximum page size or bio-only/complete matching.
Once the provider confirms the maximum returned profiles per call, an operator
can run `python crypto_social_tracking.py --execute --mode discover
--verified-user-search-max N`, replacing N with that verified bound (1–144).
This reserves N×18 per query, inside the existing 2,600-credit allocation. It
queries zcat, zcash, and Anonymous Cat at most once per ISO week, first page only.
Do not invent a page-size bound to activate this path. A changed response contract
halts collection; provider-side billing ultimately depends on their contract.

New discoveries enter the candidate queue. Existing tracked members are retained;
we do not silently replace the baseline cohort with whichever accounts are popular
this week. Initial selection fills only vacant cohort slots, up to 20 per coin.

## Metrics and interpretation

The dashboard offers six separate views:
- Attention: distinct incoming participants across all saved coin edges in 28 days.
- Follower growth: positive net seven-day gains; percentage shown alongside gain.
- Emerging: positive follower growth and more distinct incoming participants this
  week than the previous week. Provisional because collection coverage can differ.
- Posting: observed coin-related posts, separate from overall account timelines.
- Connections: distinct incoming/outgoing partners; not a community-bridge claim.
- Bio matches: observed self-description terms; not evidence of holdings/endorsement.

Growth requires a real snapshot on the corresponding UTC baseline date. Missing
baseline means no growth result. Acceleration compares per-day net gain over two
seven-calendar-day spans using their actual elapsed seconds; 14-day history is
required. No historical counts are backdated from newly retrieved profiles.

Account detail provides bio, profile time, discovery reason, observed posts/active
days, median total engagement, eligible engagement sample count, separate 24–30h
engagement median/count, repeat participants (two or more interaction dates),
reciprocal partners, and top-five participant concentration. Concentration caps
one source/target pair at one contribution per day before summing; raw evidence
is preserved. Fewer than 10 eligible engagement posts or five active days is
explicitly provisional. Age-matched metrics remain empty without qualifying
snapshots. We do not reconstruct historical 24h engagement or extrapolate total
posting volume from one page. No opaque composite score or bot classification.

Network calculations use the full saved coin graph, before the 20-node/60-edge
visual limit. Typed edges point actor → target and retain supporting post URLs.
Likes remain counts only; identities are not available from those counts.
Interactions do not prove endorsement, coordination, shared control, or price
causation. Community detection/PageRank and price-impact attribution remain
research extensions requiring sufficient coverage, not launch features.

## Database and operational notes

Additive SQL creates profile history, candidate memberships/tracking dates, bio
matches, account timeline coverage, a fixed campaign lifetime, and a read-only
metrics view. Endpoint metadata and idempotency keys extend the original request
ledger. Existing content remains; no production data is deleted by migration.
An older deployment without the new view continues using the original panel.
All displayed new metrics are computed from Neon; browser refreshes cost no
TwitterAPI.io credits. The original pilot endpoint displays a shared budget;
its legacy estimated total includes all requests because the ledger is shared.

Tests use fake provider responses and a disposable Postgres schema only:

```bash
python -m pytest tests/test_crypto_social_pilot.py tests/test_crypto_social_tracking.py -q
cd apps/web
npm run typecheck
node --experimental-strip-types --test lib/crypto-social.test.ts
```

Set `CRYPTO_SOCIAL_TEST_DATABASE_URL` only to a disposable local/test database;
the integration fixture recreates `crypto_test`. The PR's dedicated CI job runs
these tests against PostgreSQL 16. No tests require a live provider key.

## Price linkage and event study (2026-09-17)

`crypto_market_history.py` now archives every tracked coin, not only ZCAT/ZEC/PONS: daily
candles for up to five pools per contract and **hourly candles for the pinned default pool
only** (`crypto_market_hourly`, 1,000-candle pages, roughly 41 days per fetch), plus hourly
CoinGecko observations for ZEC. Source IDs are unchanged, so existing pins and archives
carry over. A contract with no indexed pool is reported under `skipped`, never as a failure.

`crypto_event_study.py` (runs after the market archive in the same 6-hourly workflow; no
provider calls) writes `crypto_price_events`: for each saved non-repost post mentioning the
coin (same text rules as the dashboard's "Coin words in post" filter), the pinned pool's
hourly close one hour before, at, and 1/6/24 hours after the post hour, plus 24-hour volume
sums either side. A row is written only once the 24-hour candle is complete and both
endpoints exist, and it is never rewritten (`version = price-events-v1`; a rule change is a
new version, not an update). `episode` marks an author's first eligible post on a coin in
24 hours so a burst during one move counts once. The weekly voice evaluation now also records
each group's episode count and median 24h forward move.

`GET /api/market/crypto/impact?coin=ALL|<coin>` aggregates episodes per account (median
+1h/+6h/+24h, median excess over the coin's own median 24h move across every archived hour,
share up, share beating drift, 24h-after over 24h-before volume ratio, best/worst post) and
returns the per-coin baselines. Accounts → "Price after posting" renders it with a
minimum-episodes control (default 3). All of it is association: a post can follow a move,
react to news, or be one of hundreds in the same hour, and pool prices are one pool.

## Coin registry (2026-09-17)

Tracked-coin identity lives in one file, `apps/web/lib/crypto-coins.json`: symbol, name,
network, contract address, archive start date, name/cashtag patterns, context pairs,
name-collision exclusions, discovery name, profile terms, official handles and the
provider search query. `apps/web/lib/crypto-coins.ts` (`matchesCoin`, `isCoin`,
`coinConfig`) and `crypto_coins.py` (`mentions`, `markets`, `archive_start`) read it;
`tests/fixtures/crypto-coin-matches.json` pins both to the same answers. Consumers:
`filterEvidence`, voice roles, coin discovery, every `/api/market/crypto/*` allowlist and
archive start, the research coin picker, the market archive, the event study, profile bio
matching and the rolling collector's names/queries. Adding a coin is one JSON entry plus a
pinned pool appearing on GeckoTerminal; the SQL coin CHECK was dropped for that reason.

Do not edit `searchQuery` for a coin with saved windows: window rows store the query text
verbatim and pagination cursors belong to it (`tests/test_crypto_coins.py` pins the live
ones). The dated, bounded campaign scripts (`crypto_social_pilot/history/catchup/dpons`)
keep their own literals on purpose; they describe finished or fixed-scope collections.

## Collection-time ranking snapshots (2026-09-17)

`crypto_rankings_cache.py` runs at the end of the rolling (voices) and watcher (watchers)
collection workflows. It loads every saved row with the same SQL the routes use
(`apps/web/lib/server/crypto-ranking-queries.json`), ranks it with the dashboard's own
TypeScript through `scripts/crypto-rankings.mts`, and upserts one payload per
(key, version) into `crypto_ranking_cache` (`voices:<coin>`, `watchers:ALL|<coin>`). The
voices and watchers routes serve that snapshot when present (`source: "snapshot"`, with
its computed time shown in the UI) and otherwise rank live over the old bounded load, so a
deploy never depends on a collector having run, and a ranking code change simply produces
a new version key. Because the builder has no row limit, the earliest-50,000 truncation
that was quietly dropping the newest posts from voice roles no longer applies once a
snapshot exists. Every read route now also sets `Cache-Control: s-maxage=300` so repeat
page loads within five minutes are served by the CDN instead of Neon; the run route still
ships the raw post set the explorer needs, which is the remaining large read.

## Front door and URL state (2026-09-17)

`/market/crypto` is the research workspace's own route (in the app nav as "Crypto
Research"; the Market → Crypto tab keeps prices and links across). It opens on an
**Overview** view (`GET /api/market/crypto/overview`): one card per tracked coin with the
pinned pool's 24h price change, saved posts and distinct accounts over 24h/7d, first-time
voices this week and price-linked episodes, plus a table of untracked coins named by the
watcher accounts in the last seven days (from the coin-discovery extractor). Coin, view,
selected account and date range live in the query string (`?coin=&view=&account=&from=&to=`),
written with `history.replaceState`, so any state is shareable.

## One account, every coin (2026-09-17)

`GET /api/market/crypto/account?id=<x id>` returns an account's profile, per-coin saved
activity, its role per coin from the voice snapshots, its price-linked episodes per coin from
the event study, its watcher category, and its latest posts with the pool's 24h move where
linked. The account drawer now shows this "Across tracked coins" section and opens for any
account id (from the Overview, a leaderboard, or a `?account=` link), not only accounts with
posts in the current selection. `GET /api/market/crypto/leaders` ranks accounts by how many
tracked coins they were early on, then coin count, then episode-weighted median excess 24h
move (`apps/web/lib/crypto-leaders.ts`); the Overview shows the top 15 as "Accounts to
watch". Both build only from snapshots and the event study, so they read nothing live.

## Chart as navigation (2026-09-17)

On Insights → Price & timing, tapping a day on the price/attention chart (or the "Largest
daily price gain" card) opens **Before the move**: every account that posted in the 24 hours
before that UTC day, one row per account, largest saved audience first, with the day's daily
close change and the count of posts during the day itself. "Mark on chart" draws that
account's posts as pink marks above the price line so timing can be read against price;
the account link opens the cross-coin drawer. Dragging still selects a range. All of it is
computed from the posts already loaded for the coin; nothing new is fetched.

## BACKPACK, PERSPAD and OMFG added (2026-09-17)

Two user-supplied Solana contracts joined the registry: BACKPACK
(`BPxxfRCXkUVhig4HS1Lh7kZqV6SPJhzfEk4x6fVBjPCy`) and PERSPAD
(`PerPsCe2SJ7Q25CN4R5TTX4fmBdmknE2hQmqCt96fHL`), archive start 2026-09-01. "Backpack" is an
ordinary word, so its words-mode match needs the cashtag/hashtag or token context
(`token|coin|solana|sol|pump|ca` in the same post); the contract always matches. Adding them
enrolled both in the rolling collector on its next run (a separate 30,000-credit ceiling
each under the existing campaign) and in the market archive, which finds their pools on
GeckoTerminal. Neither has official handles configured yet. Tests that counted coins now
derive counts from the registry.

OMFG (`omfgRBnxHsNJh6YeGbGAmWenNkenzsXyBXm3WDhmeta`) was added the same way; the ticker is
inferred from the address's vanity prefix and not verified. Bare "omfg" is slang, so words
mode needs `$OMFG`/`#OMFG`, the contract, or token context in the same post.

## Origin search for newly added coins (2026-09-17)

KNOTS (`8RVBk8vxLiUHueLUW1f4izFVqN3nWippLhkohKg6EGkS`) and STONK
(`6GmAFSYs4gk3FDao5FzzySQpPZaWsa4rUJHacpMpUNgx`) joined the registry; both are common words,
so bare mentions need the cashtag or token context. A registry coin may now carry
`originFrom`: on the rolling collector's next run, `setup_origin` creates one contract-only
search window from that date to the coin's first live window (recorded in
`crypto_origin_windows`, excluded from forward gap-filling). Lane four of each run prefers
focus and origin windows, so the backfill paginates until exhausted within the coin's
ordinary 30,000-credit ceiling. Once an earliest contract post is saved, the existing
30-hour focus windows around it are created automatically. All five coins added today carry
`originFrom: 2026-06-01`; the established coins had dedicated history campaigns and do not.

## Workspace redesign (2026-09-17)

`/market/crypto` now runs on `crypto-workspace.tsx`: five destinations organised around the
questions a researcher asks, replacing the eleven tab/sub-tab views.

- **Signals** (`GET /api/market/crypto/signals?hours=24`): a coin board sorted by 24h pool
  volume ratio (price, volume, posts and distinct accounts vs the prior window, watched
  accounts posting) and a feed of named-rule signals evaluated by `lib/crypto-signals.ts`:
  watched account posts the contract; volume ≥2× with ≥30 posts; first saved contract post
  on a coin; attention without price (≥20 posts, |price| <2%, top three authors ≥60%); a
  watched account reports a trade or volume figure. "Watched" = the ten reviewed watcher
  accounts plus every account with a track record (early on a coin, or ≥3 episodes). Below:
  top five track records and untracked coins named by watchers in the last seven days.
- **People** (`GET /api/market/crypto/leaders?all=1`): one ranked list whose columns are the
  questions — Who · Where early · After they post · How they post · Evidence. *Where early*
  shows coin chips with the calendar day of the account's first post counted from the coin's
  earliest saved contract post (`CoinRole.day`, computed in `loadRoles` from the voices
  snapshot's `anchor`; day 1 = same day). *After they post* is plain text (`up 7 of 9 times ·
  typically +14% vs the coin's own drift a day later`) and appears only with
  `MIN_LINKED_POSTS` (3) episodes; thinner rows say so and are dimmed. *How they post* is a set
  of icon chips from `postingStyles()` (first to share the contract / among the first to post /
  explains rather than hypes / mostly quotes others / alert feed / project account / reports
  whale trades). *Evidence* is counts (`21 price-linked · 4 coins · 12 days`, `days` = distinct
  posting days from `crypto_social_posts`) with the one-line `whyLeader` under it. Ranking
  (`rankLeaders`): early-coin breadth, then any account with a sample beats one without, then
  hit rate, then median excess move. The former Largest audiences, Voice roles, Price after
  posting and Watcher rankings views are sorts and filters here.
- **Coins** (`?view=coins&coin=X&tab=timeline|posts|sentiment|connections`): opens on the
  timeline with four stat cards (largest gain, accounts before it, earliest contract post,
  coverage), the price/attention chart, and Before the move as the drill-down (`day=`,
  `highlight=` in the URL). All posts, Sentiment and Connections are sub-views of the coin.
- **Account** (`?view=account&account=<id>`): a full page backed by `/api/market/crypto/account`.
- **Data** (`GET /api/market/crypto/status`): coverage per coin (days searched, origin search
  progress, hourly candles), every credit ledger, and the registry with official handles.

The header search (`GET /api/market/crypto/search?q=`) resolves coins by symbol, name or
contract and accounts by handle; an untracked contract address offers the registry. Legacy
URLs (`view=impact`, `view=network`, `account=` without a view) map onto the new destinations
in `lib/crypto-workspace.ts`. Removed components: the social panel, influencer panel, large
accounts, voices, watchers, watcher feed, coin discovery, price impact, overview and the
account drawer; their libraries (`crypto-voices.ts`, `crypto-watchers.ts`,
`crypto-coin-discovery.ts`, `crypto-run-reach.ts`) remain because the snapshot builder and
routes use them. Weekly cohort validation tables are no longer rendered; the data is still
written and served by the voices route.

## Real routes and a left rail (2026-09-17, second pass)

The five destinations are now paths under `apps/web/app/market/crypto/`: `/market/crypto`
(Signals), `/people`, `/coins/<SYMBOL>`, `/accounts/<id>` and `/data`, sharing one layout
(`crypto-shell.tsx`) with a left rail (icon, name, four-word purpose; a bottom bar on phones),
the header search, and a `PageHeader` (eyebrow, question, one sentence) used identically on
every page. Child pages carry breadcrumbs (People › @handle; Signals › coin) and the coin page
shows every tracked coin as chips. The Coin page is one scrolling page: stats, chart with
Before the move, the posts list (sentiment under its Insights disclosure), and connections
in a collapsed section; there are no sub-tabs. Only the coin page keeps query state
(`from`, `to`, `day`, `highlight`). Old `?view=` links redirect through `legacyPath` in
`lib/crypto-workspace.ts`. The app nav item is "Crypto"; the Market → Crypto tab keeps
prices and a one-line link.
