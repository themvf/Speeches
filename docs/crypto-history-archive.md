# Historical research and market archive

## Historical ZCAT collection

The existing July 25–September 13, 2026 UTC campaign retains its non-resetting
150,000-credit reservation ceiling. The separate live pilot keeps its own limit.
The September 14 authorization raises the total ceiling, not an additional allowance.
The database migration preserves all prior reservations, saved posts and cursors.
Each execution now permits 80 searches at most: 24,000 reserved credits using
the existing 300-credit worst-case reservation per page. Failed/uncertain calls
retain reservations and block further paid calls. Re-running never resets spend.

Priorities come from the pinned ZCAT pool's archived, completed daily candles.
The first pair of consecutive observed days with a gain of at least 50% anchors
a focus period: five preceding days, the anchor day, and two following days.
This is a collection heuristic, not a validated breakout or causal explanation.
Missing days are never bridged. Each batch saves its period, reason, market source,
and anchor fetch ID so the priority decision can be reconstructed.

Unsearched windows are covered first, with the focus period ahead of other gaps
chronologically from July 25. After the first pass, additional pages are collected
in rounds with the same focus priority. Completed searches are skipped; no window
receives more than eight pages. If the
focus is exhausted, slots fall back to broader coverage. Without qualifying saved
prices the batch continues chronological gap filling. Partial windows at the page
limit remain marked partial, not complete. This budget cannot guarantee exhaustive
coverage of every period.

The existing history workflow runs daily at 08:43 UTC and on relevant main pushes.
It saves market data before choosing priorities. A market-provider failure allows
use of previously saved prices; an unavailable database stops collection.

## Persistent market history

`crypto_market_history.py --execute` saves ZCAT's five currently most liquid returned
verified pools and ZEC observations from public endpoints. Once chosen, the default
ZCAT pool remains pinned; it is also refreshed if it later falls outside the top five.
The source registry, immutable fetch records, raw responses and per-fetch daily
observations live in Postgres. A latest-observation view selects the most recently
retrieved revision for each source/day while retaining all older revisions and days
omitted by later responses. No pool histories are joined into a synthetic series.

Each fetch records provider URL, retrieval time and the source metadata at that time.
ZCAT metadata includes the exact contract and its base/quote orientation. ZCAT records
store OHLCV. ZEC records store timed price observations and rolling 24-hour volume,
with null OHLC fields; these are not exchange candles. Current-day records are marked
incomplete and only a later fetch can supply a completed-day observation. Incomplete
prices are excluded from largest-gain insights and priority selection.

The archive workflow runs every six hours (02:23, 08:23, 14:23, 20:23 UTC), on relevant
main pushes and on manual dispatch. It requests at most eight public market resources per
run, spaced by three seconds. A 429 response permits one bounded retry per resource
(up to 16 attempts total), honoring numeric Retry-After values up to 30 seconds. Longer cooldowns stop
that resource until a future run.
There are no X requests or unbounded retries. Independent source failures retain
successful saves and previous history; the workflow reports a failure for inspection.
The database advisory lock prevents overlapping archive execution outside Actions.

`/api/market/crypto/history` only reads Postgres. Page visits do not fetch providers or
write data. The UI shows last-save time and flags archives over 12 hours old. New
sources or lost external history cannot fill days before available observations.

## Validation

CI runs real disposable Postgres tests for append-only revisions, retention of omitted
days, exact pool identity, priority ordering, broad-coverage slots, page caps, campaign
budget preservation and reference capture. Parsing tests cover malformed prices,
partial days and timestamp semantics. TypeScript tests exclude incomplete candles
from daily-gain insights. No tests call paid endpoints.

Provider references:
- https://apiguide.geckoterminal.com/faq
- https://docs.coingecko.com/reference/coins-id-market-chart
- https://docs.twitterapi.io/api-reference/endpoint/tweet_advanced_search

## September 14 catch-up

`crypto_social_catchup.py --execute` collects only ZCAT on September 14, 2026 UTC,
through the earlier of execution time or September 15 midnight. It uses the
existing live pilot's 50,000-credit total and 16,800-credit search sublimit; the
fully reserved 150,000-credit historical campaign is unchanged. At most 40 calls
and eight pages per window are allowed, reserving at most 12,000 credits per run.
Reruns append intervals after the previous cutoff and resume partial windows.
A same-day cutoff does not represent a completed day or exhaustive search.
The dedicated workflow runs on relevant main pushes or manual dispatch.
The timeline API now returns up to 10,000 saved posts, enough to include the
existing archive and this bounded catch-up; its total still reports all stored matches.

## PONS investigation

PONS has a separate 150,000-credit campaign, `pons-july-2026`, authorized September
14. The intended range is July 1 through September 14 inclusive (UTC). The collector
only creates closed six-hour intervals; the final interval becomes available after
September 14 ends. A complete first pass reserves at most 91,200 credits (304 pages).
Each daily batch allows 80 calls, with eight pages per window and existing uncertainty
stops. Campaign-specific ledger queries prevent PONS and ZCAT totals from mixing.

The PONS workflow saves public pool history before collecting posts, including exact
Robinhood Chain contract checks and support for Uniswap v4 pool identifiers. One
pool is pinned and its raw responses are archived; absent price dates stay blank.
Searches use the contract, cashtag, PONS with Robinhood context, and the listed project
account ponsdotfamily. Bio/name matches from returned profiles are unreviewed leads.
Search-author snapshots supply a present-day baseline, not historical follower counts.
The initial campaign does not automatically enroll PONS accounts in the separately
budgeted daily profile tracker.
