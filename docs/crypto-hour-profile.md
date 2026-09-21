# Time-of-day profile for tracked coins

`crypto_hour_profile.py` answers one question about the coins in
`apps/web/lib/crypto-coins.json`: **does the hour of day predict anything?** It reports three
separate profiles — return, trading activity, and volatility — because they do not behave the
same way, and conflating them is the easiest mistake to make here.

```
python crypto_hour_profile.py --days 14            # reads the hourly archive
python crypto_hour_profile.py --days 14 --live     # re-fetches from the providers instead
python crypto_hour_profile.py --days 7 --coins ZCAT,ZEC
```

With `DATABASE_URL` set it reads `crypto_market_hourly_latest` for each coin's pinned default
source. Without it, `--live` re-derives the same candles from GeckoTerminal and CoinGecko using
`crypto_market_history`'s own pool-pin rule, so the analysis reproduces with no database access.

## The rules this analysis is built on

**A per-hour t-statistic is not a finding.** There are 24 hours, so the most extreme one is
expected to look impressive even when nothing is going on. Every p-value reported here is a
*global* day-rotation permutation test whose statistic is `max|t|` across all 24 hours, so it
already prices in the fact that all 24 were examined. Rotating each calendar day's values by a
random offset destroys hour-of-day alignment while preserving the value distribution and the
within-day serial structure exactly. The per-hour `t` column in the output is there to show
*where* the structure sits once the global test says there is structure — never as evidence on
its own.

**An absent candle means no trades, not missing data.** GeckoTerminal is queried with
`include_empty_intervals=false`, so a no-trade hour is an absent row. Volume profiles count those
hours as genuine zeros, because a dead hour is the signal. Returns take the opposite treatment and
are never bridged across an absent hour, since a move spanning a gap did not happen "at" either hour.

**One test per metric; the ET column is a relabel.** Shifting every hour label by a constant cannot
change the set of per-hour t-statistics, so a separate "ET p-value" would be the same test reported
twice. Days are bucketed on their UTC date. Eastern hours are computed from the offset actually in
effect, so the labels stay right across the DST change.

**An hour that cannot be tested is named, not dropped.** `untestable_hours` lists hours whose
t-statistic is undefined, with the reason: fewer than `MIN_HOUR_OBS` observations, or every
observation identical. A permanently dead hour has exactly the second shape, and an hour silently
missing from the statistic must never read as an hour that was tested and cleared. When *no* hour
is testable the report says "too short to test" and withholds the p-value rather than printing
`p = 1.0`, which would read as a tested null.

**Three observations is not a sample.** `MIN_HOUR_OBS = 5` exists because of a real artifact: in a
3-day window ZEC's 22:00 UTC bucket held `[-0.62%, -0.59%, -0.59%]`. Three coincidentally-similar
values give a near-zero standard error, so that trivial -0.6% mean produced `t = -56` and a global
`p = 0.0055`. A window shorter than five days cannot support a per-coin hourly test at all, and the
tool now says so instead of reporting one.

**Report what the window can resolve.** Per coin the output carries `mde_pct` and
`mde_pct_corrected`: the minimum recurring same-hour move that window could detect at 80% power,
before and after a 24-hour multiplicity penalty. For a coin with 20%+ hourly volatility and a few
days of history this runs to tens of percent, which is the honest reason a "no effect found" result
is usually a statement about the window rather than about the coin.

## What the data said on 2026-09-21 (13 coins, live fetch)

| Metric | 3-day | 5-day | 7-day | 14-day |
| --- | --- | --- | --- | --- |
| Return by hour | p = 0.29 | p = 0.19 | **p = 0.11** | **p = 0.49** |
| Volume share by hour | p = 0.16 | p = 0.060 | **p = 0.010** | **p = 0.0043** |
| Volatility by hour | p = 0.80 | p = 0.015 | **p = 0.037** | **p = 0.011** |

The shape of that table is itself the evidence. Activity and volatility get *more* significant as
the window lengthens, which is what a real effect does when you give it more data. Returns wander
with no trend, which is what noise does.

Direction has no detectable time-of-day structure; **when people trade does.** Activity troughs at
01:00–07:00 ET (05:00–11:00 UTC), bottoming near 03:00 ET at about 0.69x an even hour, and peaks
11:00–17:00 ET at roughly 1.25–1.31x. Volatility tracks activity, calmest at 05:00 ET (0.71x).

No individual coin showed a significant return pattern in either window (smallest of 24 per-coin
tests: ZEC, p = 0.085). The one pattern that looked compelling — 09:00 ET down across 10 of 11
coins in the 7-day window, pooled t = -2.90 — does not survive: the global test that accounts for
scanning 24 hours puts it at p = 0.29, and it is absent from the 14-day window entirely. Cross-coin
correlation is low (mean pairwise r = +0.02, about 9 effectively independent series), so that was
not one move counted eleven times; it was a genuine coincidence of the kind 24 hours of searching
produces.

## Caveats that matter

- The pinned pool is one pool per coin. On a thin AMM a single swap can set an hourly close.
- ZEC comes from CoinGecko `market_chart`, which is sampled spot price, not traded OHLCV. It has no
  OHLC and different microstructure from the DEX pools.
- Coverage is ragged. ASKR, AD, FLX and STANDARD were only indexed within the last week, so their
  per-hour counts are 2–6 observations and their tables are illustrative at best.
- Activity has an obvious mechanism (these are US-retail-driven microcaps, so the lull is overnight
  in the US). Direction does not, which is one more reason to treat a directional hour as noise
  until a pre-specified window confirms it out of sample.
