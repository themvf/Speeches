# Time-of-day profile for tracked coins

`crypto_hour_profile.py` answers one question about the coins in
`apps/web/lib/crypto-coins.json`: **does the hour of day predict anything?** It reports three
separate profiles — return, trading activity, and volatility — because they do not behave the
same way, and conflating them is the easiest mistake to make here.

```
python crypto_hour_profile.py --days 14            # reads the hourly archive
python crypto_hour_profile.py --days 14 --live     # re-fetches from the providers instead
python crypto_hour_profile.py --days 7 --coins ZCAT,ZEC
python crypto_hour_profile.py --out report.json --summary   # JSON to a file, verdict to stderr
```

**It runs itself.** `crypto-hour-profile.yml` recomputes this from the archive every Monday
(no provider calls, no credits), writes the plain-language verdict to the run summary, and commits
`apps/web/lib/server/crypto-hour-profile.json` when it changes — so a change in the answer arrives
as a diff rather than waiting to be asked for. Run it by hand with `workflow_dispatch` to pick a
different window.

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

**Three nulls, and the strictest one is the headline.** Rotating each coin-day independently treats
a coin's days as separate evidence for a shared shape, which is anti-conservative whenever a coin's
daily profiles are correlated for reasons unrelated to the clock. So `p_global` carries all three:
`day` (each coin-day rotated independently), `coin` (one offset per coin, all its days moved
together, so a coin is one vote however long its history), and `market` (one offset per calendar day
applied to every coin, so a single market-wide move cannot be counted once per coin). Read `coin`
as the headline and treat a result that only survives `day` as unproven.

**A rolling 24-hour volume is not hourly volume.** CoinGecko's `market_chart` reports
`total_volumes` as a rolling 24h total: ZEC's series drifts about 2% an hour where a real hourly
series swings 47-56%. Any `price_observation` source is therefore excluded from the volume profile
and named in `volume_excluded`, because averaging a smoothed window against real hourly series
would pull the profile toward uniform. Its *returns* are still analysed — only its volume is unusable.

**A global p-value does not mean the shape will hold next week.** `split_half` splits each coin's
days chronologically, measures the 24-hour profile in each half, and correlates them against a
market-wide rotation null. A profile can be significantly non-flat in-sample and still be a
different shape a week later; this is the only statistic here that distinguishes the two.

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

**The window is whole UTC days, and today is excluded.** A rotation unit is a day, so a part-day at
either edge is a near-degenerate unit: a 3-hour edge day can only rotate three ways and drags the
null around. It also made the answer depend on the clock time the job ran — rolling the window nine
hours moved volatility's coin-null p from 0.030 to 0.104 while the hour profile was identical.
Anchoring to whole days makes a scheduled run reproducible whenever it fires, and two runs of the
same window are now byte-identical.

**The open candle is not an hour.** The candle covering the current hour holds only the minutes
elapsed so far — 36 of 60 when this was last run. Counting it as a full-hour return would drop a
systematically short, systematically quiet bar into whichever hour-of-day the run happens to start
in, and a day-rotation null cannot move a trailing singleton day, so it would be pinned there in
every permutation. Incomplete candles are excluded from both returns and volume days.

**Report what the window can resolve.** Per coin the output carries `mde_pct` and
`mde_pct_corrected`: the minimum recurring same-hour move that window could detect at 80% power,
before and after splitting alpha 24 ways. The multipliers are `z(1-a/2) + z(0.80)` — 2.802
uncorrected and 3.920 corrected — over the *median observations an hour actually has*, not the day
count, which would flatter a coin with partial coverage. Over 7 days:

| Coin | hourly SD | obs/hour | MDE | MDE, 24-hour corrected |
| --- | ---: | ---: | ---: | ---: |
| ASKR | 25.0% | 2 | 64% | 100% |
| AD | 21.6% | 3 | 42% | 63% |
| FLX | 16.6% | 4 | 26% | 39% |
| DPONS | 14.4% | 7 | 17% | 24% |
| KNOTS | 9.6% | 7 | 11% | 15% |
| ZCAT | 6.3% | 7 | 7% | 10% |
| STONK | 3.8% | 7 | 4% | 6% |
| PONS | 2.8% | 7 | 3% | 4% |
| ZEC | 1.4% | 7 | 1.5% | 2.1% |

That table is why "no time-of-day effect found" is mostly a statement about the window. A real
+5%-every-day-at-3pm pattern in ASKR would be invisible here; the same pattern in ZEC would be
obvious. Any null result has to be read against the coin's own row.

## What the data said on 2026-09-21 (13 coins, live fetch)

Global p-values under each null, ZEC excluded from the volume profile:

| Metric (14-day, whole UTC days) | day | coin | market |
| --- | --- | --- | --- |
| Return by hour | 0.56 | **0.38** | 0.83 |
| Volume share by hour | 0.0050 | **0.067** | 0.027 |
| Volatility by hour | 0.028 | **0.070** | 0.035 |

Split-half persistence over the same 14 days (profile measured in the first half, correlated with
the second, against a market-wide rotation null):

| Metric | r | p |
| --- | --- | --- |
| Return by hour | **-0.181** | 0.80 |
| Volume share by hour | +0.469 | 0.043 |
| Volatility by hour | +0.473 | 0.049 |

Run over each coin's full history instead of 14 days, the same split reaches r = +0.83 (volume) and
r = +0.57 (volatility) while returns stay at r = -0.01. The activity shape is the same shape a week
later; the return shape is not a shape at all.

**The two tables disagree, and that is the honest reading.** Activity and volatility clear the day
and market nulls but sit just above 0.05 under the coin null, which is what twelve coins buys when
each contributes a single vote — while their shape does come back in a held-out half. Returns clear
nothing under any null and have *negative* persistence. So: the activity shape is real and recurs,
but with twelve coins this window cannot prove it against the strictest null; the return shape is
not a shape at all. Report both numbers rather than picking the flattering one.

An earlier draft of this document reported volatility at coin-null p = 0.030 and called it
significant under all three nulls. That came from a window cut at the hour the job happened to run,
which included a part-day at each edge. Anchoring to whole days moved it to 0.070 while the profile
itself was unchanged (r = 1.000 between the two). The shape was never the unstable part.

**Volume and volatility are not two independent confirmations.** On an AMM the size of an hourly
move is mechanically driven by swap flow, so the volatility profile is largely the activity profile
restated. Conditioning volatility on within-coin volume quintiles leaves 09:00 UTC standing but not
10:00 or 11:00. Treat this as one finding about when people trade, not two.

Direction has no detectable time-of-day structure; **when people trade does.** Activity troughs at
02:00–07:00 ET (06:00–11:00 UTC), bottoming near 03:00 ET at 0.65x an even hour, and peaks at
13:00–14:00 ET (1.35x, 1.28x) and again 21:00–23:00 ET. Volatility tracks it, calmest at 05:00 ET
(0.71x of the coin's own average hour).

The lull is a sleep effect, not a work-schedule effect: over 14 days the 01:00–07:00 ET window
takes 21.8% of weekday volume and 21.2% of weekend volume against a 29.2% even share (t = -5.71
and -4.41). It is also not one chain's quirk — relative volatility across 05:00–11:00 UTC is 0.89
on the Solana coins, 0.88 on the Robinhood coins and 0.81 on ZEC, each significant on its own.

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
  OHLC and different microstructure from the DEX pools, and its volume column is excluded here.
- **Upstream bug, not fixed by this tool:** `crypto_market_history.py` stores CoinGecko's rolling
  24h `total_volumes` into `crypto_market_hourly.volume` for ZEC. `crypto_event_study.py` then sums
  24 of those overlapping totals into `volume_before_24h` / `volume_after_24h`, which
  `/api/market/crypto/impact` divides into its volume-ratio column. For ZEC that ratio is computed
  from a smoothed series and is damped toward 1. Contract coins are unaffected — their candles carry
  real per-hour volume.
- ZCAT's pinned pool is `ZEC / ZCAT` and KNOTS' is `STONK / KNOTS`, so three of the tracked series
  are priced through two others. Their USD returns inherit some of the quote token's movement.
- The 14-day window contains the 7-day window, so agreement between them is not an out-of-sample
  check. Treat it as a stability check only; `split_half` is the actual holdout.
- Day of week cannot be answered here and the tool does not try. Seven days puts exactly one date in
  each weekday bucket, so a label-rotation null returns p = 1.000 for any dataset whatsoever; at
  fourteen days the exact p floor is 1/7 = 0.143. That needs months.
- GeckoTerminal sets each candle's `open` to the prior `close`, so a gap-versus-session
  decomposition of the hourly move is not available from this feed.
- Coverage is ragged. ASKR, AD, FLX and STANDARD were only indexed within the last week, so their
  per-hour counts are 2–6 observations. `MIN_HOUR_OBS` keeps those buckets out of the test, so they
  appear in the table and never in a p-value.
- Activity has an obvious mechanism (these are US-retail-driven microcaps, so the lull is overnight
  in the US). Direction does not, which is one more reason to treat a directional hour as noise
  until a pre-specified window confirms it out of sample.
