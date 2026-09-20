
---

# Built 2026-09-19 — V1 (chain adapter + Solana + opening trade capture)

`launchpad_chains.py` (new), `launchpad_archive.py`, `sql/launchpad_archive.sql`,
`.github/workflows/solana-archive.yml`, `tests/test_launchpad_chains.py`.

**Step 1, the refactor, changed no Robinhood behaviour**: chain identity moved into a `Chain`
adapter, and the existing 18 tests passed unmodified against it. Module-level `NETWORK`,
`CURVE_DEXES`, `GRADUATE_DEXES` and `RUNGS` remain as aliases for the default chain so nothing that
already read them had to change. The advisory lock is now per chain, and the two chains have
separate workflows and concurrency groups, so a Solana failure cannot stop the archive that is
already collecting.

**The bug only a live run could find.** The first live Solana sweep failed on *every* enrichment
call. Cause: `parse_pool` folded addresses to lower case — correct for EVM hex, which is
case-insensitive, and destructive for Solana's **case-sensitive base58**. Every `/info`, `/pools`
and `/trades` lookup was made against an address that resolves to nothing. Mocked tests would never
have caught it, because the mock answers whatever address it is asked about. Address folding is now
a chain property (`lowercase_addresses`), pinned by a test.

**Opening trade capture** fires when a graduation is first detected: it reads the measurement pool's
trades, dedupes across pages on the transaction itself (pages overlap rather than extending
backwards), stores every wallet-level row with its sequence, and writes a summary beside them —
trades, distinct wallets, buyers, sellers, top-wallet share, repeat wallets. Crucially it records
`capture_started_at`, `earliest_trade_at`, `latest_trade_at`, `window_seconds` and `lag_seconds`
(earliest trade minus graduation), so **how much of the opening window we actually caught is a
measurement, not an assumption**. It is never described as "the first N trades".

**Cohort sampling** is a SHA-256 draw on the mint address: stable across re-runs, and incapable of
correlating with liquidity, buyers or anything else that might later be treated as an outcome.
Recorded per row in `cohort_sampled`, so the sampling fraction is a property of the data.

**`graduation_pool` now holds the launchpad's declared destination** even when the graduation was
detected from a pool arrival — the declared field is metadata, and `measure_pool` is what the ladder
reads. A test caught the two being conflated.

## Not built, deliberately

The X propagation capture. TwitterAPI.io's per-tweet price is genuinely cheap, but the population
is not: **~1,250 graduations/day** means 100 tweets per graduate is ~125,000 tweets/day, roughly
**$19/day (~$560/month)** at $0.15/1,000 — and repeat captures at +10m/+30m/+1h/+3h multiply it,
since the charge is per returned tweet. A sampled sub-cohort with a hard credit ceiling, reusing the
existing reservation ledger in `crypto_social_pilot.py`, is the right shape; it is its own change.

---

# Live-smoke gate for a new chain adapter (required before merge)

Every defect in the Solana build was in a path the mocked suite executes but cannot evaluate: a mock
answers whatever address it is asked, finishes instantly so no deadline expires, and never lets one
phase starve another. Fixtures caught regressions; they caught nothing that was wrong with the
design. So a new chain adapter is not merged until one live run against the real API confirms:

1. **Address format survives end to end** — read back the stored addresses and confirm they match
   the chain's native format. Solana's base58 is case-sensitive; the EVM-safe `.lower()` normaliser
   silently 404'd every enrichment call.
2. **The deadline path actually executed** — a sweep recorded a budget stop in its errors. The
   deadline's own code was broken by a shadowed variable and no fixture run ever entered it.
3. **One real enrichment lookup** — a token has `developer_address` and whatever else the chain's
   `/info` returns.
4. **One real trade capture** — `launchpad_trades` holds wallet-level rows, and the capture's
   `lag_seconds` shows how much of the opening window was reached.
5. **One measurement pool validated by eye** — re-query the chosen pool and agree it is where
   trading happens. This is the only item that cannot be a pass/fail assertion, and it is the one
   that would have caught the $0.43-vs-$30,562 destination-pool trap.

Run 2026-09-19 (three sweeps): 321/321 addresses mixed-case; one sweep recorded a reserve stop;
5 enriched, 4 with a developer address; 3 captures holding 758 trades from 356 wallets with lag
0-220s; chosen pools re-queried at 1,336 / 130 / 29 h1 buyers. Item 5 also produced the archive's
first real finding: one graduate had already fallen to **$4 of liquidity** within the hour, so the
archive is seeing both live and dead graduates rather than survivors only.

# Measured arrival rate, and the enrichment split (2026-09-19)

**Graduations arrive at 3.4/minute (~4,900/day) across all Solana launchpads**, measured over a
15.6-minute span of 53 graduations. Of those, **16 (30%) are Pump.fun lineage** (~1,470/day) and the
rest are Meteora DBC and other launchpads. The earlier "29 per sweep" was a cold-start artifact of
the first sweep seeing a full window; steady state is ~6.8 per 2-minute sweep.

Those graduations are real, not a classification error: spot-checking tokens against GeckoTerminal
returned `completed: true`, `graduation_percentage: 100` and genuine destination pools, one of them
at $36M FDV. But the curve pool is often indexed at or near migration rather than at launch (median
delta 0s for Pump.fun, 52s for Meteora DBC), so **`first_pool_created` is not a launch time on
Solana** and this remains a graduate archive, not a launch census.

At 6.8 arrivals per sweep against a capacity of 8, service rate barely equalled arrival rate and the
backlog only grew — 48 of 53 graduates unmeasured after three sweeps. Enrichment therefore moved to
its own worker (`--enrich`, `solana-enrich.yml`, every 10 minutes, ~60 per run against ~34 arrivals,
so service is about 1.8x arrival):

- **The 2-minute sweep keeps only what expires**: discovery, graduation detection, measurement-pool
  selection for cohort members, and the opening trade capture.
- **The worker owns everything durable**: `/info` enrichment, late pool selection for rows the sweep
  never reached, and the rung ladder. Being slow here costs latency, not data.
- **Pool selection stays at graduation** even though the rest is deferred, because a token can fall
  from a real market to a few dollars within the hour; a pool chosen later may name a different
  market. When the worker has to choose one anyway, it records `selected late, not at graduation`,
  so the two are never mixed in analysis.
- **The queue is the archive itself** — graduates with no `enriched_at`, oldest first. No second
  queueing system.
- **Health is three separate states**, because one boolean blurs different failure modes. See the
  review section at the end.

# Enrichment worker commissioning (2026-09-19)

Two live runs against a real 48-item backlog:

```
pending:          48  →   5  →  0
oldest age:    2074s → 1288s → drained
processed:      43/48 →  5/5
rungs filled:       0 →   15
errors:             0 →    0
service/arrival: 48/53 → 53/53
run time:        370s  → 133s
```

Throughput is ~7/minute (43 in 370s, deadline-bound rather than batch-bound), so a 10-minute run
sustains ~70 against ~34 arrivals — about **twice the arrival rate**. Durable fields landed: 53
enriched, 45 with developer address, holding and mint authority, 36 X handles, 27 websites, 15
descriptions, 4 Telegram; holders present for 27 of 53, confirming the ~50% availability flagged
earlier. The second run processed only the 5 genuinely pending rows, not the 43 already done.

## Two things to expect, so neither reads as a fault

**The ladder lags enrichment by one run.** A rung needs a measurement pool, and rows have none until
enrichment sets one — hence 0 rungs in the first run and 15 in the second. Empty early rungs are
normal on a cold backlog.

**Measurement-pool selection timing is a commissioning metric.** 44 of 53 pools in this backlog were
`selected late, not at graduation`, because it predates the split. Once fresh data accumulates the
ratio should invert, and `backlog_health()` reports it as `at_graduation_share`. **If that share
stays low, the cadence-critical sweep is not reaching its cohort members and the backlog is quietly
covering for it** — a pool chosen hours later may name a different market than the one that mattered,
so a low share silently degrades every post-graduation measurement. It read 0.25 at commissioning,
as expected for a pre-split backlog.

## What this archive is, and is not

It is a **graduation and post-graduation archive**. It is **not** a reliable archive of launch times:
`first_pool_created` on Solana is indexing/migration-adjacent, with a median gap to graduation of 0s
for Pump.fun and 52s for Meteora DBC. Any "time to graduation" or "minutes since launch" statistic
computed from these columns will be confidently wrong, which is why the caveat is a
`COMMENT ON COLUMN` rather than prose here alone. True launch timing needs Solana RPC or Bitquery
against the launchpad program.


# Semantic review, and what it changed (2026-09-19)

A review against five silent-failure concerns found three real defects. All are fixed; the re-review
numbers are from the commissioning data.

**Pump.fun analysis could not filter correctly.** `launchpad` is only ever a curve DEX, so it was
NULL for every graduate first seen arriving at its destination — and a fast graduator is frequently
only ever seen that way. `WHERE launchpad='pump-fun'` returned **9 of 16** real Pump.fun graduates,
a silent 44% undercount biased toward exactly the fastest tokens. Fixed with `launchpad_family`,
which spans both sides of a pairing (`pump-fun` and `pumpswap` are both `pump.fun`), derived by the
adapter so there is one source of truth. Re-review: **16 of 16, zero unattributed.**

**The worker could report perfect health while captures were being missed.** Enrichment health said
`idle` — the strongest possible signal — while capture coverage was 3 of 16. Health is now three
states kept deliberately apart: `enrichment_state` (durable work, judged on the age of the oldest
pending graduate), `capture_state` (perishable work, judged on coverage), and `state`, the worse of
them. The capture denominator counts only graduates that arrived **after the first capture we ever
took**, so a pre-mechanism backlog cannot manufacture a failure that never happened — it cut the
denominator from 16 to 5 on this data. A sample below 20 reports `unknown`, never `healthy`.

**`at_graduation_share` was reported but not surfaced.** It is now on the `--daily` row with its
numerator and denominator (`pools_at_graduation` / `pools_measured`), alongside capture coverage,
and it only produces a verdict once the sample justifies one.

**Also, selection timing is an explicit column.** `measure_pool_timing` is `at_graduation` | `late`,
not a `LIKE '%selected late%'` against a prose reason. The reason string keeps the detail; analysis
never parses it. Encoding analytical semantics indirectly is how a filter silently comes to mean
something other than what it says — the same class of mistake as the `launchpad` undercount above.

**Robinhood was confirmed unchanged**: worst-case optional work is 150s against its new 180s
deadline, so nothing it previously completed is cut, and discovery's half-deadline cap (90s) is far
above its ~25s cost. One immaterial change: on an arrival-only row `graduation_pool` now prefers the
declared destination over the observed pool; those were verified identical for Pons.

# Chain scoping, a watermark ratchet, and an automated report (2026-09-20)

Building a read-only reporting workflow turned up two live defects underneath it. Both were found by
validating the report rather than by reading code, and both are the archive's signature failure
mode: not a crash, but a plausible number that is wrong.

## 1. launchpad_sweeps had no `network` column

Both archives wrote into it undifferentiated, and seven reads across the collector and both report
surfaces were chain-blind.

The reporting consequence is that `report()`'s `continuous` verdict compares sweeps recorded against
sweeps expected **for that chain's cadence**. Blended, Solana's 720/day alone clear Robinhood's 288
expected, so **`continuous` reads true straight through a total Robinhood outage** - the V1
deliverable, silently inverted. Reproduced in a test: with the scoping removed, Robinhood reports 4
sweeps where it has 1.

The worst of the seven was not in a report at all, but the gap-detection watermark inside `sweep()`,
which read the most recent sweep of either chain. That breaks in both directions and neither is
visible in the output: against a busy chain (newest pool ~ now) the gap is pinned at 0 and the alarm
can never fire; against a quiet one it fabricates a gap on every sweep.

Fixed by scoping all 25 reads to `chain.network` and writing `network` on every sweep row, indexed
on `(network, started_at DESC)`. Rows written before the column existed cannot be attributed after
the fact, so they are counted and named as `sweeps.unattributed_pre_migration` rather than guessed
into a chain or dropped from the denominator - a window straddling the migration reads as partly
unknown, never as an outage.

## 2. The watermark ratchet, measured in production

Two consecutive Robinhood sweeps on main, 24 minutes apart:

```
14:38  sweep 537  pages 10  window 1100s  gap_seconds 72731  errors: ["state reads stopped ..."]
15:02  sweep 547  pages 10  window  756s  gap_seconds 74555  errors: ["state reads stopped ..."]
```

`gap_seconds` grew 1824s across 1435s of wall clock; the residual is the feed window shrinking by
344s, which moves `oldest_pool_at` later by the same amount. So `previous_newest` was **frozen** and
the reported gap was just now-minus-a-fixed-instant: 20.7 hours and climbing.

`complete` is false whenever a sweep records **any** error, including the entirely benign
"state reads stopped to reserve budget for captures" - the budget reserve working as designed. The
watermark read only advanced across `complete` sweeps, so one such sweep froze it, the next sweep
measured its gap against that stale instant and was therefore incomplete too, and the freeze became
permanent. It is self-reinforcing twice over: with the watermark stuck in the past the early stop
`oldest <= previous_newest` can never fire, so discovery walks all ten pages every sweep, exhausts
the budget, and emits the very error that holds the latch shut.

Chain scoping alone would **not** have cleared this - Robinhood's own recent sweeps are all
incomplete, so a scoped query still reaches back past them. The predicate itself had to go.

Dropping it is also what `gap_seconds` already documents: "the previous sweep's newest pool", not
the previous *complete* one. `newest_pool_at` comes from page 1, which every sweep fetches however
early it stopped, so it is trustworthy even on an incomplete sweep - unlike `oldest_pool_at`, which
is exactly what varies with depth and is still read only from full-depth sweeps. No information is
lost: the gap is recorded on the row where it happened and `daily()`/`report()` aggregate
`max(gap_seconds)` over the window. Re-deriving the same gap forever only prevented the archive from
ever reading healthy again.

## The report workflow

`.github/workflows/graduation-archive-report.yml` runs the three read-only surfaces for either or
both chains and writes the JSON into the job summary, on `workflow_dispatch` and a 6-hourly cron.

Read-only by construction: neither `--execute` nor `--enrich` is reachable from it, so dispatching it
can never disturb collection or compete for the archive's advisory lock. Inputs arrive through `env`
and are validated in the shell rather than interpolated with `${{ }}`, because anyone who can press
dispatch reaches a job holding a live database credential. The runner starts the step as `bash -e`,
so the script turns errexit off explicitly - otherwise the first failing surface would abort before
its exit code was recorded. Every surface is attempted and the run then fails if any could not be
read, so a partial report is never mistaken for a complete one.

It is deliberately **not** registered in `DISPATCH_TARGETS`. That file is for workflows whose value
depends on hitting their cadence, because a missed sweep loses data permanently. Nothing this prints
is perishable, so GitHub's scheduling drift costs nothing and it does not belong on the 2-minute
Vercel tick.

`schema_ready()` closes the deploy-order hazard. The read-only surfaces deliberately never call
`setup()`, because `setup()` writes (the family and timing backfills) and a report that mutates the
archive is not a report. That leaves the trap this repo has hit before - a reader shipped ahead of a
Python-owned `ALTER` that lands only when a collector next runs. `--daily`, `--backlog` and
`--report` now check for the column and print `{"status":"schema_pending"}` with a non-zero exit
instead of crashing or, far worse, reporting zeros. It self-heals on the next sweep.

## A note on how these were found

The live-smoke gate above exists because mocked tests execute these paths but cannot evaluate them.
Both defects here extend that lesson one step further: neither was reachable from a single-chain
test of any kind, mocked or live, because both require **two collectors running against one
database**. The ratchet additionally required a *sequence* - three sweeps, one of them erroring -
that no single run produces. Each now has a regression test that was confirmed to fail against the
pre-fix code with the exact symptom observed in production.
