
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
