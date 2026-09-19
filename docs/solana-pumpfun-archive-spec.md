
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
