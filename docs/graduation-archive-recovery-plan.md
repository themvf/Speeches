# Graduation Archive recovery — 2026-09-20

Approved scope: commit this plan, implement the repairs, and push to main.

## Baseline

At 15:13 UTC, deployed main was dbd60f9. Robinhood sweep 551 exhausted ten
pages and reported a 75,067-second watermark gap. Solana enrichment reported
1,726 pending, oldest age 29,594 seconds, service 147/hour versus arrivals
198/hour, capture coverage 342/912, and at-graduation selection 374/674.
These are pre-repair observations, not trustworthy cross-chain continuity.

## Implementation sequence

1. Recover pending work if available; otherwise implement against current main
   in an isolated checkout, preserving unrelated local changes.
2. Scope sweep writes, watermark reads, daily summaries and continuity reports
   by network. Leave legacy sweep attribution unknown. Read-only reports must
   explicitly report schema pending until a collector applies the migration.
3. Advance the discovery watermark from the latest observed feed even when
   optional work failed. Preserve real gaps and errors instead of erasing them.
4. Bound request waits, retries and timeouts by the phase deadline. Prioritize
   graduation captures, avoid pool lookups outside the sampled cohort, and reserve
   worker capacity for the measurement ladder as well as metadata enrichment.
5. Add a read-only, manually dispatched report workflow for either or both chains,
   with JSON artifacts and a readable GitHub summary. Include measurement coverage
   and lateness so an empty ladder cannot hide behind metadata health.
6. Run regression tests against disposable PostgreSQL, validate workflow scripts,
   review the diff, push main, and inspect real collector/report runs.

## Acceptance and follow-up

- A failure or sweep on another chain cannot freeze or advance this chain's watermark.
- Reports cannot label a chain continuous because another chain ran more often.
- Retries cannot consume the next phase's reserved time; actual observation times
  are retained and missed historical captures are never fabricated.
- Live sweeps advance their watermark and stop paging when overlap is reached.
- Enrichment and ladder throughput are reported independently, with numerator and
  denominator for capture and measurement coverage.
- Start a fresh 48–72-hour commissioning window only after live recovery is
  demonstrated. Check backlog trend, opening capture coverage, selection timing,
  rung coverage/lateness and per-chain continuity. Deployment alone is not recovery.

The expired opening-trade window cannot be backfilled. Retain historical gaps and
separate pre-repair evidence from the new commissioning window. No paid data,
expanded social collection, new chain, trading score or UI is part of this work.

## Execution record

- Plan committed separately before implementation (4b6a8a4).
- No published pending patch or open archive PR was available; implemented from
  deployed main in `C:/archive-recovery` on `codex/archive-recovery`.
- Added nullable sweep network migration and chain filters throughout continuity
  and daily queries. Unknown legacy rows explicitly prevent a complete verdict.
- Removed the complete-sweep watermark predicate; previous errors/gaps remain
  stored. Reports also check leading/trailing silence, not just sweep counts.
- All request retries/timeouts share a phase deadline. Graduate arrivals receive
  state lookup priority. Solana takes one trade page because pages overlap.
- Enrichment uses at most 65% of an 85%-of-cadence budget, reserving the remainder
  for the ladder. Pool lists are skipped outside the sampled cohort. Failed pool
  requests stay retryable rather than becoming permanent "no pools" outcomes.
- Ladder measurements use the public multi-pool endpoint, up to 30 pools/request.
  Live API probe returned both requested pools with exact address matches (HTTP 200).
  Responses retain actual fetch times; reports include per-rung eligibility,
  missing observations, missing pools, coverage, and median lateness.
- New read-only workflow supports both chains and all three report surfaces,
  publishing six JSON artifacts plus a summary. The database enforces read-only
  transactions; schema migration remains collector-owned.
- Local non-database checks passed. Before deployment, [CI run 35519910593](https://github.com/themvf/Speeches/actions/runs/35519910593)
  passed all 60 archive checks against disposable PostgreSQL and the broader Python
  suite (765 passed, 86 skipped). The report script produced all six JSON sections
  with PostgreSQL enforcing read-only mode. Initial CI caught a missing required
  `last_seen_at` in the new test seed helper; corrected before the successful run.
- Repair code is deployed; production verification is recorded below. A clean
  commissioning verdict remains pending: historical capture deficits do not vanish
  when code is deployed, and current provider throttling must remain visible.

### Production verification findings

- Deployed Solana sweep 560 advanced to `gap_seconds: 0` after the first
  chain-attributed sweep, confirming the watermark no longer freezes.
- The first report invocation failed with PostgreSQL startup options. The pooled
  connection-compatible version uses read-only session transactions instead; all
  six production report sections completed in run 35520091620.
- Concurrent live jobs still received HTTP 429s. Added one shared API pacing row:
  reserve request starts at 2.5-second intervals across archive workers, propagate
  provider cooldowns, and decline reservations beyond the current phase deadline.
  This is a deliberate exception to batching all database writes at sweep end:
  tiny coordination transactions prevent three jobs each consuming the entire
  public allowance. The existing two-minute collector already keeps the database
  awake. No new service or paid API is introduced.
- Completed jobs were followed by repeated `already_running` results. The original
  lock was session-scoped but transactions were committed throughout each run;
  [Neon transaction pooling](https://neon.com/docs/connect/connection-pooling)
  does not support session-level advisory locks. Replaced these with atomic,
  owner-checked worker leases keyed by chain and job type. Leases expire after
  six minutes (sweep) or fifteen minutes (enrichment), beyond workflow hard limits
  of five/twelve minutes. A terminated runner cannot leave a permanent lock, and
  an expired owner cannot release its successor's lease. Existing workflow
  concurrency groups serialize rollout with the prior version.
- Daily pool-selection counts now exclude graduates outside the sampled cohort,
  matching the backlog surface rather than inflating its denominator.
- Production exposed a nested connection context in Robinhood's inline ladder
  after shared pacing was introduced. Fixed by closing the lookup transaction
  before requesting an API slot, with a regression that uses the actual database
  coordination path (HTTP only is mocked). Final repair gate:
  [35520831033](https://github.com/themvf/Speeches/actions/runs/35520831033),
  **63 archive tests passed** against PostgreSQL; broader suite **765 passed,
  89 skipped**. Code deployed as `a8dc648`.

### Live evidence and remaining acceptance

| Evidence | Result |
| --- | --- |
| [Solana sweep 563](https://github.com/themvf/Speeches/actions/runs/35520781372) | Completed after its predecessor, three pages, zero feed gap, one opening capture with 13 trades; rate-limit/budget errors retained |
| [First repaired enrichment](https://github.com/themvf/Speeches/actions/runs/35519976617) | 42 tokens enriched, 570 ladder measurements; previously the inspected window had zero ladder measurements |
| [Next enrichment](https://github.com/themvf/Speeches/actions/runs/35520494144) | 48 tokens enriched, 676 measurements, backlog 1,685, service 163/hour versus arrivals 145/hour |
| [Production read-only report](https://github.com/themvf/Speeches/actions/runs/35520783131) | Six JSON sections and summary; historical unknown-network rows explicitly listed; ladder denominators/lateness visible |

Oldest enrichment age remains about 8.7 hours. Opening capture and selection health
remain degrading over the trailing 24-hour sample. The 629-second Solana gap in
the first lease-based run is a real missed interval following skipped collectors;
the next run returned to zero. It is retained, not rewritten as recovered data.

The fresh 48–72-hour commissioning acceptance must therefore remain open. Require
repeated successful per-chain collection, shrinking oldest backlog age, improving
new-cohort capture/selection coverage and ladder coverage/lateness before calling
the archive healthy. Use `graduation-archive-report.yml` with `chain=both`,
`mode=all`, `hours=48` (or 72); a shorter continuity window helps separate the
new deployment from legacy unattributed sweeps. No historical capture backfill
or claim of restored completeness is made.

The multi-pool endpoint is documented in the [GeckoTerminal API changelog](https://apiguide.geckoterminal.com/changelogs).
