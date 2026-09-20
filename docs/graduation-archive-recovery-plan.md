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
- Live deployment verification and the new commissioning start are pending.

The multi-pool endpoint is documented in the [GeckoTerminal API changelog](https://apiguide.geckoterminal.com/changelogs).
