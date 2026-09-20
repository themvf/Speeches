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

Implementation and live verification pending.
