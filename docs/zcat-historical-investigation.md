# ZCAT historical investigation

Scope: July 25, 2026 00:00 UTC through September 14, 2026 00:00 UTC (51 days).
The user requested late July as the start of the proposed ZCAT investigation.
The separate 75,000-credit historical allowance never resets; the current live
tracking pilot retains its original 50,000-credit ceiling. Maximum combined
allowances are 125,000 credits. Existing posts/windows are reused, with no
historical follower counts, bios, or engagement-at-publication invented.

The first pass scans 204 six-hour windows, reserving at most 61,200 credits before
reuse. Remaining allowance deepens the earliest unfinished periods chronologically.
This prioritizes potential origin evidence; it does not guarantee full coverage
or allocate equally to the eventual peak. Up to 40 requests/12,000 reserved credits
per run. The first run starts on merge; daily 08:43 UTC runs resume the fixed range.
Once the allowance or searchable windows are exhausted, reruns make no paid calls.
Default manual/CLI mode is plan-only. No refunds or retries for unknown outcomes.

All collectors serialize on the original singleton lock, and all outstanding or
uncertain request records block new paid calls. History reservations update a
separate campaign row. Shared post/match/edge records retain original timestamps
and source URLs. API reporting excludes historical estimated spend from the live
pilot subtotal. A changing cursor with the exact same returned post set halts
rather than looping. Search gaps and deleted/unindexed posts remain limitations.

The dashboard's expandable 'Before the run' section shows the earliest 30 saved
posts, coverage progress and history spending. These are earliest *found*, not
proven first mentions. Historical market price, liquidity and volume alignment,
token/pool verification, and causal analysis remain the subsequent research step.
The present change does not claim a verified explanation of the price move.

Validation: dedicated tests exercise the fixed date range, separate non-resetting
budget, reused windows, live-pilot window isolation, saved evidence and failure
handling against disposable Postgres. Existing collector and UI checks still run.
