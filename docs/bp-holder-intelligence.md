# BP holder intelligence: implementation status and runbook

Contract: [`bp-holder-intelligence-spec.md`](bp-holder-intelligence-spec.md) (Revision 2, 2026-09-26). This file
records what is built, where it deviates from the contract and why, and how to operate it. It extends the
Backpack monitor ([`backpack-monitor.md`](backpack-monitor.md)); nothing here is a separate service.

Status (2026-09-26): **Milestones 2 and 3 are built and tested against fixtures and PostgreSQL. Milestone 1's
probe is built but has not been run against production. No production data has been collected, and no alert
has been checked against live transactions.** Milestones 4 and 5 require elapsed time and decisions (below).

## What is built

| Area | Code | Notes |
| --- | --- | --- |
| Cohorts (section 5) | `backpack/cohorts.py` | Ranks owners from the latest complete, supply-reconciled BP capture already stored by the daily job; withholds the ranking when the retained holder rows do not match the capture's owner count. Raw and filtered rankings (top 250 each) persist in `bp_holder_rankings`, so they survive the 30-day raw-snapshot retention. Hysteresis: enter at filtered rank <= 200, leave after 2 consecutive versions below rank 220 or absent; confirmed/high system labels leave immediately. Entrant cap 20; overflow is queued in the version (`event='queued'`) and admitted oldest-first. Versions are immutable, one per capture date. |
| Original cohort | `cohorts.approve_original`, `/admin/backpack` | One reviewed current version is frozen once, with notes; it cannot be replaced (unique index). Its wallets stay tracked after they leave the current cohort. |
| Portfolio collector (section 6) | `backpack/portfolio.py`, `intel.collect_portfolios` | One batched JSON-RPC request per wallet (`getBalance` + `getTokenAccountsByOwner` x2 programs, finalized). Raw integers throughout. Per-wallet read status `complete / partial / unavailable / oversized`; a failed program read never implies absence. Token-2022: confidential accounts are kept with `balance_visibility='partial'` even at a zero public balance; the RPC display amount is stored separately (`ui_amount`, interest-bearing/scaled mints). NFTs are stored but excluded from totals. Stale (>1h) and missing prices keep `value_usd` NULL. |
| Metadata and prices | `Providers.assets`, `Providers.prices`, `Providers.block_times` | DAS `getAssetBatch` (weekly refresh), Token-2022 mint extensions as flags; Jupiter Price V3 in batches of 50 with block-derived timestamps via batched `getBlockTime`. Jupiter's `liquidity` and `createdAt` feed the liquidity column and the recently-launched flag. |
| History (section 8.1) | `intel.poll_wallet` | Helius `getTransactionsForAddress`, `tokenAccounts='balanceChanged'`, finalized, `jsonParsed`. Separate per-run page budgets for newer activity and backfill; a total backfill cap per wallet. Coverage per wallet in `bp_history_coverage` (requested start, earliest retrieved, completion status, recorded gaps). Complete only when a transaction older than the requested start is seen or the provider reports no older history. |
| Classifier (section 8.2) | `backpack/events.py` | Works on the raw RPC shape: owner deltas from pre/post token balances, fee and token-account rent removed from native SOL, native + wrapped SOL combined within a transaction. Tiers `parsed_swap` (Helius Enhanced parse, fetched only for swap-shaped transactions), `inferred_swap` (one asset out, one in, known DEX program from `backpack/dex_programs.json`), `unclassified`. Transfers, airdrops, wraps, liquidity, permanent-delegate transfers and failed transactions are separate kinds. Deterministic event index per signature and wallet. |
| New / pre-balance (section 8.3) | `events.classify`, `intel.refresh_flags` | Pre-balance from the transaction; snapshot fallback only for token accounts outside the transaction (`pre_balance_source='snapshot'`). ATA creation recorded. First observed purchase, re-entry, first observed group purchase and recently launched are recomputed after every poll so out-of-order deliveries converge. |
| Valuation | `events.valuation` | Stablecoin leg at $1 (estimated), else SOL leg at the nearest stored Jupiter SOL observation within one hour. Other trades stay unpriced. |
| Alerts (section 10) | `backpack/alerts.py`, `bp_alert_rules` | New position, multiple buyers, accumulation, major sale; thresholds are DB rows. Only finalized parsed/inferred swaps by members of the evaluated version, after they joined tracking. Stable keys; overlapping windows within the cooldown update the existing alert. Evaluated for the original cohort (if approved) and the latest current version. |
| Reconciliation | `intel.reconcile`, `bp_reconciliation` | Daily: token balance change between two complete reads ~24h apart vs the net of classified events between their slots, only for wallets with gap-free history. Discrepancies appear on the page. |
| Retention | `intel.maintain` | Raw transactions 45 days (min 30). Hourly portfolio reads older than 48h thin to the first read of each UTC day; reads older than 90 days are dropped. Events, cohorts, rankings and alerts are permanent. |
| Web (section 12) | `lib/server/bp-intel-store.ts`, `components/backpack/holder-intel.tsx` | Sections on `/market/crypto/backpack`, read through `/api/market/crypto/backpack?section=overview|roster|overlap|activity|alerts|wallet|token` with `cohort`, `run`, `mint`, `wallet`, `days`; `format=csv` on every section (`ranking=raw` for the raw top 200). The page shows the cohort version, the portfolio read, read/history coverage and a warning list whenever collection is incomplete, prices are stale, history is capped, or reconciliation disagrees. Live monitoring is shown as off. |
| Admin | `lib/server/bp-intel-admin.ts`, `components/backpack/cohort-admin.tsx` | Version list and one-time original-cohort approval; Vesting and Burn labels added. |
| Milestone 1 probe | `backpack/feasibility.py` | Read-only. See runbook below. |

## Deliberate deviations from the contract

1. **Coverage table.** Section 8.1 says to extend `backpack_transaction_cursors`. Its `(asset_id, wallet)` rows
   already carry the daily BP swap sampler's cursor (`through_at` equal to the previous UTC midnight, an
   Enhanced-API signature). Advancing them from an hourly all-token poll would break that sampler, so coverage
   lives in `bp_history_coverage` with the fields the spec lists.
2. **History endpoint chosen before the live check.** The collector uses `getTransactionsForAddress` because
   its documented `tokenAccounts='balanceChanged'` filter targets exactly the incoming-SPL gap, and it returns
   the raw `meta.pre/postTokenBalances` the classifier needs. The Milestone 1 probe tests this claim on real
   wallets; if it fails, switch the fetch in `intel.poll_wallet` and record the gap here. The Enhanced API is
   kept only for the `parsed_swap` tier.
3. **Rent and incidental SOL.** Each token account's rent is attributed from the parsed instructions (who funded
   `createAccount` / the ATA `source`, where `closeAccount` sent the lamports). Only rent the wallet itself paid or
   received is netted from its SOL; rent a sender paid, or a refund sent elsewhere, is ignored. Without instruction
   evidence, rent only cancels the part of the wallet's own SOL change it can explain. After that, SOL changes of 0.003 SOL or less
   (`BP_INCIDENTAL_LAMPORTS`) are treated as incidental (rent for someone else's account, tips). A trade smaller
   than that loses its SOL leg and is classified from its token side only.
4. **Several swaps in one transaction** are kept as unpaired `unclassified` legs. Pairing them from balances is
   not possible and the Enhanced parse reports route legs, not independent swaps.
5. **First observed group purchase** is computed across all tracked wallets, not per cohort version.
6. **Major sale excludes SOL and stablecoins by default** (`exclude_mints` in `bp_alert_rules`): spending a
   whole USDC balance to buy a token is a funding leg, not a sale signal. Editable per rule.
7. **Pre-membership.** A wallet's `tracked_since` is its first entry into any version. A wallet that leaves and
   later re-enters keeps its first entry time; flows are always restricted to members of the selected version.
8. **Runs share `backpack_ingestion_runs`** with a new `job` column (`daily` default, `bp_intel`). Every daily
   reader, the provider-budget alert and the cost report now filter or group by job, using an expression that
   also works before the column exists.

## Operating

```sh
python backpack_monitor.py --migrate            # daily workflow already does this
python backpack_monitor.py --bp-cohort          # create today's current version (no provider calls)
python backpack_monitor.py --bp-intel           # portfolio, history, flags, alerts, reconcile, retention
python backpack_monitor.py --bp-intel --bp-steps portfolio,alerts
python backpack_monitor.py --bp-feasibility     # Milestone 1 probe, read-only
python backpack_monitor.py --bp-approve-original 12 --bp-notes "reviewed labels and exclusions"
```

Workflows: the daily `backpack-monitor.yml` refreshes the cohort after a successful capture.
`bp-holder-intel.yml` on manual dispatch runs `mode=holdings` (default: cohort plus one portfolio read, no
transaction history, enough for the top-200 common-holdings view), `mode=hourly` (holdings, history, alerts) or
`mode=feasibility` (the probe).
Its hourly schedule is **off** until the repository variable `BP_INTEL_SCHEDULE=1` is set; it is also not in
the Vercel dispatcher (`lib/server/github-dispatch.ts`). Both are deliberate: the monthly API budget is an open
decision (spec section 15).

Configuration (environment or repository variables): `BP_COHORT_SIZE` 200, `BP_COHORT_EXIT_RANK` 220,
`BP_COHORT_EXIT_RUNS` 2, `BP_COHORT_ENTRANT_CAP` 20, `BP_HISTORY_DAYS` 30, `BP_HISTORY_POLL_PAGES` 5 (workflow 3),
`BP_HISTORY_BACKFILL_PAGES_PER_RUN` 5 (workflow 3), `BP_HISTORY_MAX_BACKFILL_PAGES` 30, `BP_HISTORY_PAGE_SIZE` 100,
`BP_INTEL_MAX_REQUESTS` 2500, `BP_MAX_TOKEN_ACCOUNTS` 10000, `BP_DUST_USD` 1, `BP_INCIDENTAL_LAMPORTS` 3000000,
`BP_RAW_TX_RETENTION_DAYS` 45, `BP_PORTFOLIO_RETENTION_DAYS` 90, `BP_ALERT_LOOKBACK_HOURS` 48,
`BP_RECENT_LAUNCH_DAYS` 14, `BP_ENHANCED_PARSE` 1, `BP_RPC_BATCH` 1. Web: `BP_MEANINGFUL_USD` 100.

The worker never runs DDL; on an unmigrated database it returns `schema_pending`. The web reader returns
`schema_pending` too, so deploy order does not matter. Worker exclusion uses the `bp_intel` lease row.

## Milestone 1 runbook (not yet run)

1. Merge, let `backpack-monitor.yml` migrate and create the first cohort version (or dispatch it).
2. Dispatch `bp-holder-intel.yml` with `mode=feasibility`. It reads the top filtered holders, spends at most
   `BP_FEASIBILITY_MAX_REQUESTS` (400) requests, writes nothing, and uploads `bp-feasibility.jsonl`.
3. Record the answers here:
   - `history_endpoints[].token_account_only` > 0 with `owner_in_account_keys=false` confirms that the
     token-account filter finds incoming transfers the owner query misses; `token_account_only_in_enhanced`
     says whether the Enhanced endpoint also finds them.
   - `closed_accounts.checked[].earlier_activity_in_owner_query` vs `earlier_activity` says whether history of
     since-closed token accounts is still returned when querying the owner.
   - `portfolio_sizes` (account counts, bytes, latency), `metadata` (DAS and Token-2022 coverage, extension
     kinds), `prices` (priced share, staleness), `transaction_volume` (30-day counts per sampled wallet).
   - `swap_programs.unknown`: programs in swap-shaped transactions not in `dex_programs.json`; add real DEXes
     with a source.
   - `usage_estimate`: monthly calls/credits for the configured cohort size.
4. Webhook matching is not probed (it needs a registered webhook); it stays a Milestone 4 question.
5. Then run `mode=hourly` manually a few times, check `bp_history_coverage`, a sample of events against
   Solscan, and the reconciliation table, before setting `BP_INTEL_SCHEDULE=1`.

## Open decisions and remaining work

- **Monthly API budget** (gates the hourly schedule). The probe's `usage_estimate` is the input.
- **Manual validation of a live sample** of parsed and inferred swaps before relying on alerts (section 13).
- **Original cohort approval**: an admin must review the first version's labels and exclusions and approve it.
- **Milestone 4**: live ingestion decision and the seven-day pilot (gaps, classification accuracy,
  reconciliation, latency, actual cost). Nothing live is built.
- **Milestone 5**: alert enablement decision, monitoring, and the support runbook appended to
  `backpack-engineer-handoff.md`.
- Historical SOL valuation only exists from the first hourly run onward; backfilled SOL-leg trades before that
  stay unpriced. A timestamped SOL source would fill it.
- Related-wallet warnings are omitted (spec version 1).

## Verification (2026-09-26, local)

- `tests/test_bp_intel.py`: 35 fixture tests (every classifier case in section 13, portfolio parsing, cohort
  hysteresis/cap, alert rules including "no transfer-only fixture raises a buy alert", third-party rent).
- `tests/test_bp_intel_integration.py`: 12 PostgreSQL tests (cohort versions from stored captures, partial
  capture withheld, original approval via both the Python and the web SQL, full worker run replayed with no
  duplicate events or alerts, page caps recorded as gaps, reconciliation, retention, no DDL on an unmigrated
  database, probe is read-only, a wallet with no history keeps being polled) plus every web SQL template on
  empty and populated schemas.
- Three independent reviews (classifier/worker, web layer, providers/schema/workflows) found seven issues, all
  fixed with tests: rent a sender paid counted as the wallet's SOL, then netted against an unrelated swap; roster
  amounts assuming 9 decimals; detail-panel request races and stale scope; explicit out-of-window cohort/run IDs
  silently becoming empty (now 404); a double-counted JSON-RPC call metric; NULL block times in flag refresh.
- Existing Backpack suites unchanged and passing (124 Python tests in total). TypeScript: `npm run test:backpack`
  (13), `tsc`, targeted ESLint and `next build`.
- Local PostgreSQL was PGlite 0.5.8 (PostgreSQL 18, WASM); CI runs PostgreSQL 16.
