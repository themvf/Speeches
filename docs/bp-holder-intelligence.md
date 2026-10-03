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
| Retention | `intel.maintain` | Raw transactions 45 days (min 30). Portfolio reads older than 48h thin to one per UTC day, the day's most complete read (latest on ties); reads older than 90 days are dropped. The newest stored read, which the page shows, is never deleted. (Until 2026-10-03 the first read of the day was kept; on 2026-10-03 that deleted 50,000 balance rows from the better 2026-09-27 reads.) Events, cohorts, rankings and alerts are permanent. |
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
python backpack_monitor.py --seed-labels        # committed wallet labels; daily workflow already does this
```

**Cadence.** Cohort: daily, inside `backpack-monitor.yml` after its capture (GitHub-scheduled for 00:30 UTC;
observed starting around 05:15). Holdings and transaction history: **daily** (since 2026-10-03), dispatched by the
Vercel cron dispatcher (`github-dispatch.ts`, every 24 hours, no inputs, so `mode=daily`): holdings, history poll,
flags, alerts, reconciliation and retention, with the 100-credit Enhanced parse off (`BP_ENHANCED_PARSE=0`; swaps are
`inferred_swap` from balance changes through known DEX programs). `mode=holdings` remains for a holdings-only read. The dispatcher ignores runs
GitHub created only to skip: the gated hourly schedule creates one every few hours, and until 2026-10-02 they made
the job look fresh, so no holdings refresh ran from 2026-09-27 to 2026-10-02. History and alerts: hourly only
once `BP_INTEL_SCHEDULE=1`.

**Helius credits (published costs, 2026-09-27: standard RPC 1, each DAS request 10, Enhanced 100 per request,
`getTransactionsForAddress` 10 per 100 transactions).** A holdings run is about 600 credits for 200 wallets (three
calls each), about 10 per 1,000 new or stale mints of metadata, 2 for the slot reference, and one per exact price
time: token price times are estimated from slot distance and looked up exactly only within the band where the
one-hour staleness call could go either way (SOL's is always exact because stored SOL prices value swaps). Roughly
1k credits a day, about 30k a month. For comparison, the existing daily Backpack job recorded about 178 Enhanced
requests a day in 2026-09-24/26 runs, about 530k credits a month on its own, so both fit the free plan's 1M with
the hourly history schedule off. Requests are paced to 8 RPC calls/s and 1.5 DAS calls/s (`BP_RPC_CALLS_PER_SECOND`,
`BP_DAS_CALLS_PER_SECOND`) to stay under the free plan's 10 and 2 per second. `BP_PRICE_TIME_MODE=exact` restores
one lookup per priced token.

Daily history estimate (published `getTransactionsForAddress` cost, 10 credits per call of up to 100 transactions):
about 222 wallets x 1-3 poll pages plus up to 3 backfill pages for wallets still backfilling, roughly 2k-10k credits a
day, about 100k-200k a month with holdings; with the daily Backpack job's ~530k that stays under the free plan's 1M.
Measured per run in `backpack_provider_usage` (calls carried in batches are recorded alongside HTTP requests).

Workflows: the daily `backpack-monitor.yml` refreshes the cohort after a successful capture.
`bp-holder-intel.yml` on manual dispatch runs `mode=holdings` (default: cohort plus one portfolio read, no
transaction history, enough for the top-200 common-holdings view), `mode=hourly` (holdings, history, alerts) or
`mode=feasibility` (the probe).
Its hourly schedule is **off** until the repository variable `BP_INTEL_SCHEDULE=1` is set; it is also not in
the Vercel dispatcher (`lib/server/github-dispatch.ts`). Both are deliberate: the monthly API budget is an open
decision (spec section 15).

Configuration (environment or repository variables): `BP_COHORT_SIZE` 200, `BP_COHORT_EXIT_RANK` 220,
`BP_COHORT_EXIT_RUNS` 2, `BP_COHORT_ENTRANT_CAP` 20, `BP_COHORT_EXCLUDE_MARKET_MAKERS` 1, `BP_HISTORY_DAYS` 30, `BP_HISTORY_POLL_PAGES` 5 (workflow 3),
`BP_HISTORY_BACKFILL_PAGES_PER_RUN` 5 (workflow 3), `BP_HISTORY_MAX_BACKFILL_PAGES` 30, `BP_HISTORY_PAGE_SIZE` 100,
`BP_INTEL_MAX_REQUESTS` 2500, `BP_MAX_TOKEN_ACCOUNTS` 10000, `BP_DUST_USD` 1, `BP_INCIDENTAL_LAMPORTS` 3000000,
`BP_RAW_TX_RETENTION_DAYS` 45, `BP_PORTFOLIO_RETENTION_DAYS` 90, `BP_ALERT_LOOKBACK_HOURS` 48,
`BP_RECENT_LAUNCH_DAYS` 14, `BP_ENHANCED_PARSE` 1, `BP_RPC_BATCH` 1. Web: `BP_MEANINGFUL_USD` 100.

The worker never runs DDL; on an unmigrated database it returns `schema_pending`. The web reader returns
`schema_pending` too, so deploy order does not matter. Worker exclusion uses the `bp_intel` lease row.

## On X tab: Backpack posts next to the holder data (2026-10-03)

`?section=social` (tab "On X", CSV) joins the X tracker's saved posts with the holder data. Pure logic in
`apps/web/lib/bp-social.ts`, SQL in `bp-intel-store.ts`, tests in `lib/bp-social.test.ts` and
`tests/test_bp_intel_integration.py`.

- **What counts as a Backpack post**: posts matching the BACKPACK registry entry (contract, `$BACKPACK`, or `$BP` with
  Backpack/Solana context). Saved search results that only contain the word "backpack" are excluded.
- **Coverage, never zero by default**: each day shows whether its X search windows with the current query were searched to
  the end or partly; "earlier search only" when only the contract-only origin search or the pre-2026-10-03 query reached
  it (counts are lower bounds: $BP was not searched); otherwise "not searched", and posts read "not searched", not zero.
- **Daily row**: posts, accounts, copy-paste posts, search coverage, BP holders (all owners and $100+ economic holders
  from the daily capture, null when the capture is incomplete), day-on-day change, top-200 entered/left (current
  versions created that day), and top-200 BP buyers/sellers from classified swaps. Trades after the earliest
  `last_poll_at` among members with complete history read "not observed".
- **Copy-paste campaigns**: near-identical posts (first twelve words after removing links, mentions, numbers and
  addresses) from three or more accounts. The page states that who coordinated them is unknown.
- **Coins the top holders share, on X**: coins in the selected portfolio read that the tracker follows, linked by
  contract on Solana, or by ticker for a coin the tracker follows on another chain (labelled; not the same asset; only the
  most-held token with that ticker, and only with 3+ holders at $100+, because copycat tokens share tickers). Posts
  in the last 7 days count only where they match that coin, with the windows still unsearched.

## Committed wallet labels (2026-09-27)

The cohort is meant to be investors. Labels decide who is not one, and labels normally come from `/admin/backpack`.
The first review was done without an admin login, so its results are committed as evidence in
`backpack/wallet_labels.json` and loaded by `python backpack_monitor.py --seed-labels`, a step in
`backpack-monitor.yml` that runs before the capture. The loader inserts a label **only when the wallet has none**
and records a revision with actor `committed_evidence`. It never overwrites or revives anything set in admin: to
undo a committed label, relabel the wallet in admin (for example `Unknown`, low). Invalid evidence (bad address,
unknown label, non-HTTPS source, duplicate) fails the whole file.

Labels are recorded with each day's holder rows, so a new label takes effect at the **next capture**, not
retroactively; earlier cohort versions keep the labels of their own capture. Two exclusion paths:

- System labels (Treasury, Known Exchange, Liquidity Pool, DEX and the rest of `metrics.SYSTEM_LABELS`) at
  confirmed or high confidence are excluded everywhere, as before, including the securities adoption metrics.
  Adding one changes those metrics' exclusion fingerprint when the wallet holds a tracked security, and adoption
  windows that span the change are skipped until they fill again.
- **Market Maker** at confirmed or high confidence is excluded from the BP cohort only
  (`BP_COHORT_EXCLUDE_MARKET_MAKERS`, default on; `0` keeps them). Adoption metrics are unchanged.

Members removed this way leave with `exit_reason='excluded_by_label'`, and the next wallets by filtered rank enter
(within the entrant cap).

Reviewed 2026-09-27 against Solscan public tags, on-chain account owners and Backpack's published token
allocation. The file records each wallet's evidence.

| Rank (raw, 2026-09-27) | Wallet | Label | Evidence |
|---|---|---|---|
| 1 | `GySFHF...MHVH` | Treasury | Squads vault "BP Token"; exactly 750M BP, the two locked 375M allocations Backpack publishes |
| 2 | `EyzzXx...sJiS` | Treasury | Squads vault "BP"; multisig-executed distributions |
| 5 | `6qz7TH...wfjd` | Liquidity Pool | Account owned by the Meteora DLMM program (BP-USDC pool) |
| 6 | `ASTyfS...iaJZ` | Known Exchange | Solscan: MEXC exchange wallet |
| 7 | `43DbAv...pecN` | Known Exchange | Solscan: Backpack Exchange wallet |
| 16, 37, 85 | `D7BgNq...Qdwu`, `FCnqsz...cjzP`, `5a2HBB...1vgk` | Known Exchange | Solscan: JTX.com deposit addresses |
| 30 | `GpMZbS...xFbL` | Liquidity Pool | Solscan: Raydium Vault Authority #2 |
| 78 | `BM9Ccy...jvMN` | Market Maker | Program-derived, 177k token accounts, funded 1,023 accounts, most of the first major-sale alerts; operator unknown |
| 106 | `9xMB7V...xmFs` | Known Exchange | Solscan: Backpack Exchange deposit address |
| 109 | `WLHv2U...JVVh` | DEX | Solscan: Raydium Launchpad Authority |

Deliberately **kept** as investors: vaults of the Definitive program and a Fuse Squads vault (each is one user's
smart wallet), Pump.fun traders, `.sol`-named wallets and token creators. A large balance alone is never a reason
to exclude. **Not reviewed**: raw ranks 152 to 200 (Solscan's bot check stopped the review; it was not bypassed),
including rank 175 `7Yydv4GF2NyRb6x17c2odQBKStRwRg2p6MY9A3ZrF6Qf`, a program-derived address worth checking first.

## Milestone 1 results (probe run 36317286207, 2026-09-27)

Top 20 filtered holders from cohort version 1; 3 sampled for history. Read-only; 64 Helius RPC, 6 Enhanced and 131
Jupiter requests.

- **History endpoint: confirmed.** One sampled wallet had 250 transactions under `getTransactionsForAddress` with
  `tokenAccounts='balanceChanged'` versus 200 without it: 50 token-account-only transactions whose account keys do not
  include the owner (incoming SPL transfers). The Enhanced address endpoint returned none of those 50 and nothing the
  token-account query missed. For 3 token accounts the owner had since closed, their earlier activity was still returned
  when querying the owner (5 of 5 transactions). The other two wallets had no token-account-only activity in the window.
- **Enhanced API** still responds (200 transactions per sampled wallet), so the `parsed_swap` tier works for now; Helius
  lists it as in maintenance mode.
- **Transaction volume:** 80, 63 and 675 transactions in 30 days for the three sampled wallets (complete in two pages).
  The 30-day backfill is cheap for this cohort.
- **Portfolio sizes:** the largest of the top 20 had 6,330 token accounts (3.7 MB, 0.8 s); most had under 30. The
  production holdings run found 3 of 200 wallets over the 10,000-account limit.
- **Metadata:** DAS returned all 6,521 distinct mints of 10 portfolios (6,301 FungibleToken, 121 FungibleAsset, 93 NFTs).
  806 are Token-2022 mints; extensions seen: metadata 737, transfer fee 111, permanent delegate 30, transfer hook 21,
  confidential transfer 19, pausable 19, default account state 19, scaled UI amount 18, interest bearing 3.
- **Prices:** Jupiter priced 2,745 of 6,521 mints. The block-time sample failed with HTTP 429 because it sent 100 calls
  in one batch; fixed in PR #140 (batches of at most 8).
- **Swap programs:** swap-shaped transactions of the sample used Jupiter v6 (15), Meteora DLMM (15) and Raydium CLMM (3),
  plus programs not in `dex_programs.json`: `goonuddtQRrWqqn5nFyczVKaie28f3kDkHWkHtURSLE` (6),
  `HpNfyc2Saw7RKkQd8nEL4khUcuPhQ7WwY1B2qjx8jxFq` (3), `ZERor4xhbUycZ6gb9ntrhqscUcZmAbQDjEAtCf4hbZY` (2),
  `BiSoNHVpsVZW2F7rx2eQ59yQwKxzU5NvBcmKshCSUypi`, `TessVdML9pBGgG9yGks7o4HewRaXVAMuoVj4x83GLQH` and
  `61DFfeTKM7trxYcPQCM78bJ794ddZprZpAwAnLiwTpYH` (1 each). Direct swaps through these, without Jupiter or a Helius
  swap parse, land in `unclassified` until each is identified from a primary source and added.
- **Usage estimate caveat:** the probe's `block_time_calls_month` (4.7M) assumes one exact lookup per priced token per
  hourly run; it predates slot-based price ages and overstates that line. The hourly history poll alone
  (`history_poll_credits_month` 1.44M) exceeds the free plan, as documented above.
- **Webhooks:** not probed (would require registering one).

## Milestone 1 runbook

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
  Approve a version created after the committed labels took effect (source date 2026-09-28 or later), not version 1.
- **Label review of raw ranks 152 to 200** (see Committed wallet labels).
- **Milestone 4**: live ingestion decision and the seven-day pilot (gaps, classification accuracy,
  reconciliation, latency, actual cost). Nothing live is built.
- **Milestone 5**: alert enablement decision, monitoring, and the support runbook appended to
  `backpack-engineer-handoff.md`.
- Historical SOL valuation only exists from the first hourly run onward; backfilled SOL-leg trades before that
  stay unpriced. A timestamped SOL source would fill it.
- Related-wallet warnings are omitted (spec version 1).

## Verification (2026-09-26, local)

- `tests/test_bp_intel.py`: 37 fixture tests (every classifier case in section 13, portfolio parsing, cohort
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
- Existing Backpack suites unchanged and passing (126 Python tests in total). TypeScript: `npm run test:backpack`
  (13), `tsc`, targeted ESLint and `next build`.
- Local PostgreSQL was PGlite 0.5.8 (PostgreSQL 18, WASM); CI runs PostgreSQL 16.
