# BP Holder Intelligence: Senior Developer Handoff

Revision 2 (2026-09-26). Supersedes the first draft. The main change is that this is now specified as an
extension of the existing Backpack on-chain monitor (`backpack/`, `sql/backpack.sql`,
`docs/backpack-monitor.md`) rather than a standalone application. Roughly half of the first draft already
existed in that code; this revision says what to reuse, what to add, and which technical assumptions the
first draft got wrong.

Implementation status, deviations and the Milestone 1 runbook live in
[`bp-holder-intelligence.md`](bp-holder-intelligence.md). This file is the contract.

## 1. Objective

Identify the largest BP holders, inventory their other Solana token holdings, measure common ownership, and
detect subsequent purchases.

- Target network: Solana mainnet
- BP mint: `BPxxfRCXkUVhig4HS1Lh7kZqV6SPJhzfEk4x6fVBjPCy` (already registered in `backpack_assets` with
  `asset_type = 'bp'`; the schema enforces that exact mint)
- Initial cohort size: 200 owner addresses

The application must answer:

- Which wallets are the largest BP holders?
- Which other tokens do those wallets own?
- Which tokens are held by multiple wallets, and how significant are those positions?
- Which wallets are purchasing tokens they did not previously hold?
- Which tokens are attracting purchases from multiple tracked wallets?
- Is the group accumulating or selling a particular token?

This is an analytics and alerting application. Automated trading is outside scope.

## 2. Scope and interpretation

### Initial coverage

Include:

- Native SOL.
- Direct fungible-token holdings under the original SPL Token program and Token-2022.
- Successful on-chain swaps involving tracked wallets.
- Transfers, airdrops, and other balance-changing events as separately classified activity.
- Both the original holder cohort and a periodically refreshed cohort.

Exclude from initial portfolio totals:

- NFTs.
- Centralized-exchange account balances.
- Assets on other chains.
- Underlying assets deposited into lending, staking, liquidity, or other DeFi positions that require
  protocol-specific decoding.

Retain recognizable position tokens (LP tokens, receipt tokens), but label them and avoid double-counting
their underlying assets if protocol support is added later.

### Token-2022 caveats

Token-2022 extensions change what a balance means. Record the mint's extensions at metadata enrichment time
and apply these rules:

- **Confidential transfers:** the encrypted balance cannot be read. Store the public balance and mark the
  holding `balance_visibility = 'partial'`. Never report it as zero.
- **Interest-bearing mints:** the raw amount differs from the UI amount. Store raw and the interest-bearing
  display amount separately.
- **Transfer hooks and transfer fees:** swaps and transfers of these mints can include extra accounts and fee
  deductions. The classifier must not treat the fee leg as a separate sale.
- **Permanent delegate:** a balance can be moved without the owner signing. Outgoing transfers on such mints
  are labelled `delegate_transfer`, never `sale`.

A wallet is not necessarily a person. One person may control several wallets, and one custody wallet may
represent many people. All reporting must use "wallets" unless stronger attribution is available.

The application must not claim complete personal portfolios or lifetime purchase histories.

## 3. Architecture: build on the Backpack monitor

The first draft proposed FastAPI, a new PostgreSQL database, a new dashboard and a new data model. This
repository already has the storage, provider adapters, worker pattern and UI that the draft asked for.
Reuse them.

| Component | Status | Where |
| --- | --- | --- |
| Holder collector (paginated DAS `getTokenAccounts`, cursor-repeat guard, `last_indexed_slot` watermark, page budget, supply reconciliation) | Exists | `backpack/providers.py::Providers.holders`, `backpack/metrics.py::holders` |
| Owner aggregation and holder state | Exists | `backpack_current_holders`, `backpack_holder_events`, `backpack_holder_checkpoints` |
| Evidence-backed wallet labels with confidence and revision history | Exists | `backpack_wallet_labels`, `backpack_wallet_label_revisions`, admin flow at `/admin/backpack` |
| BP whale cohorts (new/exited, net accumulation) | Exists | `backpack/metrics.py::whale_cohorts`, `backpack_bp_whale_daily_snapshots` |
| Per-wallet Helius Enhanced history with cursors | Exists | `Providers.history`, `backpack_transactions`, `backpack_transaction_cursors` |
| Outer-swap normalization with USDC proxy valuation | Exists | `backpack/metrics.py::normalize_swap` |
| Per-wallet holdings across both token programs plus native SOL | Exists (TypeScript, request-time) | `apps/web/lib/server/trader-watch.ts` |
| Pricing (Jupiter Price V3 with block-derived timestamp), DexScreener labels | Exists | `Providers.price`, `Providers.validation_market` |
| Worker lease, request budget, per-run provider usage, immutable snapshots | Exists | `backpack_job_leases`, `backpack_provider_usage`, `backpack_ingestion_runs` |
| Cohort versioning | New | Section 5 |
| Portfolio collector (Python, stored) | New, port the Trader Watch logic | Section 6 |
| Overlap analytics | New | Section 7 |
| Economic-event classifier with confidence tiers | New, extends `normalize_swap` | Section 8 |
| Alert evaluator | New | Section 10 |
| UI sections and CSV routes | New, inside the existing Backpack page | Section 12 |

Runtime:

- **Workers:** Python, run by GitHub Actions like `backpack-monitor.yml`, using the existing lease, deadline
  and request-budget pattern. GitHub schedules are best-effort and most fires are dropped; the repo already
  works around this with a Vercel cron that presses `workflow_dispatch` (`/api/cron/dispatch-workflows`).
  Any schedule in this document is a target, and every stored observation carries its actual observation
  time.
- **Reads:** Next.js API routes on Vercel reading Neon, as the Backpack page does today. Web reads never
  execute DDL; the collector owns the schema (`backpack_monitor.py --migrate`).
- **Live ingestion:** there is no resident process in this stack. A webhook receiver can be a Vercel route
  that persists before acknowledging, but a continuously running consumer cannot. Live ingestion is
  therefore a Milestone 4 decision (section 9), not an assumed component.
- **Providers:** Helius (already configured via `HELIUS_API_KEY`), Jupiter (already configured),
  DexScreener (no key). Solscan Pro is an optional independent holder cross-check and is paid; do not make it
  a dependency. Record API usage per collection job in `backpack_provider_usage`.

Provider references: Helius DAS `getTokenAccounts`; Helius Enhanced Transactions by address; Helius
`getTransactionsForAddress`; Helius webhooks; Solana RPC `getTokenLargestAccounts`.

## 4. Holder discovery and ranking

### 4.1 Validate the target

At startup the existing collector already fetches the BP mint account, confirms the program and decimals, and
records finalized supply. Keep using the mint address as the identifier. Names, symbols and logos are display
metadata, not identifiers.

### 4.2 Discover all BP token accounts

Use `Providers.holders(mint)`. It pages Helius DAS `getTokenAccounts` with `showZeroBalance: false`,
`limit: 1000` and a cursor, detects repeated cursors, records the minimum and maximum `last_indexed_slot`
seen, and raises rather than returning a partial list when the page budget (`BACKPACK_MAX_HOLDER_PAGES`) is
exhausted.

Do not use `getTokenLargestAccounts` for a top-200 owner ranking: it returns only 20 token accounts.

For each token account the DAS response already carries the address, owner, mint, raw amount and frozen flag.
Keep the token account's parsed owner distinct from the program that owns the account at the Solana account
layer.

Aggregate every account belonging to the same owner:

    owner_bp_balance = sum(raw BP balances across that owner's accounts)

Sort by raw balance descending, with owner address as a deterministic tie-breaker.

### 4.3 Snapshot consistency

Paginated live reads are not an atomic snapshot. The existing check is the right one: reconcile the sum of
enumerated balances against finalized supply within `BACKPACK_SUPPLY_TOLERANCE` (default 0.1%). Withhold the
ranking when reconciliation fails. Store the collection start/end times and the slot range. Label a
reconciled ranking Estimated, as the monitor already does; it is an observation over a time window, not a
point-in-time state.

Persist failed or incomplete jobs in `backpack_ingestion_runs`; never publish them as complete rankings.

### 4.4 Classification

Reuse `backpack_wallet_labels`. Its label set (Backpack, Treasury, Custody, Market Maker, DEX, Liquidity Pool,
Lending Protocol, Bridge, Known Exchange, Protocol, Unknown) with confidence, source, `verified_at` and notes
covers the draft's needs. Add two labels: Vesting and Burn. Frozen accounts are a fact from section 4.2, not a
label.

Exclusion for the filtered ranking follows the monitor's rule: only labels with confirmed or high confidence
exclude. Medium and low labels are displayed but do not exclude. Do not classify an unidentified large wallet
as an exchange by balance size alone; the engineering handoff already records that zero holders were excluded
in the baseline and that attribution is incomplete.

Produce:

- **Raw top 200:** largest owners without analytical exclusions.
- **Filtered top 200:** largest eligible owners after confirmed/high exclusions.

If fewer than 200 eligible owners exist, use the actual count and show it in all denominators.

## 5. Cohort management

New tables `bp_cohorts` and `bp_cohort_members` (section 11). Cohort versions are immutable.

### Original cohort

The first approved filtered top-200 set is fixed for longitudinal analysis. Continue tracking its members
even if they sell all BP.

### Current cohort

Recompute the filtered top 200 daily from the same run that writes `backpack_current_holders`. Store each
version's membership, ranks, balances, exclusion fingerprint and effective time.

Membership changes are separate events: entered, left, rank changed.

**Churn control.** The rank-200 boundary will churn daily, and every entrant needs a 30-day history backfill
(section 8.1). Apply hysteresis: a wallet enters when it ranks 200 or better and leaves only when it ranks
below 220 on two consecutive runs. Cap entrants at 20 per run; any beyond the cap are queued for the next run
and the cohort version records that the cap bound. Both numbers are configurable.

A newly added member's existing tokens must never be reported as new purchases. Its backfill window starts at
entry time minus 30 days, and events before entry are stored with `pre_membership = true` and excluded from
cohort flow metrics.

Cross-time metrics must use a consistent cohort version. For the refreshed view, expose membership effects
separately from trading activity.

## 6. Portfolio collection

Port the request-time logic in `apps/web/lib/server/trader-watch.ts` to a stored Python collector. For each
tracked owner:

1. `getBalance` for native SOL.
2. `getTokenAccountsByOwner` with `jsonParsed` encoding, once per token program
   (`TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA` and `TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb`). This
   call is not paginated; it returns all accounts. Wallets with more than roughly 10,000 token accounts are
   exchange-like and should already be excluded; if one is encountered, mark the read oversized rather than
   truncating.
3. Aggregate balances by mint. Decimals arrive in the parsed response.
4. Enrich distinct mints with metadata (DAS `getAsset`, batched) and Token-2022 extensions.
5. Attach Jupiter prices with their block-derived timestamps; mark stale after one hour as the monitor does.

Cost: three RPC calls per wallet, about 600 per hourly refresh for 200 wallets, about 14,400 per day, plus
metadata and price calls for distinct mints. Record actuals per run.

Use raw integer amounts and Decimal arithmetic. Store decimals separately. Never use floats for quantities.

SOL and wrapped SOL remain distinguishable in raw data. Provide an optional combined exposure view without
counting wrapping as a purchase.

Capture per owner and mint: raw amount, decimals, contributing token-account addresses, observation time and
slot, price, price timestamp, provider, pricing status, estimated USD value, frozen status, balance visibility
(section 2), and spam/dust classification with reason.

Missing prices remain null. A failed program read makes the wallet's portfolio partial or unavailable; it
never implies a token is absent. Never convert missing prices or failed reads to zero.

Preserve all nonzero fungible balances. Apply display filters separately: all holdings, positions worth at
least $100, unpriced holdings, suspected spam/dust, SOL and stablecoins. The $100 threshold matches the
monitor's existing meaningful-holder threshold, is configurable, and applies only when an acceptable price
exists.

## 7. Overlap analytics

For cohort C, token T, and a specified portfolio snapshot:

    holder_count(T) = count(distinct owners in C with positive T balance)
    meaningful_holder_count(T) = count(distinct owners in C with qualifying T value)
    ownership_percentage(T) = holder_count(T) / cohort_size

Display portfolio collection coverage beside these metrics. If collection is incomplete, label counts as
observed lower bounds; never treat a missing wallet as a nonholder.

The overlap table includes: mint, name and symbol; positive-balance holder count; meaningful-position holder
count; percentage of cohort; combined estimated position value; median portfolio weight among holders;
largest owner's share of the group's token balance; new buyers over 1 hour, 24 hours and 7 days; gross
purchases, gross sales and net flow; price, liquidity and collection-quality indicators.

Portfolio weight uses the priced, in-scope portfolio as its denominator. Label it accordingly and disclose
unpriced holdings.

Default sort: meaningful-position holder count descending. Exclude BP from the default "other tokens" table.
Provide explicit toggles for SOL, stablecoins and suspected spam.

Common ownership must not be described as evidence of buying unless transaction history supports it.

## 8. Transaction history and purchase classification

### 8.1 Historical backfill

Backfill 30 days per tracked owner.

**Endpoint decision (Milestone 1).** The monitor uses Helius Enhanced `/v0/addresses/{address}/transactions`,
which returns parsed transactions with `events.swap` and `tokenTransfers`. Helius also offers the newer
`getTransactionsForAddress` RPC method with filters. Verify, against a real cohort wallet, which of the two
returns transactions that touch the wallet's token accounts when the owner address is not itself in the
account keys (incoming SPL transfers), and how each handles closed token accounts. Pick one, document the gap,
and do not assume either covers token accounts until observed.

**Per-wallet page cap.** Market-making and bot wallets can produce tens of thousands of transactions in 30
days. `Providers.history` already stops at `BACKPACK_MAX_TX_PAGES_PER_WALLET`. Keep that: when the cap is
hit, store the wallet's coverage as incomplete with the earliest retrieved timestamp, and surface it. Do not
let one wallet consume the run budget.

Store per-wallet coverage in `backpack_transaction_cursors` extended with: requested start/end, earliest
retrieved transaction, completion status, and known gaps.

An empty filtered page does not prove history is exhausted; only reaching a timestamp before the requested
start does.

### 8.2 Classification pipeline

For each transaction:

1. Save the raw provider payload with parser version.
2. Verify execution success (`transactionError` null, `meta.err` null).
3. Resolve account keys and token-account ownership.
4. Calculate owner-level token and SOL deltas from `preTokenBalances`, `postTokenBalances`, `preBalances` and
   `postBalances`.
5. Separate fees, rent, wrapping and other incidental movements.
6. Produce normalized economic events with a confidence tier.

**Confidence tiers.** The first draft made classification binary, which would push most aggregator routes into
"unclassified". Use three tiers:

| Tier | Evidence | Alert-eligible |
| --- | --- | --- |
| `parsed_swap` | Helius `type = SWAP` with `events.swap` listing token inputs and outputs (what `normalize_swap` already requires) | Yes |
| `inferred_swap` | Owner-level deltas show one asset down and another up in one transaction, and a known DEX or aggregator program (Jupiter, Raydium, Orca, Meteora, Pump AMM, Phoenix; maintained list with program IDs) is in the account keys | Yes, labelled as inferred |
| `unclassified` | Anything else with a balance change | No |

A positive balance change alone is never a purchase. Transfers in and out, wrapping and unwrapping,
recognized rewards, liquidity operations, failed transactions and delegate transfers are their own event
kinds.

A token-to-token swap is a disposal of the input and an acquisition of the output; preserve both sides. For
multi-hop routes count the economic input and final output; route legs are not positions (`normalize_swap`
already refuses to sum inner legs). If one transaction contains several independent swaps, preserve separate
events where decoding permits.

**Valuation at execution time.** Price the trade from its own counter-asset when possible: USDC or USDT at $1
(labelled Estimated, as today), or SOL at the Jupiter SOL price nearest the block time. Only trades with no
stable or SOL leg need a per-token historical price, and those remain unpriced until a timestamped
token-specific source is chosen. This removes the market-data provider as a blocker for most alerts.

### 8.3 Definition of "new"

Determine the pre-event balance from the transaction itself: `preTokenBalances` for the owner's accounts of
that mint. An associated-token-account creation instruction in the same transaction is direct evidence that
the position is new. Only when the owner holds the mint in an account not present in the transaction does the
classification fall back to the latest stored portfolio snapshot, and it is then marked
`pre_balance_source = 'snapshot'`.

Flags:

- **New position:** pre-event balance zero, purchase creates a positive balance.
- **First observed purchase:** no earlier purchase in available history.
- **Re-entry:** prior ownership and a later zero balance are both observed before the purchase.
- **First observed group purchase:** first purchase within the selected cohort and known history.
- **Recently launched token:** separate classification from mint and market history.

Never label an event "first purchase ever" with a 30-day backfill.

## 9. Live ingestion and recovery

Live ingestion is a Milestone 4 decision because this stack has no resident worker. Until then, freshness is
the hourly portfolio refresh plus history polling per wallet on each run.

If live ingestion is enabled, the design is:

- A Helius enhanced webhook registered for the cohort's owner addresses, delivered to a Vercel route that
  verifies the auth header and inserts the raw payload before returning 200.
- A worker run (dispatched or scheduled) that drains persisted deliveries, deduplicates by signature and event
  index, classifies, and evaluates alerts.

**Known gap, not fixable by registration.** An SPL transfer into a wallet references the destination token
account, not the owner, and a new associated token account cannot be registered before it exists. Incoming
airdrops and deposits into new accounts will be found by the portfolio refresh and history polling, not by the
webhook. State this on the status panel.

Required behaviour: authenticate deliveries, persist before acknowledging, deduplicate (Helius documents
possible duplicate deliveries), retry with backoff, keep per-wallet checkpoints, recover after outages through
history polling, reconcile portfolios daily.

Target schedule (best-effort under GitHub Actions; see section 3):

| Job | Target frequency |
| --- | --- |
| Portfolio refresh and history poll | Hourly |
| Full balance reconciliation | Daily |
| BP ranking and cohort refresh | Daily, in the existing 00:30 UTC run |
| Live webhook drain | Every 5 minutes, only if Milestone 4 enables it |

Alert on finalized events by default. Confirmed-only events may display as provisional. If provisional alerts
are enabled later, support correction or retraction.

## 10. Alert rules

Record every classified purchase. Apply thresholds only to notifications.

| Rule | Initial condition |
| --- | --- |
| New position | At least $250 acquired into a previously zero balance |
| Multiple buyers | At least 3 distinct tracked wallets purchase the same mint within 1 hour |
| Accumulation | At least 5 buyers within 24 hours, with positive net purchase flow |
| Major sale | A sale disposes of at least 50% of the pre-sale holding, taken from `preTokenBalances` |

Multi-wallet rules initially require at least $250 cumulative qualifying purchases per wallet within the
window. Count a wallet once per mint and window. `inferred_swap` events count but the alert states how many of
its buyers are inferred rather than parsed.

USD thresholds require a valuation from section 8.2. Otherwise record the event as unpriced and exclude it
from USD-qualified alerts.

Each alert includes: rule and cohort version; mint and display name; wallet count and addresses; quantities
and valuation status; window start and end; transaction links; lowest classification tier among contributing
events; finality and data freshness.

**Related-wallet warnings.** The first draft promised these with no mechanism. Version 1 omits the field.
Version 2 may add a minimal heuristic: two cohort wallets whose first SOL funding came from the same address
within 24 hours, or that transferred tokens directly to each other in the history window, are flagged
`possibly_related` with the evidence signature. It is a warning, never a merge.

Use cooldowns and stable alert keys (rule + mint + window start + cohort version). Update an existing alert
when more buyers join instead of emitting a new one.

Keep alerts in the application. External delivery is a separate integration.

## 11. Data model

Reuse: `backpack_assets` (BP row), `backpack_current_holders`, `backpack_holder_events`,
`backpack_holder_checkpoints`, `backpack_wallet_labels` and its revisions, `backpack_transactions`,
`backpack_transaction_cursors`, `backpack_ingestion_runs`, `backpack_provider_usage`, `backpack_job_leases`,
`backpack_market_prices`.

New tables, collector-owned, additive, in `sql/backpack.sql`:

| Table | Purpose |
| --- | --- |
| `bp_cohorts` | Version id, kind (original/current), effective time, run id, exclusion fingerprint, size, entrant cap applied |
| `bp_cohort_members` | Cohort version, wallet, rank, raw BP balance, entered/left/rank-change event |
| `bp_tracked_assets` | Mint, program, decimals, Token-2022 extensions, metadata, spam classification |
| `bp_portfolio_balances` | Owner, mint, run id, raw amount, decimals, contributing accounts, slot, price fields, visibility, read status |
| `bp_economic_events` | Transaction, owner, event index, kind, tier, input/output mint and raw amounts, USD and valuation source, pre-balance and its source, flags from section 8.3, parser version |
| `bp_raw_transactions` | Signature, slot, finality, raw payload, source, received/ingested times |
| `bp_alert_rules`, `bp_alerts` | Configuration and evaluated alerts with stable keys |

Constraints:

- Transaction identity: network plus signature.
- Asset identity: network plus mint; native SOL is an explicit row.
- Economic-event identity: signature, owner and deterministic event index. Never deduplicate by signature
  alone when several tracked wallets or swaps share a transaction.
- Keep raw payloads and parser versions so events can be reprocessed.
- Keep API keys out of logs, stored request URLs and exports (the monitor already strips them).

## 12. Dashboard and exports

Do not build a new dashboard. Add sections to the existing Backpack page at `/market/crypto/backpack`, which
is the authorized crypto exception to the one-screen rule, and read through `/api/market/crypto/backpack` like
the current sections do:

- **Holder roster:** BP ranking, labels, exclusions, cohort version and membership changes.
- **Common holdings:** the section 7 table with coverage indicators.
- **Recent purchases:** buyers, quantities, tier, and transaction evidence.
- **Token detail:** tracked holders, position changes, purchases, sales and alerts for one mint.

Every section shows the selected cohort version and data freshness. CSV export is a `format=csv` parameter on
the same read routes for holder rankings, wallet holdings, overlap results, activity and alerts.

Show a prominent status when collection is incomplete, pricing is stale, a wallet's history is capped, or live
monitoring is disabled or disconnected.

## 13. Validation and acceptance criteria

### Holder ranking

- All available BP account pages are processed and reconcile to supply.
- Multiple accounts belonging to one owner aggregate correctly.
- Rankings are reproducible from stored observations.
- Exclusions have evidence and remain inspectable.
- Partial collections cannot appear as completed results.

### Cohort and portfolio

- Hysteresis and the entrant cap behave as specified and are recorded on the version.
- Both token programs are covered; Token-2022 extension cases are covered by fixtures.
- Raw quantities retain precision.
- Unknown prices, partial visibility and failed reads remain distinguishable from zero.
- Every overlap count traces to owner/mint balances.
- Membership changes do not create artificial purchases.

### Transaction classification

Fixtures: direct and multi-hop swaps; token-to-token swaps; an aggregator route with no `events.swap` (must
land in `inferred_swap`); incoming transfers and airdrops; SOL wrapping and unwrapping; account creation and
closure; liquidity operations; failed transactions; several tracked wallets in one transaction; several swaps
in one transaction; transfer-fee and transfer-hook mints; a permanent-delegate transfer; duplicate and
out-of-order deliveries; unsupported transactions.

No transfer-only fixture may generate a confirmed-buy alert. Tier is visible on every event.

Manually inspect a representative live sample against transaction evidence before enabling alerts.

### Operational reliability

- Re-running a backfill or draining the same deliveries twice does not duplicate events.
- A simulated outage is recovered through history polling.
- Reconciliation discrepancies produce a visible diagnostic.
- Throttling and page caps never lose work silently; coverage rows say what was skipped.
- If live ingestion is enabled, measure end-to-end alert latency and report the 95th percentile after
  finalization; the two-minute target from the first draft only applies if a resident consumer exists.

## 14. Delivery sequence

- **Milestone 1: Data feasibility.** Verify, with real calls and a written result: which history endpoint
  covers token-account transactions and closed accounts (section 8.1); whether Helius webhooks match owner
  addresses or only account keys (section 9); `getTokenAccountsByOwner` response sizes for the current top
  holders; DAS metadata and Token-2022 extension coverage; Jupiter price availability for the distinct mints in
  a sample of ten portfolios. Produce the API usage estimate and the limitations list.
- **Milestone 2: Snapshot analysis.** Cohort tables, portfolio collector, overlap report, exclusions and
  collection-quality report, rendered in the Backpack page.
- **Milestone 3: Historical activity.** 30-day backfill with page caps, the tiered classifier, purchase and
  sale analytics, manual validation.
- **Milestone 4: Live decision and pilot.** Decide whether live ingestion is worth a hosted consumer or
  whether hourly polling is sufficient. Run seven days either way, measuring gaps, classification accuracy,
  reconciliation differences, latency and actual API cost.
- **Milestone 5: Production readiness.** Enable selected alerts, operational monitoring, restart recovery and
  a support runbook, appended to `docs/backpack-engineer-handoff.md`.

**Definition of done:** a reviewer can select a cohort version, inspect common holdings, open the transactions
behind a purchase alert, see its classification tier, and determine whether the underlying data is complete
and current.

## 15. Decisions and defaults

Decided in this revision:

- Extend the Backpack monitor; no new service, database or dashboard.
- GitHub Actions workers plus Vercel read routes; live ingestion deferred to Milestone 4.
- Reuse the existing label table and confirmed/high exclusion rule.
- Three-tier classification; inferred swaps are alert-eligible and labelled.
- Execution-time valuation from the counter-asset first.
- Pre-event balances from `preTokenBalances`, snapshot fallback labelled.
- Hysteresis 200/220 and 20 entrants per run.
- Related-wallet warnings omitted from version 1.

Defaults carried over: 200 filtered owners with the raw ranking retained; original and daily refreshed
cohorts; direct Solana holdings only; 30-day window; $100 meaningful-position threshold; $250 notification
threshold; finalized-event alerts; seven-day pilot.

Open before production: monthly API budget, whether to host a live consumer, external alert destination, and
any DeFi-position coverage.

No claim of complete wallet ownership, lifetime transaction coverage, or profitable trading signals is part of
this specification.
