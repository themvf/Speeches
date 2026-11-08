# Portfolio workspace: product and implementation specification

Status: **implementation specification; the current holdings tracker is the baseline, and additions marked Planned are not yet implemented.**

Date: 2026-11-08

## 1. Outcome

The portfolio workspace lets its single owner create multiple hypothetical investment portfolios, record positions and transactions, and return later to review value, performance, concentration, and changes since entry. Cross-device access uses a private device-pairing link rather than an account or sign-in flow.

The experience must answer these questions without implying brokerage-grade precision when the required data is missing:

1. What portfolios have I saved, and what does each contain?
2. How much did I invest, what is it estimated to be worth now, and what changed today?
3. Which positions contribute most to gain, loss, and concentration?
4. What transactions, prices, and assumptions produced those results?
5. How current and complete are the displayed prices?
6. Can I recover, export, or reconcile my data?

This is a review and research tool, not a brokerage account, tax-lot system, order-entry system, or investment recommendation.

## 2. Product principles

- **Never substitute a partial total for a complete portfolio total.** If one required quote is unavailable, label the aggregate market value and return as incomplete or unavailable and identify the affected positions.
- **Keep facts, estimates, and assumptions distinct.** User-entered transactions are facts supplied by the user. Prices are sourced observations. Returns, tax lots, and derived allocations are calculations with declared assumptions.
- **Preserve the ledger.** Editing the current number of shares must not silently rewrite transaction history once transaction tracking is enabled.
- **Autosave safely.** Local persistence should feel immediate; remote writes should be debounced and occur only when data changed.
- **Keep costs proportional to value.** Viewing or refreshing quotes must not write to Neon. Historical snapshots must be batched, not produced by every browser refresh.
- **Design for failure.** A quote failure, database cold start, offline session, provider rate limit, or cross-device conflict must have a visible, recoverable state.
- **Do not overstate market-data quality.** Display the provider, observation time, market session, and delay/availability limitation.

## 3. Existing baseline and additions

| Capability | State | Required change |
| --- | --- | --- |
| Create and switch named portfolios | Implemented | Preserve; add archive, duplicate, and recovery |
| Add ticker, shares, trade price, optional trade date | Implemented | Evolve into transaction entry without losing the simple form |
| Edit shares and trade price inline | Implemented | Replace balance rewriting with ledger-safe adjustment flows |
| Remove a ticker | Implemented | Add undo; transaction mode records an exit or deletes only an erroneous draft |
| Local browser autosave | Implemented | Preserve; expose pending/offline state |
| Passwordless private-link Neon sync | Implemented | Add link rotation, recovery/export, and version-aware conflict handling |
| Latest Yahoo quote lookup | Implemented | Place behind a provider interface; confirm production licensing and service limits |
| Cost basis, market value, unrealized gain/loss, daily change | Implemented for simple holdings | Add fees, sales, realized return, dividends, splits, and completeness metadata |
| Portfolio treemap | Implemented | Add allocation modes and accessible tabular equivalent |
| Multiple purchase lots | Partially supported as duplicate ticker rows | Make lots explicit and reconcile them through transactions |
| Automatic price refresh | Implemented | Preserve visibility-aware, market-aware five-minute refresh and measure provider usage |
| Historical performance | Not implemented | Add daily snapshots or reconstruct from transactions and historical prices |
| Import/export | Not implemented | Add CSV import, validation preview, error report, and CSV/JSON export |
| Alerts and targets | Not implemented | Add after core accounting is reliable |

## 4. Navigation and information architecture

Primary route: `/portfolios`.

Desktop layout:

- Left rail: portfolio list, portfolio counts, create action, archived view.
- Main header: active portfolio name, persistence status, last price update, primary actions.
- Summary: deployed capital, open-position market value, realized and unrealized gain/loss, today's change, and data completeness.
- Positions: sortable holdings ledger with inline drill-down.
- Portfolio map: value/concentration visualization.
- Activity: transaction ledger and filters.
- Analysis: allocation, historical performance, benchmark, and realized/unrealized attribution.

Mobile layout:

- Portfolio selector replaces the persistent left rail.
- Summary cards appear before the positions table.
- Positions render as accessible cards or a constrained table with no page-level horizontal scrolling at 320px.
- Primary add-transaction control remains reachable without scrolling through the full portfolio.
- Tap targets are at least 44px; important state is never conveyed by color alone.

URL state should preserve the selected portfolio and active section where safe:

```text
/portfolios?portfolio=<id>&view=positions
/portfolios?portfolio=<id>&view=activity
/portfolios?portfolio=<id>&view=analysis&period=1y
```

Do not expose a user ID or sequential database identifier in a shareable URL.

## 5. Portfolio management

### 5.1 Create

- Name is required, whitespace-normalized, and limited to 80 characters.
- Duplicate names are allowed but the UI should warn because they are difficult to distinguish.
- Optional fields: description, benchmark, and tags.
- Phase 1 is USD-only. The UI rejects non-USD instruments and transactions until multi-currency valuation is explicitly implemented.

### 5.2 Rename, duplicate, archive, and delete

- Rename is non-destructive and autosaves.
- Duplicate copies portfolio settings and, by user choice, either current positions or full transaction history.
- Archive hides a portfolio from the default list without deleting it.
- In Phase 1, every portfolio, including a local-only IndexedDB portfolio, enters Trash for 30 days. Setting `deleted_at` on the parent hides its children and produces a tombstone that synchronizes to other devices.
- Restore clears the tombstone. A scheduled purge removes child transactions, snapshots, import artifacts, manual valuations, mutation receipts, and the portfolio after 30 days.
- Phase 0 localStorage supports immediate Undo but not durable 30-day Trash.
- The immediate UI offers Undo after removing a position, transaction, or portfolio.
- Permanent deletion requires explicit confirmation and applies only to the selected object.

### 5.3 Limits

Initial guardrails:

- 50 active portfolios per user.
- 200 open positions per portfolio.
- 10,000 transactions per portfolio before the UI requires pagination and a reviewed higher limit.
- 100 unique symbols per quote request.

Limits must be enforced on both client and server. Reaching a limit must not corrupt or truncate previously saved data.

## 6. Transactions and accounting

### 6.1 Transaction types

Required transaction ledger types:

- Buy
- Sell
- Dividend
- Fee
- Stock split or reverse split
- Manual adjustment, with required reason

Each transaction records portfolio, instrument identity, trade date, optional settlement date, quantity, price, fees, currency, source, notes, and creation/update timestamps. The workspace does not model deposits, withdrawals, buying power, sale-proceeds balances, margin, or brokerage cash.

Transaction effects are deterministic:

| Type | Quantity effect | Basis/proceeds effect | Strategy-result treatment | Required specialized data |
| --- | --- | --- | --- | --- |
| Buy | Increase by quantity | Add `quantity × price + fees` to lot basis | Establishes deployed capital; not itself profit/loss | Quantity, price |
| Sell | Decrease by quantity | Net proceeds = `quantity × price - fees`; remove allocated FIFO lot basis | Realized gain = net proceeds − allocated basis | Quantity, price; available quantity |
| Dividend | None | No position-basis change | Net dividend is strategy income; show gross and withholding separately | Gross amount and any withholding |
| Fee | None | No position-basis change | Standalone fee reduces strategy result | Amount and reason |
| Split/reverse split | Multiply quantity by numerator/denominator | Total basis unchanged; per-share basis changes | Not profit/loss | Positive numerator and denominator |
| Adjustment | Explicit signed quantity and/or basis effect | Explicit signed basis effect | Determined by declared reason; always identified as an adjustment | Quantity/basis effects and required reason |

Buy and sell fees are stored only on their buy or sell transaction. A standalone `FEE` is an independent strategy expense and never retroactively changes another transaction's basis or proceeds.

### 6.2 Instrument identity

A ticker alone is not a durable global identifier. The target model should retain:

- Display symbol
- Instrument name
- Exchange or venue when known
- Asset type
- Quote-provider identifier
- Trading currency
- Optional FIGI, CUSIP, ISIN, or provider-specific ID when lawfully available

Symbol changes must not create a false sale and repurchase. Delisted or unsupported securities remain in the ledger with an unavailable quote state.

### 6.3 Position derivation

Positions are derived from transactions; they are not independently authoritative balances.

```text
open quantity = buys + adjustments in - sells - adjustments out

market value = open quantity × latest usable price
```

- The positions table shows weighted-average entry price for orientation.
- Lot-level records remain available for reconciliation.
- FIFO is the initial realized-gain estimate unless the user selects a supported alternative. Label it as an estimate and do not represent it as tax advice.
- A sale cannot exceed available quantity unless short positions are explicitly enabled later.
- Fees affect invested cost and realized return according to documented rules.
- Dividends are included in total return but shown separately from price return.
- Phase 1 rejects non-USD instruments and transactions; it never silently converts currencies.
- Every nonzero open position requires a usable USD quote or an explicit current manual valuation. There is no materiality threshold. If any position is missing, the portfolio total is incomplete and only a clearly labeled priced subtotal may be shown.

### 6.4 Corporate actions

Initial release supports manual split entry and manual dividend entry. Automated corporate actions require a validated provider and reconciliation process.

- Splits change quantity and per-share basis without changing total basis.
- Symbol changes retain instrument continuity.
- Mergers, spin-offs, return of capital, and option events are out of scope until specific accounting rules and evidence sources are approved.
- Every automated corporate action must retain source, effective date, observed date, and whether a user confirmed it.

### 6.5 Transaction lifecycle and corrections

- Draft transactions are editable and removable.
- Posted transactions are append-only.
- Correcting a posted transaction atomically posts a reversing transaction and, when needed, a replacement transaction.
- After the reversing transaction commits, the original row becomes `reversed`; the reversing transaction remains `posted` and links to the original through `reverses_transaction_id`.
- An individual posted transaction is never soft- or hard-deleted.
- A backdated posting, reversal, or replacement queues a Phase 2 snapshot rebuild from its effective date through the present.

### 6.6 Return calculations

Required metrics:

- Invested cost
- Current market value
- Unrealized gain/loss
- Realized gain/loss
- Dividends and other income
- Fees
- Today's price change
- Total strategy profit/loss
- Open-position return percentage
- Realized return per closed lot or trade
- Unitized strategy return once a validated capital-normalization method exists

```text
total strategy P/L = realized gain/loss
                   + unrealized gain/loss
                   + net dividends
                   - standalone fees
```

Buy fees are already included in lot basis and sell fees are already deducted from proceeds; do not subtract them again. Because the workspace deliberately has no cash balance or contribution ledger, Phase 1 must not display a portfolio-level time-weighted return, money-weighted return, or a misleading percentage based on cumulative purchases. Validate every displayed result against hand-worked fixtures covering buys, partial sells, dividends, fees, splits, missing prices, and non-trading days. A later unitized strategy-return model requires its own approved definition and validation fixtures.

## 7. Price data and refresh behavior

### 7.1 Provider boundary

All quote access goes through a server-side provider interface:

```ts
type QuoteObservation = {
  instrumentId: string;
  symbol: string;
  price: number;
  currency: string;
  absoluteChange: number | null;
  percentChange: number | null;
  marketState: "pre" | "open" | "after" | "closed" | "unknown";
  asOf: string;
  provider: string;
  delayedByMinutes: number | null;
};
```

The current Yahoo adapter is provisional. Before production use, verify licensing, redistribution rights, reliability, rate limits, supported instruments, delay semantics, and attribution requirements. The provider must be replaceable without changing portfolio accounting or UI contracts.

### 7.2 Refresh policy

- Fetch on initial portfolio load, portfolio switch, new instrument, and manual refresh.
- Optional automatic refresh every five minutes while the tab is visible.
- Pause background refresh when `document.visibilityState !== "visible"`.
- During closed markets, reduce refresh frequency or stop after one closing-price confirmation.
- Do not poll a portfolio with no priced instruments.
- A manual refresh respects the shared quote cache; it does not guarantee a new upstream observation.
- Cache quotes for 60 seconds initially, with stale-while-revalidate for a short provider-approved interval.
- Deduplicate quote requests by normalized provider instrument ID, not only ticker text.

One browser refresh should produce one portfolio-quote API invocation, not one Vercel invocation per ticker. The server may fan out bounded upstream calls or use a provider batch endpoint.

### 7.3 Completeness and stale states

Every response includes:

- Requested instrument count
- Priced instrument count
- Missing instruments and reasons
- Oldest and newest observation time
- Provider status
- Whether cached or freshly fetched

Aggregate open-position market value and performance are complete only when every nonzero open position has a usable same-currency quote or an explicit manual valuation. There is no materiality threshold. If incomplete:

- Show the known subtotal only when clearly labeled **Priced subtotal**.
- Do not display it as total market value.
- List missing positions.
- Preserve cost basis and saved transactions.

A manual valuation is persisted independently from provider metadata:

```ts
type ManualValuation = {
  id: string;
  portfolioId: string;
  instrumentId: string;
  price: string; // serialized decimal, not a floating-point database value
  currency: "USD";
  asOf: string;
  validUntil: string;
  note: string;
  createdAt: string;
  supersededAt: string | null;
};
```

The latest non-superseded, unexpired valuation is used. It is always labeled **Manual valuation** with its `asOf`, age, and `validUntil` time and is never silently presented as a live quote. Phase 1 requires `validUntil > asOf` and limits validity to 90 days. After expiry, the position is unpriced until the user creates or reconfirms a valuation; reconfirmation creates a new row and supersedes the old one. A manual valuation must be valid on the snapshot date to complete that snapshot. Automated alerts do not trigger from manual valuations in the initial alert release.

## 8. Analysis features

### 8.1 Portfolio map

- Size by market value, cost basis, or portfolio weight.
- Color by total return, today's return, or neutral allocation category.
- Group by sector, industry, asset type, or no group.
- Provide a table containing the same information for keyboard and screen-reader users.
- Unknown classifications remain **Unclassified** rather than being guessed.

### 8.2 Allocation

- Ticker, sector, industry, and asset-type weights.
- Target allocation and drift when the user defines targets.
- Concentration measures: largest position, top-five weight, and optional Herfindahl index with an explanation.
- Classification source and as-of date must be available.

### 8.3 History and benchmark

- Periods: 1M, 3M, YTD, 1Y, 3Y, 5Y, and maximum available.
- Chart open-position market value, cumulative realized/unrealized strategy P/L, and the selected benchmark separately.
- Benchmark comparison must align dates, currency, and market calendars.
- Missing history is shown as a gap; do not forward-fill across unexplained data outages without labeling it.
- A daily portfolio snapshot is sufficient for the initial chart. Intraday history is not required.
- Phase 2 uses `America/New_York` and US market dates initially. Create at most one snapshot per portfolio after the official close; do not create weekend or market-holiday rows.
- Store incomplete valuations with completeness metadata, but exclude them from performance chaining.
- A backdated posted or reversed transaction queues a coalesced rebuild from its effective date through the present.
- Retain snapshots while the portfolio exists and through its 30-day Trash period. A chart may visually carry the last complete value only when labeled; the database does not manufacture daily rows.

Transactions can reconstruct historical quantities, but valuation also requires historical prices and repeated calculation. Daily snapshots keep chart reads bounded and preserve the calculation version and completeness state that were known at the time.

### 8.4 News and research integration

Optional later addition:

- Show existing Policy Research Hub articles and regulatory events relevant to held symbols.
- Keep news relevance separate from price causation.
- Never state that an article caused a move without supporting evidence.

## 9. Import, export, and reconciliation

### 9.1 CSV import

- Provide a documented template and common column aliases.
- Upload opens a preview; it does not immediately mutate the portfolio.
- Validate date, symbol, action, quantity, price, fees, and currency.
- Show accepted rows, rejected rows, warnings, duplicates, and resulting position changes.
- The user confirms the final import once.
- Import is idempotent through a source row hash or external transaction ID.
- Produce a downloadable error report without including other users' data.

Broker-specific importers are separate adapters. Do not claim brokerage support until sample statements and reconciliation cases are verified.

### 9.2 Export

- Export portfolios, positions, and transactions to CSV.
- Export a complete JSON backup with schema version and generated timestamp.
- Export should not require current quotes.
- Restore from JSON must preview changes and reject an unsupported future schema version.

### 9.3 Reconciliation

- Show calculated quantity against an optional user-entered statement quantity.
- Identify mismatches without silently adjusting the ledger.
- Manual adjustments require a reason and remain auditable.

## 10. Alerts and targets

Planned after accounting and price reliability are validated:

- Price above/below threshold
- Position gain/loss threshold
- Portfolio weight above/below target
- Target-allocation drift
- New relevant regulatory or news event

Alert evaluation must run in a scheduled server job, not in an open browser tab. Deduplicate repeated alerts, store the triggering observation, and enforce per-user and global limits. Email, SMS, push, or third-party delivery requires separate authorization and cost review.

## 11. Persistence architecture

### 11.1 Local persistence

- IndexedDB is preferred once the transaction ledger grows; localStorage may remain for the current small holdings payload and migration marker.
- In Phase 1, IndexedDB is a full local replica containing portfolios, user-relevant instruments, transactions, manual valuations, tombstones, and a mutation outbox.
- Every outbox operation has a stable mutation UUID and the portfolio's last known server `baseVersion`.
- Local writes occur immediately after a valid change.
- The UI displays `Saved locally`, `Syncing`, `Synced`, `Offline`, or `Sync error`.
- A remote error must not discard local changes.
- Cross-tab changes use a broadcast or storage event and avoid overwrite loops.
- On first device pairing, load the remote workspace before creating starter data. A new device must never overwrite an existing workspace with an empty starter portfolio. Future ledger migration unions records by stable UUID, never by name or client timestamp. Equal content deduplicates; divergent content with the same UUID creates a visible conflict. Tombstones participate in the merge.
- An unresolved local portfolio is retained as a clearly labeled recovered copy rather than discarded. First-pairing import is transactional and idempotent.
- On a `409`, pause only the affected portfolio's outbox. Pull server state and automatically rebase non-overlapping appended transactions; require user resolution for overlapping metadata edits or competing corrections of the same transaction.

### 11.2 Remote persistence

This personal deployment does not require an identity provider. The browser creates a 256-bit random workspace token and stores it locally. `Connect another device` shares that token in a URL fragment, which is not included in the page request; the receiving browser stores it and removes the fragment from the visible URL. API requests send the token in a custom header over HTTPS. The server validates the token, stores only its SHA-256-derived workspace identifier, and uses that identifier as the owner key. Anyone holding the private link can access and replace the workspace, so the link must be treated like a password. There is no server-side recovery if every paired device and every saved link is lost. Use Next.js server route handlers with the Neon serverless driver; do not expose database credentials or a generic database API to the browser.

Target normalized schema:

```sql
CREATE TABLE portfolios (
  id UUID PRIMARY KEY,
  user_id TEXT NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  base_currency TEXT NOT NULL DEFAULT 'USD',
  benchmark_instrument_id UUID,
  version BIGINT NOT NULL DEFAULT 1,
  created_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL,
  archived_at TIMESTAMPTZ,
  deleted_at TIMESTAMPTZ
);

CREATE INDEX portfolios_user_active_idx
  ON portfolios (user_id, updated_at DESC)
  WHERE deleted_at IS NULL;

CREATE TABLE instruments (
  id UUID PRIMARY KEY,
  symbol TEXT NOT NULL,
  name TEXT,
  exchange TEXT,
  asset_type TEXT,
  currency TEXT NOT NULL DEFAULT 'USD',
  provider TEXT,
  provider_id TEXT,
  active BOOLEAN NOT NULL DEFAULT TRUE,
  UNIQUE (provider, provider_id)
);

CREATE TABLE portfolio_transactions (
  id UUID PRIMARY KEY,
  portfolio_id UUID NOT NULL REFERENCES portfolios(id),
  instrument_id UUID REFERENCES instruments(id),
  type TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'draft',
  trade_date DATE NOT NULL,
  settlement_date DATE,
  quantity NUMERIC,
  price NUMERIC,
  fees NUMERIC NOT NULL DEFAULT 0,
  basis_amount NUMERIC,
  gross_amount NUMERIC,
  withholding_tax NUMERIC,
  split_numerator NUMERIC,
  split_denominator NUMERIC,
  acquired_at DATE,
  transfer_value NUMERIC,
  reverses_transaction_id UUID REFERENCES portfolio_transactions(id),
  currency TEXT NOT NULL DEFAULT 'USD',
  source TEXT NOT NULL DEFAULT 'manual',
  external_id TEXT,
  notes TEXT,
  details JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL,
  UNIQUE (portfolio_id, source, external_id)
);

CREATE INDEX portfolio_transactions_ledger_idx
  ON portfolio_transactions (portfolio_id, trade_date, created_at)
  WHERE status IN ('posted', 'reversed');

CREATE TABLE portfolio_manual_valuations (
  id UUID PRIMARY KEY,
  portfolio_id UUID NOT NULL REFERENCES portfolios(id),
  instrument_id UUID NOT NULL REFERENCES instruments(id),
  price NUMERIC NOT NULL CHECK (price > 0),
  currency TEXT NOT NULL DEFAULT 'USD',
  as_of TIMESTAMPTZ NOT NULL,
  valid_until TIMESTAMPTZ NOT NULL,
  note TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  superseded_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX portfolio_manual_valuations_current_idx
  ON portfolio_manual_valuations (portfolio_id, instrument_id)
  WHERE superseded_at IS NULL;

CREATE TABLE portfolio_mutations (
  user_id TEXT NOT NULL,
  mutation_id UUID NOT NULL,
  portfolio_id UUID NOT NULL REFERENCES portfolios(id),
  base_version BIGINT NOT NULL,
  resulting_version BIGINT NOT NULL,
  result_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (user_id, mutation_id)
);

CREATE INDEX portfolio_mutations_portfolio_idx
  ON portfolio_mutations (portfolio_id, created_at);

CREATE TABLE portfolio_daily_snapshots (
  portfolio_id UUID NOT NULL REFERENCES portfolios(id),
  snapshot_date DATE NOT NULL,
  market_value NUMERIC,
  open_cost_basis NUMERIC,
  cumulative_realized_gain NUMERIC,
  cumulative_unrealized_gain NUMERIC,
  cumulative_dividend_income NUMERIC,
  cumulative_standalone_fees NUMERIC,
  cumulative_total_pl NUMERIC,
  priced_position_count INTEGER NOT NULL,
  total_position_count INTEGER NOT NULL,
  valuation_status TEXT NOT NULL,
  missing_position_count INTEGER NOT NULL DEFAULT 0,
  missing_instrument_ids UUID[] NOT NULL DEFAULT '{}',
  missing_reasons JSONB NOT NULL DEFAULT '{}'::jsonb,
  valuation_timezone TEXT NOT NULL DEFAULT 'America/New_York',
  price_as_of TIMESTAMPTZ,
  calculation_version TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (portfolio_id, snapshot_date)
);
```

Add ownership checks to every query: a request may only read or mutate portfolios whose owner key matches the SHA-256-derived identifier of its presented workspace token. Child-row access must join through the owned portfolio. Rate-limit sync requests by client IP and never log the raw token.

The server validates transaction rows as a discriminated union by `type`: it rejects irrelevant or missing specialized fields, unsupported status transitions, non-USD Phase 1 data, and sale quantities exceeding available lots. `details` is for bounded provider/import metadata, not accounting values. Superseding and inserting a manual valuation occur in one transaction so the partial unique index cannot leave two current rows.

A mutation receipt and its portfolio/ledger changes commit in the same database transaction. Receipts are retained until the portfolio's permanent purge, making retries durable across Vercel instances, restarts, and Neon suspension. Mutation operations represent meaningful user actions or safely batched offline operations, not individual keystrokes; `result_json` is bounded to the reconciliation result required for an idempotent retry.

### 11.3 Migration from current JSON storage

The current `user_portfolios` JSON row remains readable during migration.

1. Add schema migrations outside request handlers.
2. On first paired access, read the existing JSON payload. The original browser migrates its legacy local payload into the new token-scoped workspace; a newly paired device prefers non-empty remote data.
3. Convert each holding into a synthetic opening buy transaction using its shares, trade price, and trade date or original creation date. No cash balance is created or inferred.
4. Show a migration preview when a date or identity is ambiguous.
5. Write normalized rows in one transaction.
6. Mark the source row with migration version and timestamp.
7. Compare derived positions to the original holdings before switching reads.
8. Retain a recoverable backup for the declared migration window.
9. Preserve or generate stable UUIDs idempotently. If local and remote records both exist, apply the UUID union, tombstone, and visible-conflict rules rather than choosing the newest payload timestamp.

Do not silently migrate invalid holdings or invent missing dates without labeling the assumption.

### 11.4 Conflict handling

- Use portfolio `version` for optimistic concurrency.
- Every mutation includes stable `mutationId` and the last known `baseVersion`. A successful response returns `newVersion`.
- Apply a mutation ID at most once. Safe retries return the prior committed result rather than duplicating a transaction.
- If versions differ, return `409 Conflict` with current version and a stable conflict classification.
- Transactions use stable client-generated UUIDs so offline retries remain idempotent.
- Never use only a client timestamp as the final conflict arbiter; clocks can differ.

Example: devices A and B both read portfolio version 7. A's mutation commits first and returns version 8. B's version-7 mutation receives `409` instead of silently overwriting A. The client may rebase a non-overlapping append, but an overlapping edit requires user resolution.

## 12. API contracts

Proposed endpoints:

```text
GET    /api/portfolios
POST   /api/portfolios
GET    /api/portfolios/:portfolioId
PATCH  /api/portfolios/:portfolioId
DELETE /api/portfolios/:portfolioId
POST   /api/portfolios/:portfolioId/restore

GET    /api/portfolios/:portfolioId/transactions?cursor=&limit=&type=
POST   /api/portfolios/:portfolioId/transactions
PATCH  /api/portfolios/:portfolioId/transactions/:transactionId # drafts only
DELETE /api/portfolios/:portfolioId/transactions/:transactionId # drafts only
POST   /api/portfolios/:portfolioId/transactions/:transactionId/reverse
POST   /api/portfolios/:portfolioId/transactions/:transactionId/correct

GET    /api/portfolios/:portfolioId/manual-valuations
POST   /api/portfolios/:portfolioId/manual-valuations
POST   /api/portfolios/:portfolioId/manual-valuations/:valuationId/supersede

GET    /api/portfolio-quotes?instruments=<ids>
GET    /api/portfolios/:portfolioId/summary
GET    /api/portfolios/:portfolioId/history?period=1y
POST   /api/portfolios/:portfolioId/import/preview
POST   /api/portfolios/:portfolioId/import/commit
GET    /api/portfolios/:portfolioId/export?format=csv|json
```

All mutation bodies use an envelope:

```ts
type MutationEnvelope<T> = {
  mutationId: string;
  baseVersion: number;
  data: T;
};
```

Success returns `newVersion`. A `409` returns the current version, conflict classification, and only the server state required for safe reconciliation.

Requirements:

- Authenticate before database access.
- Validate payload size, types, numeric bounds, dates, and portfolio ownership.
- Reject PATCH or DELETE for posted/reversed transactions and direct clients to Correct/Reverse.
- Use cursor pagination for transactions.
- Return structured errors with stable codes.
- Set quote endpoint cache headers independently of private portfolio endpoints.
- Private responses use `Cache-Control: private, no-store` unless a reviewed alternative is safe.
- Quote responses must not contain portfolio holdings or user identifiers, enabling shared caching.

## 13. Neon and Vercel cost controls

### 13.1 Neon

- Run schema changes through migrations, never `CREATE TABLE IF NOT EXISTS` on every request.
- Only write after a user mutation; do not write the payload back immediately after a successful read.
- Debounce remote editing writes for 2–5 seconds while saving locally immediately.
- Batch multi-row imports and migrations in transactions.
- Use the pooled `DATABASE_URL` for serverless application traffic.
- Keep scale-to-zero enabled for low-traffic environments.
- Keep preview branches short-lived and apply TTLs where supported.
- Daily snapshots are written once per portfolio per market day, not per quote refresh.
- Offline replay batches compatible operations when safe; it does not create one Vercel/Neon round trip for every keystroke.
- Coalesce snapshot rebuild requests per portfolio and earliest affected date so several corrections do not create one rebuild job each.
- Monitor CU-hours, storage, history storage, egress, query volume, and slow queries.

Current published Neon reference points as of this specification: Free includes 100 CU-hours and 1 GB storage per project; Launch lists $0.106/CU-hour and $0.35/GB-month. Pricing changes over time, so verify before launch.

### 13.2 Vercel

- One quote refresh equals one bounded API invocation per active portfolio view.
- Default auto-refresh is no more frequent than five minutes and runs only for a visible tab.
- Share provider quote cache entries across users by instrument ID.
- Co-locate Vercel Functions with the Neon region.
- Use Fluid Compute and keep route memory at the lowest level proven sufficient.
- Avoid long upstream timeouts; return partial quote coverage when bounded requests fail.
- Add per-user and per-IP rate limits to manual refresh and import-preview routes.
- Configure spend notifications and a hard budget appropriate to the deployment plan.
- Track invocations, active CPU, provisioned memory duration, cache hit rate, and upstream latency.

Illustrative request budget, not a traffic forecast: 100 users with one visible tab for eight hours across 22 workdays produce about 211,200 quote requests/month at a five-minute interval, about 70,400 at a fifteen-minute interval, and about 1,056,000 at a one-minute interval. Use observed session duration and concurrency before choosing a production cadence.

### 13.3 Cost acceptance thresholds

Before enabling auto-refresh for all users:

- At least 80% shared quote-cache hit rate for common instruments under a representative load test, or an approved provider batch plan that makes the target unnecessary.
- No Neon writes from quote refresh.
- No Neon write on an unchanged page load.
- A documented invocation forecast using measured daily active users, visible-tab time, and average portfolios opened.
- Spend alerts configured in both Neon and Vercel.
- Load test demonstrates bounded upstream concurrency and no unbounded retry loop.

## 14. Security, privacy, and reliability

- Treat portfolio contents and transaction history as sensitive financial information.
- Never log complete transaction payloads, portfolio exports, database URLs, or authentication tokens.
- Apply least-privilege database credentials and rotate compromised secrets.
- Rate-limit mutations, quote refreshes, imports, and exports.
- Protect CSV parsing against formula injection on export and malicious oversized files on import.
- Escape user text such as portfolio names and notes.
- Record audit metadata for imports, destructive changes, and automated corporate actions.
- Define retention behavior for workspace deletion and user-requested data export/deletion.
- Backups and restore procedures must be tested, not only configured.
- Degraded mode keeps local data usable when remote sync is unavailable.

## 15. Accessibility requirements

- Every chart has an equivalent table or textual summary.
- Gain/loss uses signs and labels in addition to color.
- Current focus remains logical after adding, editing, or removing a row.
- Destructive confirmation returns focus to the invoking control when cancelled.
- Screen readers receive save, refresh, import, error, and result-count announcements.
- Editable numeric controls have ticker-specific labels and validation messages.
- The treemap is not the sole way to discover concentration or performance.
- Keyboard users can reach every action without hover.

## 16. Observability and analytics

Operational metrics:

- Portfolio API success/error/latency by operation
- Neon query count and duration
- Sync conflicts and offline retries
- Quote provider latency, cache hit rate, missing-symbol rate, and rate-limit responses
- Quote response completeness
- Snapshot job completion and skipped portfolios
- Import preview/commit counts and validation failures
- Vercel invocation, CPU, and memory usage

Product metrics should avoid sending holdings, symbols, quantities, prices, or portfolio names to general analytics. Useful privacy-preserving events include portfolio created, position count bucket, import started/completed, view selected, and sync state.

## 17. Testing and verification

### 17.1 Unit tests

- Normalization and numeric validation
- Position derivation from every transaction type
- Weighted average display
- FIFO partial sale and remaining basis
- Fees and dividends
- Split and reverse-split invariants
- Realized, unrealized, dividend, fee, total strategy P/L, open-position return, and daily-change calculations
- Suppression of unapproved portfolio-level percentage-return methods
- Complete versus partial pricing
- Currency mismatch behavior
- Conflict/version behavior
- Import deduplication and formula-injection protection
- Gross dividend and withholding treatment
- Reversal plus replacement lifecycle
- Manual valuation selection and completeness
- Phase 1 rejection of non-USD data

### 17.2 Integration tests

- A request with a missing, malformed, or different workspace token cannot access the private workspace.
- Create, reopen, edit, archive, restore, and delete.
- Offline local save followed by remote synchronization.
- Concurrent edits return a conflict instead of silently overwriting.
- Quote provider timeout returns partial coverage.
- Migration produces positions equal to the JSON baseline.
- CSV preview does not mutate data; commit is idempotent.
- Daily snapshot job writes at most one row per portfolio/date/version.
- Same-UUID device-pairing conflicts never overwrite either side silently.
- Tombstone merge, restore, and 30-day purge preserve the declared lifecycle.
- Incomplete snapshots are excluded from performance chaining, and backdated corrections rebuild the affected range.

### 17.3 Browser and responsive tests

- Desktop and 320px/390px mobile layouts.
- Keyboard-only create/add/edit/remove/import flows.
- Screen-reader names and live status announcements.
- Page reload and cross-tab synchronization.
- Hidden-tab polling pause.
- Market-open and market-closed refresh behavior.
- Large but allowed portfolios remain usable and paginated.

### 17.4 Cost verification

- Count Vercel invocations for page open, portfolio switch, edit, and refresh.
- Count Neon queries and writes for the same scenarios.
- Prove an unchanged page load performs zero writes.
- Prove price refresh performs zero Neon operations.
- Measure cache hit rate with overlapping user portfolios.
- Run a representative one-hour soak test and extrapolate with clearly stated assumptions.

## 18. Delivery phases

### Phase 0: harden the existing tracker

- Remove the redundant remote write after initial load.
- Move schema creation to a migration.
- Increase remote-save debounce and add dirty-state tracking.
- Add explicit quote completeness, provider, and observation time.
- Add visibility-aware five-minute refresh behind a configuration flag.
- Add shared quote caching and rate limits.
- Add undo for position and portfolio removal.
- Confirm or replace the provisional market-data source for production.

### Phase 1: transaction ledger

- Add normalized portfolio, instrument, and transaction tables.
- Migrate current JSON holdings with preview and reconciliation.
- Add exact buy, sell, dividend, fee, split, and adjustment effects without a cash ledger.
- Enforce USD-only Phase 1 data and add explicit manual valuations.
- Add the draft, posted, reversed, correction, and replacement lifecycle.
- Derive positions and realized/unrealized performance from transactions.
- Add activity view and transaction editing rules.
- Add the IndexedDB replica, mutation outbox, optimistic concurrency, and idempotent retry IDs.
- Add UUID-based paired-device merge, tombstones, recovered copies, visible conflict resolution, and workspace-token rotation.

### Phase 2: history and portfolio analysis

- Add close-calendar daily snapshot job with completeness metadata.
- Add coalesced rebuilds after backdated ledger changes and exclude incomplete snapshots from performance chaining.
- Add open-exposure and strategy P/L history.
- Add benchmarks and allocation views.
- Expand treemap modes and accessible equivalents.
- Add portfolio duplication and targets.

### Phase 3: portability and automation

- CSV import preview/commit and exports.
- Reconciliation workflow.
- Alerts with cost and delivery controls.
- Reviewed corporate-action automation.
- Optional brokerage-specific adapters after evidence-based validation.

## 19. Acceptance criteria

The expanded workspace is ready for general use when:

1. The owner can create multiple portfolios, record a transaction, reload, and see the same derived position on a paired device without signing in.
2. A partial sale produces the verified remaining quantity, realized estimate, remaining basis, and audit trail.
3. Removing an item offers Undo; permanent deletion follows the retention policy.
4. Quotes display provider and as-of time, refresh no more frequently than policy allows, and never trigger Neon writes.
5. A missing quote cannot appear as zero and cannot produce a falsely complete total return.
6. Price refresh stops in hidden tabs and respects market-closed behavior.
7. CSV import previews changes, reports invalid rows, and commits idempotently only after confirmation.
8. History distinguishes open exposure, realized P/L, unrealized P/L, dividends, and fees without presenting a brokerage cash balance.
9. Ownership tests prove requests without the exact private workspace token cannot access portfolio or transaction data.
10. Cost tests prove unchanged loads perform no database writes and demonstrate bounded Vercel invocations.
11. Mobile, keyboard, and screen-reader flows pass the accessibility requirements.
12. Market-data licensing, retention, and user deletion policies are documented and approved.

## 20. Open decisions and material gaps

These decisions can materially change implementation and must be resolved before the associated feature ships:

- Approved production quote and historical-price provider, license, attribution, coverage, and cost.
- Whether portfolios can contain non-USD instruments and how FX observations are sourced and timestamped.
- Supported realized-basis methods beyond FIFO and the degree of tax-lot specificity.
- Definition and validation standard for any future unitized portfolio-level strategy return.
- Whether short positions, options, crypto, bonds, and private assets are in scope.
- Market-hours calendar source for international instruments.
- Private-link rotation, lost-link recovery policy, and financial-data deletion workflow.
- Commercial versus personal deployment, which affects the appropriate Vercel plan.
- Expected daily active users, visible-tab duration, and concurrency; current cost examples are scenarios, not forecasts.
- Whether alerts require email, SMS, push, or only in-app delivery.

No probability, return estimate, or infrastructure forecast should be presented as validated until it is measured against real usage and reconciled source data.
