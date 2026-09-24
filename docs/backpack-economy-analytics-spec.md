# Backpack Economy Analytics — Detailed Implementation Specification

Status: Phase A implemented locally; production migration/capture pending

Date: 2026-09-23

Target: Existing Backpack monitor at `/market/crypto/backpack`
Primary question: How many observable participants use Backpack-issued on-chain securities, is participation growing, and is the resulting economic activity durable?

## 1. Executive decision

This work fits the monitor already implemented. It should extend the current collector, immutable snapshot model, adoption summaries, API, and UI rather than create a second application.

The existing implementation already provides:

- an approved Backpack asset registry;
- finalized token-supply observations;
- complete, supply-reconciled holder enumeration when Helius succeeds;
- owner aggregation across token accounts;
- evidence-backed system-wallet exclusions;
- immutable daily snapshots and data-quality records;
- deduplicated ecosystem owner counts;
- whole-token and multi-security sensitivity measures;
- 1-, 7-, and 30-day owner endpoint cohorts;
- 7-, 30-, and 90-day price-independent growth assessments;
- raw transaction, DEX aggregate, liquidity, DeFi, issuer-comparison, and retention table foundations;
- a public API and the existing Backpack dashboard route.

The principal gaps are:

1. the registry is a curated starter universe, not a continuously reconciled record of every Backpack-issued security;
2. DEX collection is a bounded sample from up to ten current-holder wallets, so total volume and total traders are intentionally unavailable;
3. trader retention, repeat participation, venue concentration, and activity-quality metrics are not derived;
4. DeFi use and liquidity depth have schema support but insufficient complete observations;
5. sourced external claims such as the `$193.3M` week-over-week increase are not stored as structured evidence;
6. issuer comparisons do not yet have comparable verified observations or a defined denominator.

### Phase A implementation checkpoint — 2026-09-23

Implemented in the current worktree:

- registry lifecycle columns and immutable daily lifecycle observations;
- daily reconciliation of enabled official Solana security identities;
- unresolved/conflicting official identities retained for review rather than silently omitted;
- SPCX added to the curated approved starter universe with its official launch source;
- structured, idempotent product-event and external-observation storage;
- the Token Terminal `$193.3M` figure stored as a seven-day **change**, not a total;
- public API exposure and UI sections for official coverage, unresolved identities, external evidence and product events;
- “positive-supply securities” terminology separated from registered/launched product counts.

The live Backpack APIs exposed 61 enabled Solana security identities during implementation. The
approved research registry remains deliberately curated; newly observed identities are not promoted
without exact identity and underlying-security review. Phase B remains unimplemented.

The result will remain a monitor of the **observable on-chain economy for Backpack-issued securities**. It will not claim to measure Backpack Exchange customers, brokerage accounts, deposits, revenue, or unique people.

## 2. Product scope and boundaries

### 2.1 In scope

- Backpack Securities assets issued on Solana and later supported chains.
- Registered, launched, positive-supply, redeemed, paused, and inactive asset states.
- Non-system owners of those assets.
- On-chain transfers, issuance, redemption, DEX trades, traders, venues, pools, liquidity, and identifiable DeFi positions.
- Daily, 7-day, 30-day, and 90-day changes.
- Comparable issuer-level observations when their asset identities and collection methods are verified.
- Structured third-party observations with explicit source, period, scope, and methodology.
- Launches, campaigns, listings, and other dated annotations used as context rather than causal proof.

### 2.2 Out of scope without a new authenticated source

- Backpack Exchange account counts or monthly active users.
- Brokerage customers or off-chain securities positions.
- Fiat/stablecoin deposits, company revenue, fees, or profitability.
- A claim that one wallet equals one person.
- Investment scores, recommendations, or causal claims based only on correlated launches and volume.
- Market-wide dollar AUM when reference prices are unavailable.
- Inferred circulating BP supply or BP-token valuation as a proxy for securities adoption.

### 2.3 Required language

Use these terms consistently:

- **Registered security**: an approved Backpack security identity in the registry.
- **Launched security**: official availability is evidenced by a dated primary source or enabled issuance/redemption state.
- **Positive-supply security**: captured on-chain supply is greater than zero.
- **Observable owner**: a deduplicated wallet with a positive balance after verified system exclusions.
- **Active trader**: an identifiable non-system user account involved in at least one normalized DEX trade in the stated UTC window.
- **DEX volume for Backpack-issued securities**: trading involving those assets across covered decentralized venues. Do not call it “Backpack DEX volume.”
- **Week-over-week volume change**: current seven-day volume minus the immediately preceding seven-day volume. It is not the current total.
- **External observation**: a sourced metric not produced by this collector.

## 3. Success criteria

The feature is complete when the dashboard can answer, with dated evidence:

1. How many observable wallets own a Backpack-issued security today?
2. How many were retained, entered, or departed over 7 and 30 days?
3. How many distinct wallets traded during the last day, 7 days, and 30 days?
4. Are owners, traders, supply, and multi-security participation growing?
5. How much complete DEX volume occurred, and how did it change from the preceding equal window?
6. Is activity broad and persistent, or concentrated in a few assets, venues, pools, and wallets?
7. How much liquidity and identifiable DeFi use supports the assets?
8. Which products and launches explain the timing of changes without asserting causation?
9. How does Backpack compare with other issuers under the same definitions and coverage standard?
10. Which answers are complete, partial, stale, or unavailable, and why?

No headline may be classified from sampled volume, incomplete registry coverage, missing dates, or silently changed asset/system-wallet cohorts.

## 4. Fit with the existing implementation

| Requirement | Existing implementation | Decision |
| --- | --- | --- |
| Authoritative asset universe | `backpack_assets`, `starter_universe.json`, official API revalidation | Extend with lifecycle observations and unmatched-asset alerts |
| Daily supply | `backpack_asset_daily_snapshots` | Reuse |
| Complete holders | holder snapshots, current state, checkpoints, reconciliation | Reuse |
| Owner growth | `backpack_adoption_daily` and assessments | Reuse and extend |
| Owner entry/exit/retention | identity-free endpoint cohorts | Reuse |
| System-wallet filtering | wallet labels and immutable revisions | Reuse; add activity-specific labels where necessary |
| Raw swaps | `backpack_transactions` | Reuse as normalized event store after market-wide collection is added |
| DEX daily aggregates | `backpack_asset_dex_daily_snapshots` | Extend with complete coverage and venue-level dimensions |
| Ecosystem totals | `backpack_ecosystem_daily_snapshots` | Extend with complete activity fields or companion summary table |
| Activity cursor | current cursor is per sampled wallet | Add a source/asset/partition cursor for market-wide ingestion |
| Trader cohorts | Not implemented | Add permanent first/last-seen state and daily identity-free summaries |
| Liquidity | Jupiter quote snapshot table | Reuse; add pool liquidity/depth observations when verified |
| DeFi use | schema and protocol labels exist | Implement verified protocol-position collection |
| Competitors | issuer-neutral registry/snapshot foundation | Extend only after comparable identities and sources are approved |
| External claims | quality events are not the right shape | Add structured external observations and event annotations |
| Public UI/API | existing route, store, monitor, and adoption overview | Extend the existing page and response contract |

### Architectural decision

Keep one daily worker and one public dashboard. Add provider adapters and derived summaries behind the existing collector-owned schema. Public requests remain read-only and must never trigger ingestion.

## 5. Workstream 1 — Complete asset-universe coverage

### 5.1 Objective

Maintain a dated, auditable universe of every official Backpack-issued security and clearly distinguish registration, launch, positive supply, transfer availability, and retirement.

### 5.2 Discovery inputs

Use sources in this order:

1. Backpack `/api/v1/assets` and `/api/v1/securities` exact identities;
2. official Backpack product articles containing an exact mint address;
3. issuer-controlled token lists or signed metadata;
4. finalized on-chain mint metadata and supply validation;
5. manually approved evidence only when primary sources conflict or omit an identity.

Ticker/name similarity alone must never create or merge an asset.

### 5.3 Registry changes

Add lifecycle fields to `backpack_assets`:

- `registry_status`: `registered`, `launched`, `paused`, `redeemed`, `inactive`;
- `first_official_seen_at`;
- `last_official_seen_at`;
- `launch_evidence_source`;
- `redemption_evidence_source`;
- `deposit_enabled` and `withdraw_enabled` as current convenience fields;
- `identity_fingerprint` from network, mint, decimals, issuer, and underlying identifier.

Add `backpack_asset_registry_daily`:

| Field | Meaning |
| --- | --- |
| `asset_id`, `date` | Immutable key |
| `official_present` | Present in official source that day |
| `deposit_enabled`, `withdraw_enabled` | Official operational flags |
| `token_supply` | Reconciled daily supply reference |
| `lifecycle_state` | State derived under versioned rules |
| `source`, `observed_at` | Evidence |
| `methodology_version` | Classification version |
| `quality_status`, `limitation` | Completeness |

Add a registry reconciliation result to each ingestion run:

- official identities discovered;
- approved identities captured;
- official identities not yet approved;
- approved identities missing from the official source;
- mint/decimal/underlying conflicts;
- newly positive-supply or newly zero-supply assets.

### 5.4 Lifecycle rules

- `registered`: verified identity exists, but no launch evidence and no positive supply.
- `launched`: official launch evidence, enabled issuance/redemption, or positive on-chain supply.
- `paused`: previously launched, currently disabled, with supply still positive.
- `redeemed`: previously positive supply reaches zero with complete observations.
- `inactive`: explicit official retirement or a reviewed administrative decision.

Positive supply is a metric, not a synonym for “issued product.”

### 5.5 Acceptance tests

- SPCX and every other official exact-mint asset are either approved or surfaced as unresolved coverage.
- Five positive-supply assets display as “5 positive-supply securities,” never “5 issued securities.”
- New official identities do not silently change historical comparison cohorts.
- An asset added during a 30-day window makes same-universe growth unavailable until a comparable baseline exists.

## 6. Workstream 2 — Participant adoption and retention

### 6.1 Existing measures retained

- observable owners;
- whole-token owners;
- multi-security owners;
- owners by security;
- retained, entered, and departed owner wallets at 1/7/30-day endpoints;
- 7/30/90-day adoption state and momentum;
- system-wallet exclusion fingerprint;
- positive-supply breadth and median per-security supply growth.

### 6.2 New owner-quality measures

Add to `backpack_adoption_daily.data`:

- balance bands in token units: `<0.01`, `0.01–<1`, `1–<10`, `10+`;
- share of owners holding at least 0.01 and at least one token;
- top-10/top-20/top-100 included-owner concentration in token units;
- owner concentration excluding known liquidity/market-making entities;
- number of owners whose entire ecosystem position is below one token;
- asset breadth: number and share of positive-supply securities gaining owners;
- ownership HHI by security, clearly labeled as wallet distribution rather than persons.

Dollar cohorts remain optional and require valid reference prices. They must not block price-independent owner analytics.

### 6.3 Trader identity state

Add `backpack_trader_state`:

| Field | Meaning |
| --- | --- |
| `scope` | `ecosystem` or asset ID |
| `wallet_address` | Public on-chain account |
| `first_complete_seen_date` | First date observed after full activity coverage began |
| `last_seen_date` | Latest trade date |
| `active_days` | Count of distinct complete activity days |
| `trade_count` | Normalized trades |
| `known_system` | Current exclusion state |
| `updated_at` | Operational timestamp |

The first complete market-wide day establishes a baseline. Wallets present that day are “baseline traders,” not new users.

Add identity-free fields to a new `backpack_activity_daily` summary:

- daily active traders;
- 7-day and 30-day active traders;
- first-observed traders after baseline;
- repeat traders with at least two active days;
- retained, entered, departed, and reactivated traders over 7 and 30 days;
- trade frequency distribution;
- median trades per trader;
- median active days per trader in the rolling window.

### 6.4 Participant definitions and exclusions

- Prefer the swap instruction’s user account; never treat a relayer fee payer as the trader.
- Exclude confirmed/high-confidence DEX programs, pools, routers, bridges, treasury, custody, and market-maker operational wallets from user counts.
- Preserve raw counts alongside excluded counts for reconciliation.
- Store an activity exclusion fingerprint. A label change must invalidate comparisons rather than appear as organic growth.
- Do not merge different wallet addresses into a person or institution without explicit evidence.

### 6.5 Acceptance tests

- A wallet holding three securities counts once at ecosystem level and three times only in per-asset rows.
- A router or fee payer never becomes a trader.
- First-day wallets are baseline wallets, not acquisitions.
- Label changes interrupt comparable trader cohorts.
- Raw participant identities are not copied into public aggregate payloads.

## 7. Workstream 3 — Complete economic activity

### 7.1 Objective

Replace the experimental current-holder swap sample with complete mint-wide activity for every positive-supply Backpack security and every UTC activity day.

The current sample remains visible only as a diagnostic until complete coverage is signed off. It must not be combined with complete totals.

### 7.2 Source-adapter contract

Implement a market-wide activity adapter capable of enumerating every covered DEX trade involving a mint between finalized start and end slots. Candidate sources include an indexed Solana DEX dataset/API or a commercial indexer. Provider selection is a deployment decision; all adapters must return the same normalized records.

Required adapter output:

```text
asset mint
transaction signature
outer trade index or deterministic event index
slot and block timestamp
user account, when identifiable
venue/program/pool
base and quote mints
base and quote amounts
buy/sell direction relative to the Backpack security
USD notional and its pricing method, when available
router/aggregator identifier
source record identifier
```

Required completeness evidence:

- finalized start/end slots for the UTC day;
- pagination or partition completion;
- source watermark beyond the end slot;
- number of raw records, normalized records, duplicates, rejected records, and unpriced records;
- assets and venues covered;
- adapter version and query hash;
- independent comparison, where available.

### 7.3 Normalization and deduplication

- Count one economic trade per asset/signature/outer event.
- Do not sum inner route legs.
- When one routed transaction touches multiple pools, attribute one trade to the router and retain pool legs as detail, not additional volume.
- If one signature contains multiple independent outer trades, use a deterministic event index.
- A transaction involving two tracked securities creates one normalized observation for each security but only one ecosystem trade event when computing ecosystem transaction counts.
- Attribute volume to the security side once. Do not double the notional.
- Stablecoin quote value is measured from executed quote amount using an approved stablecoin registry.
- Non-stable quote trades require a timestamped on-chain execution price. If it is unavailable, preserve the trade with null USD volume.
- Complete trade counts may be published with partially priced volume; total USD volume remains unavailable unless the pricing-completeness threshold is met.

### 7.4 Storage changes

Extend `backpack_transactions` with additive columns:

- `event_index`;
- `activity_date`;
- `quote_mint`, `quote_amount`;
- `pool_address`, `router`, `program_id`;
- `usd_method`;
- `pricing_timestamp`;
- `coverage_run_id`;
- `is_system_activity`;
- `normalizer_version`.

Migrate the uniqueness key to `(asset_id, signature, event_kind, event_index)`.

Add `backpack_activity_cursors` keyed by:

- provider;
- asset;
- partition/query identifier;
- through slot and timestamp;
- query/adapter version.

Add `backpack_activity_coverage`:

| Field | Meaning |
| --- | --- |
| `asset_id`, `activity_date`, `provider` | Key |
| `start_slot`, `end_slot`, `source_watermark_slot` | Boundary proof |
| `raw_events`, `normalized_events`, `duplicate_events`, `rejected_events`, `unpriced_events` | Reconciliation |
| `coverage_status` | `Verified`, `Estimated`, `Partial`, `Unavailable` |
| `methodology`, `query_hash`, `adapter_version` | Reproducibility |
| `validated_at` | Timestamp |

Extend `backpack_asset_dex_daily_snapshots` with:

- `volume_usd` separate from the legacy `observed_volume_usd`;
- buys, sells, trades, active traders;
- priced-trade percentage;
- excluded-system volume and trades;
- router and pool dimensions;
- top-10 trader volume share;
- repeat-trader share;
- coverage record reference.

Add ecosystem aggregates that deduplicate signatures and wallets across securities. Never sum per-asset unique traders to get ecosystem traders.

### 7.5 Derived activity metrics

For 1-, 7-, and 30-day windows calculate:

- total DEX volume;
- immediately preceding equal-window volume;
- absolute and percentage change;
- trades and active traders;
- volume per active trader;
- median trade size and p95 trade size;
- buy/sell share;
- repeat-trader share;
- after-hours/weekend share;
- volume by security, venue, pool, router, and quote asset;
- top-1/top-3/top-5 security share;
- top-10 trader share;
- volume-to-supply turnover without requiring an underlying stock price;
- volume-to-reference-AUM only when reference prices exist.

The external `$193.3M` claim must display as:

> DEX volume for Backpack-issued tokenized securities increased by $193.3M versus the preceding seven-day period.

It must never populate a “7D total volume” field unless the actual total is separately observed.

### 7.6 Completeness gates

An ecosystem total is complete only when:

- every positive-supply asset has complete activity coverage for every day in the window;
- all source partitions and pages completed;
- registry identities did not change silently;
- all dates are consecutive;
- all trades needed for USD total meet the configured pricing-completeness standard;
- no unresolved material reconciliation discrepancy exists.

Otherwise show the observed lower bound and the explicit missing scope. Do not annualize or extrapolate.

## 8. Workstream 4 — Activity quality, liquidity, and DeFi use

### 8.1 Objective

Determine whether headline activity represents broad, repeat use or concentrated launch-period churn.

### 8.2 Quality measures

Capture and display:

- unique traders versus trades and volume;
- new versus repeat versus reactivated traders;
- one-day, 7-day, and 30-day trader retention;
- top-10 and top-100 trader volume concentration;
- security, venue, pool, and router concentration;
- median and p95 trade size;
- same-wallet rapid round trips when deterministically observable;
- known market-maker/system share;
- incentive/campaign periods;
- activity decay 1, 7, 14, and 30 days after launch or campaign end.

Do not label suspicious patterns as wash trading without transaction-level evidence and a reviewed methodology. Use “concentrated,” “repetitive,” or “system-associated” descriptions.

### 8.3 Liquidity measures

Retain the current Jupiter two-direction quotes and add, when a verified source exists:

- executable depth at $1K/$10K/$50K/$100K;
- bidirectional price impact;
- pool liquidity by asset and venue;
- price dispersion across venues;
- failed-route rate;
- liquidity concentration by pool;
- quote freshness and source timestamp.

Liquidity is supporting evidence. A quote is not a trade, and pool TVL is not user adoption.

### 8.4 DeFi utilization

Use `backpack_asset_defi_daily_snapshots` only for verified protocol contracts and positions. Capture:

- tokens and owner wallets deposited by protocol/category;
- share of token supply held in verified protocols;
- lending supplied and borrowed amounts where directly observable;
- collateral positions where the integration exposes them;
- protocol count and concentration.

Do not infer DeFi use merely because a token is transferable or “DeFi compatible.” Unknown protocol wallets remain ordinary unlabeled wallets until reviewed.

### 8.5 Durability classification

Add a descriptive, versioned `backpack_economy_assessments` record for 7- and 30-day windows. It is not an investment score.

Dimensions:

- ownership direction;
- active-trader direction;
- repeat participation;
- complete volume direction;
- supply breadth;
- security/venue/trader concentration;
- liquidity availability;
- data coverage.

Allowed states:

- `Broadening`: owners and active/repeat traders grow across a broad share of securities.
- `Trading-led`: volume/trades grow materially while owner or repeat-trader growth does not.
- `Ownership-led`: owners/supply broaden while trading is flat or unavailable.
- `Contracting`: owners and active traders decline broadly.
- `Mixed`: dimensions disagree.
- `Insufficient evidence`: any required input or comparable window is incomplete.

Thresholds must be configuration values persisted with each assessment. Initial thresholds are research conventions and require calibration after at least 30 complete observations.

## 9. Workstream 5 — Evidence, competition, and dashboard

### 9.1 Structured external observations

Add `backpack_external_observations`:

| Field | Meaning |
| --- | --- |
| `id` | Immutable identifier |
| `metric` | e.g. `dex_volume_change_usd` |
| `scope_type`, `scope_id` | issuer, asset, chain, venue, or category |
| `period_start`, `period_end` | Current window |
| `comparison_start`, `comparison_end` | Prior window, when applicable |
| `value`, `unit` | Reported value |
| `observation_kind` | total, change, percentage, share, rank |
| `source_url`, `source_publisher`, `published_at` | Provenance |
| `primary_source_url` | Origin when the publisher is a repost |
| `methodology` | Known definition |
| `coverage_status` | Verified, Estimated, Partial, Unavailable |
| `review_status` | pending, approved, rejected, superseded |
| `limitation` | Ambiguity and caveats |
| `recorded_at`, `recorded_by` | Audit data |

The Token Terminal observation should be stored as a change, with Token Terminal as the primary publisher and Odaily/KuCoin/Ground News as secondary citations only. If the underlying total and exact dates are unknown, keep them null.

### 9.2 Event annotations

Add `backpack_economy_events` for launches, campaigns, listings, operational changes, and material sourced reports:

- event date/time;
- event type;
- affected assets;
- title and description;
- source and evidence status;
- optional campaign start/end;
- no causal-effect field.

Charts may overlay events. The UI must say “coincided with,” not “caused,” unless a separate causal analysis supports the statement.

### 9.3 Competitor comparisons

Extend the issuer-neutral tables only when all of the following are true:

- exact official asset identities are approved;
- asset type and underlying exposure are comparable;
- observation dates and UTC windows match;
- metric methodology matches;
- chain and venue coverage are disclosed;
- bridged/native treatment is consistent;
- category members and denominator are persisted.

Compare:

- owner wallets by issuer, without summing asset owners into unique issuer owners;
- active traders;
- complete DEX volume and week-over-week change;
- positive-supply products;
- supply/owner/trader breadth;
- liquidity and concentration;
- directly comparable underlying exposures.

“Coinbase” must be labeled as an issuer of tokenized assets when that is the dataset dimension, not as Coinbase exchange volume.

### 9.4 Public API

Keep `GET /api/market/crypto/backpack` and add versioned sections without breaking existing clients:

```json
{
  "universe": {"summary": {}, "assets": [], "coverage": {}},
  "adoption": {"current": {}, "history": [], "assessments": []},
  "activity": {"current": {}, "history": [], "windows": [], "coverage": {}},
  "quality": {"concentration": {}, "liquidity": {}, "defi": {}},
  "context": {"events": [], "externalObservations": [], "competitors": []},
  "provenance": {"asOf": "", "methods": [], "limitations": []}
}
```

The server store should query precomputed daily summaries. It must not calculate large cohorts or scan raw transactions during a public request.

### 9.5 Dashboard information architecture

Keep the existing Backpack route and lead with four questions:

1. **How many use it?** Observable owners, 7/30-day active traders, multi-security owners.
2. **Is it growing?** Owner and trader cohorts, breadth, supply, and momentum.
3. **How much activity occurs?** Complete volume, trades, change versus prior window, and top assets/venues.
4. **Is it durable?** Repeat traders, retention, concentration, liquidity, and post-event persistence.

Recommended sections:

- Scope and coverage banner.
- Core economy scorecard.
- Ownership and trader growth timelines.
- Cohort movement.
- Activity by security and venue.
- Durability and concentration.
- Asset-universe table with lifecycle status.
- Sourced events and external observations.
- Comparable issuer view.
- Collapsed methodology and optional price/AUM/BP research.

Every metric card includes:

- exact label and unit;
- observation/window dates;
- current value and comparable change;
- coverage status;
- source/method details;
- limitations;
- `N/A`, never zero, for unknown values.

## 10. Daily processing sequence

Run in this order:

1. Reconcile official registry and flag unresolved identities.
2. Capture finalized supply and complete holder enumerations.
3. Capture mint-wide activity for the completed prior UTC day.
4. Normalize/deduplicate trades and persist completeness evidence.
5. Capture liquidity quotes and verified DeFi balances.
6. Reconcile supply, holders, transaction partitions, pricing, and independent sources.
7. Derive immutable universe, adoption, activity, cohort, concentration, and durability summaries.
8. Import or review approved external observations and event annotations separately.
9. Run maintenance only after permanent aggregate validation.
10. Write audit/cost artifacts and revalidate the public cache.

A failure in market-wide activity must not discard a valid holder snapshot. Each analytical lane carries its own coverage state.

## 11. Backfill policy

- Never reconstruct historical holders from current balances.
- Historical activity may be backfilled only from a source that supplies complete finalized transaction history and reproducible query boundaries.
- Mark backfilled records with their source, query hash, adapter version, and ingestion time.
- Do not label a trader “new” before the first complete activity baseline.
- Keep external historical observations separate from internally derived history.
- Never overwrite immutable daily summaries. Corrections require a versioned correction record or reviewed migration.

## 12. Data quality and reconciliation

### 12.1 Coverage states

- `Verified`: complete source boundaries plus successful reconciliation against an independent source or invariant.
- `Estimated`: complete primary capture with disclosed pricing/attribution assumptions.
- `Partial`: known missing assets, venues, records, prices, pages, or identities.
- `Stale`: otherwise valid evidence older than its freshness limit.
- `Unavailable`: no defensible value.

### 12.2 Required invariants

- Holder balances reconcile to finalized supply within tolerance.
- Registry identities are exact mint matches.
- Ecosystem unique owners/traders are deduplicated wallet sets, never sums.
- Routed swaps contribute volume once per asset/event.
- Current and previous comparison windows are adjacent and equal length.
- Week-over-week change equals current total minus prior total.
- Volume totals equal their published asset/venue breakdown within rounding tolerance.
- Complete windows contain every expected date and asset.
- Changed asset or exclusion fingerprints block comparison.
- Unknown data remains null.

### 12.3 Independent checks

Where possible, compare daily/weekly activity against Token Terminal, Birdeye, Dune, or another approved indexed source. Differences above a configured tolerance create a quality event and suppress `Verified`; they do not automatically select the larger number.

## 13. Testing requirements

### 13.1 Unit tests

- lifecycle classification and registry conflict handling;
- multi-hop and multi-pool swap deduplication;
- stablecoin and non-stable USD valuation;
- ecosystem trader deduplication;
- system/router/market-maker exclusions;
- first-baseline versus new-trader semantics;
- equal-window volume totals and deltas;
- activity cohort retention/reactivation;
- concentration calculations;
- durability-state rules;
- null propagation and coverage downgrades.

### 13.2 Integration tests

- schema migration against empty and populated PostgreSQL;
- idempotent replay of the same activity partition;
- incomplete pagination cannot publish a total;
- partial activity does not block valid adoption summaries;
- retention cleanup requires validated aggregates;
- API queries work with old schema during additive rollout and new schema after migration.

### 13.3 UI tests

- `$193.3M increase` never renders as `$193.3M 7D volume`;
- “Backpack-issued securities DEX volume” replaces “Backpack DEX volume”;
- positive-supply and registered counts are distinct;
- incomplete coverage is visible beside the headline;
- unknown values render as `N/A`;
- mobile tables/charts do not overflow;
- source details expose dates, definitions, and limitations.

### 13.4 Operational acceptance

- Seven consecutive production captures complete without duplicated transactions.
- Asset and ecosystem aggregates reconcile for each day.
- One independent weekly activity comparison is within the approved tolerance or has a documented explanation.
- Request, storage, and runtime budgets remain below configured limits.
- A human reviews the first complete 7-day window before enabling the public activity headline.

## 14. Delivery phases

### Phase A — Registry and terminology

- Add lifecycle observations and registry reconciliation.
- Resolve SPCX and every unmatched official product.
- Change UI labels to registered/positive-supply/launched.
- Add structured events and external observations.

This phase immediately fixes the current universe ambiguity.

### Phase B — Complete activity foundation

- Select and prove a market-wide indexed activity source.
- Add adapter, coverage records, normalized events, and cursors.
- Backfill a bounded validation interval.
- Reconcile with an independent weekly source.

No total volume headline ships before this phase passes completeness gates.

### Phase C — Traders and durability

- Add trader state and identity-free daily summaries.
- Derive active/repeat/retained/reactivated cohorts.
- Add concentration and post-event persistence.
- Calibrate classifications after sufficient observations.

### Phase D — Liquidity, DeFi, and competition

- Add verified pool depth and protocol balances.
- Approve comparable competitor identities and universes.
- Publish issuer comparisons only after same-method coverage exists.

### Phase E — UI consolidation and sign-off

- Rebuild the existing page around the four core questions.
- Move price/AUM/BP panels into supporting research.
- Complete desktop/mobile visual verification, accessibility checks, and production audit.

## 15. Prioritized implementation backlog

1. Resolve the official-asset coverage gap, especially SPCX, and ship lifecycle terminology.
2. Add external observation/event tables so the `$193.3M` evidence is captured correctly now.
3. Prove one complete market-wide activity adapter on SPCX and MU over seven days.
4. Generalize the adapter across all positive-supply securities with coverage attestations.
5. Add ecosystem activity summaries and trader cohorts.
6. Add concentration, repeat-use, and event-persistence analytics.
7. Integrate the new activity lane into the existing public API and UI.
8. Add verified liquidity/DeFi sources.
9. Add competitor coverage with identical definitions.
10. Calibrate durability thresholds after 30 complete daily observations.

## 16. Final recommendation

Proceed inside the existing Backpack monitor. Approximately two-thirds of the required foundation is already present: registry controls, holder/supply capture, immutable observations, quality status, adoption cohorts, retention safeguards, APIs, and UI. The hard new work is not another dashboard; it is acquiring and proving complete mint-wide activity data, then deriving trader and durability measures without confusing trading intensity with adoption.

The first production milestone should therefore be: **complete the official asset universe and prove complete seven-day activity coverage for SPCX and MU while preserving the current adoption headline.**
