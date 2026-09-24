# Backpack on-chain thesis monitor

Implementation branch: `feature/backpack-thesis-monitor`. Entry: `/market/crypto/backpack`.
This is a Phase 1 implementation with explicit launch gaps, not a claim that all 33 sections of the specification are complete.

For verified production status, code ownership, operating commands and prioritized remaining work, read [the engineering handoff](backpack-engineer-handoff.md).

## What is implemented

**Current lead assessment (2026-09-23):** the user's clarification prioritizes rapid/slow/status-quo/declining adoption independent of stock prices. The price-independent adoption overview below supersedes the price-required headline described in the older Growth overview section. Original financial metrics and stored `growth-v1` assessments remain unchanged as supporting evidence.

- Dedicated research subpage and per-asset drill-downs. AUM leads; issuance and meaningful wallets sit beside it. Responsive layout, sortable/searchable asset table, 7/30/90/180-day, YTD, 1Y and ALL chart ranges, per-metric evidence and run diagnostics.
- Postgres registry with manual administrator approval, exact Solana mint validation, evidence URL and approval notes. Securities are never discovered by matching symbols. Admin page `/admin/backpack` and API `/api/admin/backpack` use the existing admin cookie plus route-level authorization and same-origin mutation checks.
- Daily 00:30 UTC GitHub Actions job, manual admin dispatch, bounded request budget, retry/backoff, per-asset isolation, transactional snapshot writes, primary-key deduplication, expiring database worker lease compatible with Neon transaction pooling. Captured snapshots are immutable on rerun; failed assets can retry.
- Finalized token supply, paginated Helius token accounts reconciled against supply, owner-aggregated economic cohorts, $100/$1K/$10K/$100K thresholds, raw/economic concentration, exact-decimal supply deltas, reference AUM and on-chain market value.
- Ecosystem unique meaningful wallets and multi-asset breadth. Each counted asset requires at least $100; aggregate wallet value across securities determines ecosystem meaningful-wallet membership. System exclusions require confirmed/high evidence. Wallet addresses are not counts of known individual investors.
- Separately seeded BP mint exactly as supplied by the user. Its exact mint is now verified against the official Backpack Learn BP contract article (2026-09-22). It is unrelated to the existing crypto-workbench `BACKPACK` entry. No staking or circulating-supply estimate is invented.
- Jupiter Price V3 and optional buy/sell executable-route observations at $1K/$10K/$50K/$100K; failure is N/A. Response route legs are not mislabeled sequential hops: split routes make that equivalence unsafe. Price impact retains the API's percent units.
- Alpaca SIP stock reference adapter. The existing Yahoo helper has no official exchange feed and drops price timestamps, so it is not used as a silently verified substitute. SIP entitlements are needed.
- Real Helius outer-swap normalization with transaction/slot/time evidence, deduplication per signature/asset, independent of transfers. The initial activity collector is explicitly a **bounded current-wallet sample**, not complete token-wide transaction indexing.
- Exchange calendar calculations cover US DST, holidays, weekends and early closes. Closed-market parity remains last-available reference, never an arbitrage alert.
- Independent RPC supply comparison and highest-liquidity exact-base DexScreener pair price check. Independent pair rolling volume is not silently compared with a full UTC-day ecosystem volume.
- Mechanically calculated daily changes with exact-date comparisons; no LLM or arbitrary bullish score.

## Required setup

GitHub Actions secrets:

| Secret | Use |
| --- | --- |
| `DATABASE_URL` | Existing Neon Postgres database |
| `HELIUS_API_KEY` | DAS holders and enhanced transaction history |
| `JUPITER_API_KEY` | Price V3 and route quotes |
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | Optional licensed SIP equity snapshots |
| `SOLANA_VALIDATION_RPC_URL` | Optional independent validation endpoint; public Solana RPC fallback |

The browser never receives these keys. Vercel uses its existing `DATABASE_URL`, `ADMIN_SECRET`, `GITHUB_ACTIONS_TOKEN`, `GITHUB_REPO_OWNER`, `GITHUB_REPO_NAME`, and `GITHUB_DEFAULT_REF` for database reads/admin dispatch.

The scheduled workflow defaults `BACKPACK_REQUIRE_EQUITY_REFERENCE` to `0`, allowing
on-chain capture with Helius and Jupiter when Alpaca is unavailable. Set that GitHub
repository variable to `1` to require Alpaca again. Local preflight defaults to the
strict mode; set the environment variable to `0` for the same limited-data mode.
`ready_for_capture` controls the preflight exit status; `ready_for_security_capture`
still reports full readiness including equity references. The Alpaca check remains
Unavailable when missing or failing, with an explanation that it is optional for
capture. All other readiness checks remain required. Stock-reference AUM, dollar
issuance, securities dollar-based holder metrics, premium/discount and dependent
growth assessments stay unavailable without a usable equity reference. On-chain
supply, holder counts, token prices, quotes and sampled swaps can still be captured;
BP metrics continue to use Jupiter prices. Missing metrics are never replaced by zero.

Initialize with `python backpack_monitor.py --migrate`. Capture with `python backpack_monitor.py --execute`. The scheduled workflow does both. It is active only after the workflow is on the default branch. GitHub scheduling is best effort; timestamps show actual observation time. Exact 00:30 execution is not guaranteed.

Add each security using `/admin/backpack`, with its official mint announcement and an explicit approval. Verify one-token/one-share backing; otherwise the current AUM formula is not valid and the asset must not be approved. Do not import exchange listings, similarly named memecoins, or xStocks as Backpack-issued without issuer evidence. No security mints were seeded without that evidence.

## Observation semantics

Snapshot `date` is the UTC **capture date**, with actual `captured_at`, finalized supply slot, and holder enumeration start/end slots. It is not a fabricated midnight historical state. Activity associated with a capture is the **previous UTC calendar day**. Equity and token price timestamps are retained independently.

Reference AUM is supply times last available underlying reference price. AUM growth can come from underlying price appreciation. Daily net on-chain issuance uses exactly yesterday's supply and today's reference price. Rolling dollar issuance sums those daily dollar changes, requiring all dates; it does not price the whole interval at today's price. Missing previous days produce N/A.

`holder_count` is nonzero token accounts. `unique_holders` combines accounts by owner. Meaningful wallets and economic concentration exclude only registry-labeled systems with confirmed/high confidence. Economic concentration uses the eligible owner balance total as its denominator; raw concentration uses all owner balances. Anonymous omnibus ownership remains unresolved.

The holder enumeration is current paginated data, not an atomic historical snapshot. Totals are withheld when balances do not reconcile within `BACKPACK_SUPPLY_TOLERANCE` (default 0.1%). Its quality is Estimated even after reconciliation. Complete enumeration is necessary to compute new/lost owners and deduplicated ecosystem wallets.

An ecosystem daily row is finalized only after every eligible security has a daily row. Each portfolio metric requires all its asset components; unknown asset AUM never reduces the sum silently. BP is excluded. Comparisons with a different number of covered assets are withheld. Issuer verification and survivorship/cohort changes still require analyst review.

Price parity is a daily observation, not intraday median, p95 or duration. Jupiter's timestamp is obtained from its `blockId`; prices over one hour old carry Stale provenance. BP economic thresholds are withheld if its price is stale.

## DEX coverage limitation: remaining Phase 1 gate

Helius address history for a mint is **not** its entire holder transfer/swap history. Scanning a few current holders also misses sold-out wallets, closed accounts, unsampled wallets and some swap representations. Therefore this implementation stores an observed swap sample, exposes its value and count separately, and leaves **total DEX volume, total unique traders, turnover and total after-hours share NULL**. It does not promote a sample to complete coverage just because pagination finished.

Only outer swaps with a fully USDC-denominated counterpart are given a dollar proxy ($1/USDC), labeled Estimated/Partial. Non-USDC swaps remain unpriced. Nested routing legs are not added again. A mint-wide transaction indexing source with auditable coverage, complete UTC windows and historical pricing is needed to finish the total-volume/unique-trader acceptance gate. Its integration must retain the present deduplication/unknown-value contracts.

The first activity pass scans at most 10 highest-balance owner wallets per asset and at most 3 pages each. Fully processed wallet cursors are stored and used for the next consecutive window. Page gaps do not advance cursors. This is incremental sampled capture, not repeated complete history download.

## Costs and operational limits

Defaults: `BACKPACK_MAX_REQUESTS=500`, `BACKPACK_MAX_HOLDER_PAGES=100`, `BACKPACK_MAX_TX_PAGES_PER_WALLET=3`, `BACKPACK_MAX_ACTIVITY_WALLETS=10`, `BACKPACK_ENABLE_QUOTES=1` in the scheduled workflow. Provider request attempts, including retries, are persisted by run. Provider-specific credit usage and dollar estimates remain N/A until a billing schedule is configured; requests are not falsely equated to credits. The worker has a 20-minute internal deadline and a 30-minute lease; Actions caps the job at 25 minutes.

All asset snapshot/holder/quote/evidence writes are in one transaction. An isolated asset failure rolls back that asset, records a failure, and continues. Already captured assets are skipped before paid requests. Rerunning an incomplete-metric but successfully captured snapshot preserves it, including its unknown fields; late corrections require an explicit future revision mechanism, not silent overwrites.

## Remaining phase work

- Phase 1 launch: official security-mint seeding, real provider credential/entitlement validation, full mint-wide swaps and unique traders, independent volume reconciliation. These are not marked complete.
- Phase 2: observed protocol/vault attribution, DeFi balances/utilization, full transfers and BP inflow/outflow monitoring, verified circulating supply and >1% alerts. Schema exists, but empty schema is not live functionality.
- Phase 2 market quality: complete after-hours aggregates, intraday parity distribution/duration, venue distribution and depth, quote quality trend views. Quote observations and calendar/parity safeguards already exist.
- Phase 3: validated announcement discovery queue, editable milestones/achievement history, anomaly alerts, competitor comparisons. The milestone table exists without a fabricated achievement panel.
- Historical backfill: requires reproducible archive sources. Live RPC capture rejects prior dates. Nothing is mislabeled as reconstructed history. No trustworthy historical security supply/holder data was available in this implementation session.

## Verification

Run:

```bash
python -m pytest tests/test_backpack.py -q
BACKPACK_TEST_DATABASE_URL=postgresql://... python -m pytest tests/test_backpack_integration.py -q
cd apps/web
npm run test:backpack
npm run typecheck
npm run build
```

Integration tests create and drop isolated schemas in an explicitly supplied **disposable** database, use deterministic providers, and exercise real SQL writes, transaction rollback, failure isolation, retry idempotency, AUM reconciliation, deduplicated portfolio holders, pending registry exclusion and lease exclusion. CI provisions PostgreSQL 16. Local integration verification used PGlite's PostgreSQL WASM engine over the PostgreSQL wire protocol; it does not replace a production Neon/provider smoke test.

Acceptance tracking: registry/storage/calculation safeguards are implemented and fixture-tested; live checks of provider outputs, official mint evidence, total swap coverage and historical charts against production captures remain open. Do not mark the overall specification complete until those gates and the subsequent phases pass.

Provider contracts: [Helius token accounts](https://www.helius.dev/docs/api-reference/das/gettokenaccounts), [Helius enhanced history](https://www.helius.dev/docs/api-reference/enhanced-transactions/gettransactionsbyaddress), [Jupiter Price](https://developers.jup.ag/docs/price), [Jupiter quote](https://developers.jup.ag/docs/api-reference/swap/v1/quote), [Alpaca SIP snapshots](https://docs.alpaca.markets/us/reference/stocksnapshots-1).

Local verification on 2026-09-22: 20 Python methodology/provider tests, 5 database integration tests, 5 TypeScript methodology tests passed; Next.js production build and targeted ESLint/type checking passed. Browser smoke checks exercised unconfigured and populated fixture states, range selection, search, mobile page width, asset details, unavailable quotes, and unauthenticated admin rejection. UI fixture values were test-only and are not committed as production observations. No production collection or deployment was performed in this session.

CI follow-up: Backpack methodology/storage also passed on GitHub's PostgreSQL 16 service. The repository-wide Python job initially exposed the missing root `exchange-calendars` dependency; it has been added to `requirements.txt`. The existing crypto integration fixture used bare `PONS`, which its registry deliberately rejects without Robinhood context; the fixture now uses the accepted `$PONS` cashtag, preserving the production matching rule. A separate Intelligence Evidence check failed against the already-deployed AML endpoint because it returned no articles; Backpack does not change that endpoint.

## Ownership continuation

Administrators can add or revise evidence-backed wallet labels at `/admin/backpack`.
Each save atomically updates the current label and appends an immutable revision.
Assigning Unknown revokes attribution for future captures. Confirmed/high system
labels exclude addresses from economic metrics; Market Maker and Unknown remain
included. Historical holder snapshots now copy entity, confidence, source and
verification time. Earlier snapshots lacking those fields say evidence was not
captured instead of borrowing current attribution.

BP whale cohorts are captured relationally at $100K, $500K and $1M by default.
`BACKPACK_WHALE_THRESHOLDS_USD` can configure additional positive USD thresholds
(comma-separated environment variable, or GitHub repository variable for the scheduled
workflow); the $100K overview baseline is always retained. New/exited whales require
consecutive complete, priced holder snapshots and can reflect price movement.
Accumulation sums token changes for yesterday's economic whale cohort, including
exits. Wallets classified as verified systems on either date are excluded from
comparative metrics so attribution changes do not masquerade as flows. These are
balance changes, not verified exchange flows or purchases. Unavailable history,
prices or holder coverage produce NULL, not zero.

Run `python backpack_monitor.py --migrate` before serving the updated reader. The
migration adds tables/columns without backfilling fabricated label evidence or
rewriting old snapshots. This continuation passes 22 Python unit tests, 6 database
integration tests and 7 TypeScript tests, including price-driven threshold crossings,
system relabeling, empty vs unknown populations, immutable evidence and reruns.

## Production startup priority (2026-09-22)

The active priority is production evidence before additional analytics. The startup
workflow runs on main changes to readiness, registry or schema code: initialize,
probe providers, revalidate/register the curated starter universe, then probe again.
A push captures only when all readiness checks pass. Scheduled/manual runs continue
to isolate failed assets; a failing preflight never promotes partial metrics to complete.
Readiness failures make the job red, even if optional snapshot fields were saved.
Sanitized JSONL evidence is retained in Actions artifacts for 30 days and readiness
checks/request counts are stored relationally. Connection/SDK exceptions are reduced
to their class names; DSNs, headers and response bodies are never emitted.

Commands: `python backpack_monitor.py --migrate --preflight`, `--seed-starter`,
`--execute`, then `--audit`. The audit uses stored observations only and reports
supply slots, holder reconciliation, reference/on-chain price timestamps, reproduced
AUM and quality events. It never marks human sign-off complete. Review raw provider
observations and official mint evidence before accepting the first production dataset.

The curated starter manifest contains 14 exact identities: AAPL, MSFT, NVDA, AMZN,
GOOGL, META, TSLA, AVGO, MU, AMD, MSTR, INTC, SNDK and QQQ. Sources:
- https://api.backpack.exchange/api/v1/assets (exact Solana contract by asset ID)
- https://api.backpack.exchange/api/v1/securities (explicit securities IDs and names)
- https://docs.backpack.exchange/ (stock API interpretation)
- https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt (exchange/type verification)
- https://learn.backpack.exchange/articles/what-is-backpack-securities (issuer context)
- https://learn.backpack.exchange/articles/top-stocks-on-solana-to-trade (1:1 redemption)
- https://learn.backpack.exchange/articles/what-is-bp-backpack-token (exact BP mint)

A runtime mismatch, missing mint or different decimals prevents registration of that
asset; other assets continue. Existing registry entries are preserved, not silently
re-approved. No launch dates are invented; disabled withdrawals are not interpreted
as zero issuance or zero demand. Starter scope is intentionally not the entire market.

After the first production audit: complete token-wide swaps, add AUM composition,
then implement an issuer-neutral competitor layer for Backpack/xStocks/Ondo. Do not
create issuer-specific parallel history tables. Category ratios require comparable
scope, complete denominators, aligned timestamps and deduplicated wallets. Signed
net issuance shares are N/A when the category denominator is zero/negative or absent.
Only then add configurable 7D/30D adoption/churn states; unavailable complete volume
means classification is unavailable, never Stagnant. Keep DeFi, extended parity and
liquidity analytics behind the uninterrupted daily capture priority.

### First production startup result — 2026-09-22 02:36 UTC

Run: https://github.com/themvf/Speeches/actions/runs/35680044903

- Production Postgres schema initialized; connection validated.
- All 14 curated securities registered after exact official API checks and finalized
  on-chain mint/decimal validation (slots 449245491–449245507).
- Secondary public Solana RPC and exact-base BP DexScreener validation succeeded.
- Helius, Jupiter and Alpaca SIP credentials were absent in the production Actions
  environment. Required secret names: `HELIUS_API_KEY`, `JUPITER_API_KEY`,
  `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` (SIP entitlement required).
- Startup capture was skipped; no dataset was signed off and no missing price/holder
  observations were replaced with zero. After secrets are configured, run the
  Backpack workflow manually and review its reconciliation artifact before adding
  analytical layers. Do not place secrets in source, chat, reports or workflow YAML.
- Initialization exposed ambiguous `ORDER BY date` expressions in the web reader
  because both raw and formatted dates had the same output name. Date ordering now
  uses qualified table columns. Integration tests execute the actual web SQL
  templates against both empty and populated databases.

## Cost controls (September 2026)

The daily worker retains permanent thesis aggregates, registry evidence, ingestion runs,
quality events, wallet-label revisions and analytical records. It now maintains one
`backpack_current_holders` row per live asset/owner and permanent `backpack_holder_events`.
The first complete enumeration establishes a baseline: it does not claim all wallets
arrived that day. Subsequent complete enumerations record new/exited owners, token balance
changes, bidirectional USD threshold crossings and system-label/confidence changes.
Price-driven threshold crossings are not token accumulation. Gaps retain the previous
observation date. Failed/incomplete enumerations never replace current state.

Full holder snapshots default to 30 days (`BACKPACK_HOLDER_RETENTION_DAYS`, allowed 30–90).
Deletion requires a permanent supply-reconciled holder checkpoint matching the persisted
owner aggregate. Legacy/unvalidated snapshots are preserved. Deletes are bounded to
10,000 rows per maintenance invocation; monitor backlog before raising throughput.
Aggregate history, quality evidence and holder events are never expired. Exact daily wallet
reconstruction outside retention is not promised, particularly for price-only changes.
First-seen means first observed in the current continuous holding spell, not wallet creation.

Raw swaps have the same configurable 30–90-day policy (`BACKPACK_SWAP_RETENTION_DAYS`).
Deletion requires an explicit per-asset/activity-day aggregate validation attestation in
`backpack_transaction_retention_checks`. The current sampled collector does **not** issue
market-wide attestations, so its transactions are preserved pending review. Pin material
signatures in `backpack_transaction_evidence` before attesting a day; their raw rows survive.
Anomaly/whale/liquidity/investigation selection must be implemented with market-wide capture
before enabling automated attestations. Ordinary transfers are still not swaps and routed
swaps remain deduplicated by asset/signature before persistence.

Public API cache headers are now `s-maxage=3600, stale-while-revalidate=86400`, with an
additional tagged Next.js database-result cache lasting one hour. No polling or ingestion
runs in public requests. A POST to `/api/market/crypto/backpack/revalidate` requires
`Authorization: Bearer <BACKPACK_REVALIDATE_SECRET>`. Set the same dedicated secret in
GitHub Actions and Vercel. Both scheduled and manually dispatched captures notify it after
capture and audit steps succeed. This invalidates the Next.js data/page cache; previously
cached CDN responses can remain through their HTTP cache lifetime. Missing/rejected hook
credentials are reported safely and retain the TTL fallback. Immediate global CDN purging
is not claimed. The response timestamp remains the actual observation date.

Daily ingestion precomputes 7/30/90-day issuance and growth and reconciled top-1/3/5/10
AUM composition. Missing periods stay unavailable; composition excludes BP. The asset table
reads these stored values. Daily multi-asset adoption was already computed during capture.
Competitor share and a churn classification remain unavailable until their source coverage
is adequate. No bullish score has been introduced.

Run `python backpack_monitor.py --maintenance --cost-report` after migration. The daily
workflow persists per-table allocated/table/index bytes, individual index sizes, approximate row counts, total
database bytes and a point-in-time connection count, then saves a sanitized cost JSONL
artifact. PostgreSQL allocation includes reusable space: DELETE does not immediately reduce
allocated bytes. Database bytes/connections include the shared database, not just Backpack.
Provider request counts include retries; readiness requests are reported separately.
API function logs contain response byte counts and overview/asset scope, never payloads or
wallets. CDN-hit traffic does not invoke this logging and requires platform analytics.

Configure `BACKPACK_DATABASE_ALLOWANCE_BYTES` to alert above 50% of actual plan storage.
Additional persistent operational alerts cover >20% 30-day table growth, raw transaction
rows (`BACKPACK_RAW_TRANSACTION_ROW_ALERT`, default 1M), full holder rows
(`BACKPACK_HOLDER_ROW_ALERT`, default 2M), provider request budgets and maintenance failures.
These are operational records, not investment alerts. Row estimates depend on PostgreSQL
statistics; review actual allocation as well. No notification channel is connected.

After seven and thirty observed days, inspect exact-date growth in `--cost-report`, largest
tables/index totals and provider counts. Neon CU-hours/egress and Vercel invocations, CPU,
bandwidth/cache-hit rates require authorized platform usage exports; these and projected
monthly cost remain null until available. No claim of <$5/month is made without billing
evidence. Billing import and automated cost projection remain
follow-up work. The monitor still needs the production Helius, Jupiter and Alpaca credentials
before its first reconciled capture. The cache hook additionally needs its shared secret.

## Provider-free operating tools and research foundation

`/admin/backpack` now presents the latest **stored worker** readiness checks, their timestamps,
recent runs, per-asset holder/AUM reconciliation and storage alerts. It does not test credentials
from the web runtime. Checks older than 48 hours display Stale. Administrator authentication is
required and responses are not publicly cached. Reload reads stored evidence; it does not dispatch
a capture. Capture completion does not confer human sign-off. The workflow audit artifact remains
the first-capture review record.

Issuer-neutral tables: `tokenized_security_issuers`, `tokenized_security_assets`,
`tokenized_security_daily_snapshots`, `tokenized_security_market_snapshots` and relational
`tokenized_security_market_members`. Backpack, xStocks and Ondo are issuer labels only. Only
approved Backpack security identities are mirrored automatically; BP and pending entries are
excluded. Run `python backpack_monitor.py --research` to process stored records without provider
calls. Daily collection and maintenance also run this step. It preserves captured dates and
sources and never manufactures observations for competitors. A category denominator requires
an explicitly defined member universe and comparable verified observations; no denominator is
populated automatically. Per-asset holder counts must not be summed into category unique users.
Storage and index monitoring includes both Backpack and issuer-neutral tables. Competitor adapters, approved competitor identities and deduplicated category wallets remain future work.

The environment panel uses persisted, versioned 7D/30D observations with the actual inputs displayed.
It compares issuance / starting AUM, AUM/meaningful-holder growth, and the current trading window
against its preceding equal-length window. It requires 14 or 60 consecutive daily observations,
the same asset identities throughout, complete DEX volume and a positive trading baseline.
Every equity reference must be timestamped and no more than four days old at capture (closed-market
references retain their limitations). Partial wallet-sampled volume is never substituted.
Default thresholds: issuance >1% of starting AUM, AUM and holder growth >1%, trading growth >20%.
GitHub repository variables `BACKPACK_ISSUANCE_GROWTH_PCT`, `BACKPACK_ADOPTION_GROWTH_PCT` and
`BACKPACK_TRADING_GROWTH_PCT` configure future captures. Values and methodology are stored on
historical records; changing configuration never rewrites a past classification. Expansion and
Accumulation require all three adoption signals; Churn and Stagnant require all three below their
thresholds. Conflicting adoption signals yield Mixed. Stagnant may include contraction. Missing
inputs yield Unavailable; old observations display Stale. These are descriptive rules, not forecasts.

The public evidence guide explains registry scope, source statuses, primary-source mint links,
reference timestamps, price-driven AUM changes and wallet-versus-investor limits.

### Billing import and 7/30-day review

Use a local normalized CSV exported/prepared from authorized billing evidence (no provider APIs
are called): `python backpack_monitor.py --import-billing billing.csv --cost-report`.
Header: `date,provider,scope,metric,value,unit,source`.
Dates represent **UTC daily totals or daily measurements**, not overlapping monthly invoice totals.
Providers: Neon, Vercel, Helius, Jupiter, Alpaca. Scope is `backpack` only for documented attribution;
otherwise use `shared`. Source is a credential-free HTTPS invoice/export reference, without query
parameters. Daily keys are immutable. Identical re-imports are idempotent; conflicting values,
units or provenance reject the whole import. An incorrect import requires a reviewed correction
migration rather than silently rewriting evidence.

Supported metric/unit pairs:

| Metric | Unit |
| --- | --- |
| cost_usd | USD |
| compute_cu_hours | CU-hours |
| egress_bytes | bytes |
| api_requests | requests |
| function_invocations | invocations |
| active_cpu_seconds | seconds |
| cache_hit_pct | percent |
| average_payload_bytes | bytes |
| credits | credits |

`--cost-report` includes the imported measurements with attribution/provenance, provider requests,
largest tables/indexes, alerts and 7/30-day review windows. Storage-growth projections require
all daily storage observations including the baseline date; allocated bytes are not billed bytes.
A 30-day dollar scenario requires explicit Backpack-attributed Neon **and** Vercel cost rows for
every day of the review window, including explicit zero-cost rows. It extrapolates observed daily
charges; it does not forecast traffic or apply invented provider rates. Shared-account charges,
API-provider charges and infrastructure charges remain separate. No cost assertion is made without
complete attribution. Cache-hit and payload-size values remain dated measurements, not unweighted
averages of incompatible exports. New export formats require normalization to this contract.

## Growth overview: adoption direction, independent of trading

This section describes the retained financial `growth-v1` method, not the current lead headline. See the price-independent amendment below.

The lead overview answers whether the **tracked Backpack securities ecosystem** is growing,
growing but slowing, declining, or mixed/flat. BP ownership remains a separate card. This is
independent of the trading/adoption quadrant: missing market-wide DEX volume does not prevent
an adoption assessment. Trading, BP price and FDV cannot promote the growth headline.

`backpack_growth_daily` preserves dated 7/30/90-day assessments, inputs, thresholds and methodology.
Direction needs 8/31/91 consecutive complete daily observations respectively; adjacent-window
momentum needs 15/61/181. Both windows must contain the same asset identities. Complete meaningful
holder metrics, daily supply deltas, positive starting AUM/holder baselines, and timestamped equity
references within four days of capture are required. Unknowns stay unavailable, and a capture older
than 48 hours displays Stale evidence in the UI. Original daily observations and assessments are
never overwritten by a rerun or threshold change.

Default direction thresholds: net issuance / starting AUM >0.1% and meaningful-holder growth >1%
means Growing. Both below the corresponding negative thresholds means Declining over the selected
window. Disagreement or smaller movements means Mixed / flat. Growing, but slowing requires both
issuance/AUM and holder growth to weaken by >0.25 percentage points versus the previous equal-length
window. When contraction becomes less severe, momentum says Contraction easing, not accelerating
growth. Missing prior history explicitly leaves momentum insufficient even when direction is known.
GitHub variables `BACKPACK_GROWTH_ISSUANCE_PCT`, `BACKPACK_GROWTH_HOLDERS_PCT` and
`BACKPACK_GROWTH_SLOWDOWN_PP` configure future assessments; actual thresholds are stored per day.

Three mechanical explanations expose issuance, meaningful-holder changes and adjacent-period
momentum. The depth card compares top-five AUM share, multi-asset adoption and securities with at
least `BACKPACK_SIGNIFICANT_AUM_USD` (default $1M) AUM. Historical endpoints are recomputed with the
same threshold during that assessment. This is breadth, not evidence of liquidity or DeFi support.
New/departing ecosystem wallets are not inferred by summing asset counts. BP's separate ownership
card compares meaningful holders and top-20 economic concentration at exact dates; USD cohorts
can move with token price and system-label revisions can change concentration.

Endpoint AUM attribution uses `(S1-S0)*P0` for supply and `S1*(P1-P0)` for price for each security.
These sum exactly to endpoint AUM change, assigning the cross-term to price. They intentionally
differ from net issuance valued at each daily reference price; none are described as verified
customer deposits. Network growth can diverge from BP price, and no buy/sell score is generated.

## Price-independent adoption amendment — 2026-09-23

Review basis: this document and implementation/acceptance history in PRs #127–#133.
The original foundation measures capital, ownership and financial use separately;
the later overview explicitly excludes trading and BP price from the network headline.
The user's subsequent clarification removes stock-price dependence from that headline.
No original 33-section source specification is stored in this repository; this amendment
does not claim to complete the remaining DEX, DeFi, circulation or competitor phases.

The lead question is whether **tracked Backpack securities adoption** is growing rapidly,
growing slowly, status quo or declining. It does not measure Backpack exchange customers,
deposits, revenue or all Backpack products. BP ownership remains separate.

`backpack_adoption_daily` stores permanent, small, price-independent daily summaries:
deduplicated nonzero owners, wallets with at least two securities, wallets with at least
one whole token in any security, and each security's supply and non-system owner count.
Any wallet excluded as a verified system on any tracked security is excluded globally.
An exclusion-set fingerprint prevents label changes from being interpreted as growth.
Raw owner rows are not duplicated in the summary. All asset enumerations must have been
marked complete and supply-reconciled, and retained owner counts must match snapshots.
Partial/expired raw history cannot generate a summary. Existing summaries survive raw retention.

`backpack_adoption_assessments` preserves 7/30/90-day results, actual inputs, thresholds,
method version and explanations. Each window needs 8/31/91 consecutive complete daily
summaries for exactly the same securities and exclusions. Adjacent-window momentum needs
15/61/181. Consistently unissued (zero-supply) securities remain registered but are excluded
from supply-growth and breadth denominators. First issuance needs a positive comparable
baseline; full redemption to zero is measurable contraction. Missing days, invalid holder
baselines and cohort changes remain insufficient
evidence. Summaries and assessments are immutable; `--research` can derive them from retained
historical observations without provider calls or changing original snapshots.

Supply growth is calculated separately for each security and combined using the median;
unlike token units are never summed. The median describes a typical security, not total
capital. Holder and median supply changes are normalized linearly to a 30-day equivalent
(observed percent × 30 / period); this is a comparison convention, not a forecast.

Default rules (configurable GitHub variables, stored with each assessment):

| Variable | Default | Meaning |
| --- | --- | --- |
| `BACKPACK_ADOPTION_HOLDER_PCT_30D` | 1 | Positive/negative ownership noise band |
| `BACKPACK_ADOPTION_SUPPLY_PCT_30D` | 0.1 | Positive/negative median supply noise band |
| `BACKPACK_ADOPTION_RAPID_PCT_30D` | 10 | Rapid ownership growth threshold |
| `BACKPACK_ADOPTION_BREADTH_PCT` | 60 | Minimum share of securities confirming growth/contraction |
| `BACKPACK_ADOPTION_MOMENTUM_PP` | 0.25 | Minimum joint change in rates for momentum |

- Growing slowly: holders and median supply exceed positive bands, and at least 60% of
  securities gain holders and at least 60% expand supply above its band.
- Growing rapidly: those conditions plus ownership growth at least 10% on the normalized basis.
- Declining: both rates below negative bands and at least 60% confirm each contraction signal.
- Status quo: both aggregate rates within the bands. Individual securities may offset;
  breadth is always displayed.
- Mixed: conflicting/insufficiently broad signals. If nonzero ownership grows while the
  one-token holder comparison is available but flat/negative, growth is downgraded to Mixed.
- Momentum: both normalized rates increase/decrease more than 0.25 points against the
  previous equal-length window. Rapid growth can be Slowing; contraction can be easing.

These are explicit research conventions, not validated universal definitions of rapid growth.
Nonzero wallets are **not** renamed meaningful holders: existing $100/$1K cohorts retain their
definitions and require price evidence. One-token sensitivity is not a dollar or economic cutoff.
Dust, Sybil wallets, unlabeled custody and internal transfers can distort adoption. No retention
rate or unique-person claim is inferred from aggregate owner counts.

Each new daily summary also records identity-free 1/7/30-day ecosystem endpoint cohorts while
the required raw evidence remains inside retention. The calculation deduplicates owners across
all registered securities before counting retained, entered and departed wallets, and it requires
the same asset identities and exclusion fingerprint at both endpoints. Only the counts and
retention percentage are persisted; wallet addresses are not copied into the summary. "Entered"
means present now and absent at the earlier endpoint, not a newly created person or first-ever
Backpack customer. Per-asset arrivals and exits are never summed into ecosystem cohorts.

The UI shows real baseline counts before enough history exists, explains the observation
requirement, defaults to 7 days, and leads its security table with supply/owners/token deltas.
Original capital/trading/DeFi panels remain available in supporting evidence. Missing chart
days are explicit gaps. New schema and worker derivation must run before the new reader is populated.
