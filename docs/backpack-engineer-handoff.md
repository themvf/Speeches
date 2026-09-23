# Backpack Thesis Monitor — engineering handoff

Status checked: **2026-09-23 UTC**. Repository: [themvf/Speeches](https://github.com/themvf/Speeches).

## Product objective and scope

The user's priority is: **Is Backpack growing rapidly, growing slowly, holding steady, or declining?** Stock prices are not their focus and must not block the lead adoption assessment.

The implemented observable scope is **on-chain adoption of the registered Backpack securities**, with BP token ownership separate. Do not describe wallet counts as people, exchange customers, deposits, revenue, or growth of the entire Backpack company.

Read [backpack-monitor.md](backpack-monitor.md), especially the **Price-independent adoption amendment**, before changing calculations. Historical design/acceptance context is in PRs [127](https://github.com/themvf/Speeches/pull/127)–[133](https://github.com/themvf/Speeches/pull/133). The original referenced 33-section specification was not found in the repository; the implementation document and those PRs are the available record. The latest user clarification supersedes the earlier price-required growth headline, not the evidence safeguards or separate financial metrics.

## Shipped status

- [PR #134](https://github.com/themvf/Speeches/pull/134) merged to `main` as [`b4ca522`](https://github.com/themvf/Speeches/commit/b4ca522cf58add4301d053f8ae68f34327411602).
- Public screen: [Backpack Thesis Monitor](https://speeches-zeta.vercel.app/market/crypto/backpack).
- Admin screen: `/admin/backpack`; public API: `/api/market/crypto/backpack`.
- The headline now uses ownership, per-security supply growth and breadth, independent of stock prices, BP price and trading volume. Financial/trading/DeFi panels remain in supporting evidence.
- Current baseline counts display before a growth window is available. The default comparison is 7 days; 30 and 90 are selectable. Momentum is separate from growth speed.
- [Production derivation run](https://github.com/themvf/Speeches/actions/runs/35807982294) succeeded. It preserved all 15 existing snapshots (`15 skipped`, `0 failed`) and derived the new adoption records from their stored evidence. This was not a second collection or a backfill from current balances.
- Vercel reported successful production deployment for the merge. Production audit records were verified; the deployed screen was not independently browser-inspected after that deployment. Local browser fixtures were checked.
- [PR #135](https://github.com/themvf/Speeches/pull/135) merged to `main` as [`6aca5f9`](https://github.com/themvf/Speeches/commit/6aca5f97dc372be8e6bb8c6a4444cc00882e1f6c). It adds this handoff and handles registered securities with consistently zero supply: omit them from percentage-growth/breadth denominators, keep the registered cohort stable, and require a positive baseline when a security first issues. A full redemption to zero remains measurable as contraction.
- The [follow-up production workflow](https://github.com/themvf/Speeches/actions/runs/35808509992) and Vercel production deployment both completed successfully after PR #135.

## Verified production baseline

Observation date: **2026-09-23 UTC**, from `backpack-audit.jsonl` in the production run linked above.

| Observation | Value | Interpretation |
| --- | ---: | --- |
| Registered securities | 14 | Curated scope, not proof of complete Backpack coverage |
| Securities with positive captured supply | 5 | MU.US, AMD.US, MSTR.US, INTC.US, SNDK.US |
| Registered securities with zero captured supply | 9 | Observed zero supply; do not substitute missing data for these zeros |
| Deduplicated nonzero security owners | 14,251 | Not people or economically meaningful accounts |
| Owners holding at least two securities | 939 | Positive balances, deduplicated across assets |
| Owners with at least one whole token of any security | 240 | Dust sensitivity; not a dollar-value cutoff |
| Complete adoption days available | 1 | All 7/30/90-day assessments correctly remain Insufficient evidence |

The exclusion fingerprint corresponds to an empty excluded-wallet set in this baseline. **No observed holders were excluded as verified systems.** That does not mean there are no treasury, custody, pool or other system wallets. Attribution remains incomplete. The large difference between nonzero and one-token counts makes dust/distribution review especially important.

BP was collected separately. Its dollar-threshold cohorts and concentration must not be blended into securities adoption. Raw and economic concentration can coincide when verified system exclusions are absent.

## How growth is calculated

Implementation: `backpack/adoption.py`; methodology identifier: `adoption-v1`.

1. Require all expected securities to have complete, supply-reconciled holder snapshots. Retained raw owner counts must match the recorded snapshots.
2. Deduplicate positive-balance owners across securities. If a wallet is a verified excluded system for any security that day, exclude it globally.
3. Persist daily counts, per-security supply/holder counts and an exclusion fingerprint. No permanent duplicate wallet-address list is created.
4. Require consecutive daily summaries with identical security IDs, methodology and exclusions. Never compare cohorts merely because their counts match.
5. Calculate each issued security's percentage supply change separately, then use the median. Never add token quantities across different securities. Consistently zero-supply securities do not dilute breadth; new issuance needs a positive comparable baseline.
6. Normalize ownership and median supply percentage changes to a 30-day equivalent using `observed change × 30 / period`. This is a comparison convention, not a forecast or annualized return.
7. Apply the disclosed rules below. Keep Mixed and Insufficient evidence rather than forcing one of the four headline directions.

| State | Default rule |
| --- | --- |
| Growing slowly | Holder growth >1% and median supply growth >0.1% on the normalized basis; at least 60% of issued securities gain holders and at least 60% expand supply beyond its band |
| Growing rapidly | All slow-growth conditions, plus normalized holder growth ≥10% |
| Declining | Both rates below their negative bands, with at least 60% confirming each contraction signal |
| Status quo | Both aggregate rates within the noise bands; individual securities may offset |
| Mixed | Signals disagree, breadth fails, or growth is confined to tiny balances when the one-token comparison is available and flat/negative |
| Insufficient evidence | Missing/incomplete days, changed cohort/exclusions, or invalid comparison baseline |

Momentum compares adjacent equal-length windows. Both normalized rates must move together by more than 0.25 percentage points to report Accelerating or Slowing. Declining periods use Contraction easing/deepening. Rapid growth can still be slowing.

These thresholds are transparent research conventions, **not empirically validated universal definitions**. The one-token sensitivity is not a replacement for the existing $100 meaningful-holder definition.

| Window | Complete observations for direction | Complete observations for momentum |
| --- | ---: | ---: |
| 7 days | 8 | 15 |
| 30 days | 31 | 61 |
| 90 days | 91 | 181 |

Do not promise a classification on a particular calendar date: gaps, new issuance, asset changes and attribution changes can delay comparability.

## Code and data map

| Location | Responsibility |
| --- | --- |
| `.github/workflows/backpack-monitor.yml` | Daily 00:30 UTC schedule, manual dispatch, migration, readiness, registration, capture, audit, maintenance |
| `backpack_monitor.py` | CLI operations |
| `backpack/collector.py` | Immutable daily capture, leases, reconciliation and asset isolation |
| `backpack/providers.py` | Helius, Jupiter, Alpaca, public RPC and DexScreener adapters |
| `backpack/adoption.py` | Price-independent summaries and classifications |
| `backpack/research.py` | Calls adoption derivation; retains existing financial/trading research |
| `backpack/growth.py` | Original price-dependent financial growth method; retained separately |
| `backpack/readiness.py` | Provider checks and audit, including adoption records |
| `backpack/storage.py` | Current owners, change events, bounded retention and storage observations |
| `sql/backpack.sql` | Additive schema, including `backpack_adoption_daily` and `backpack_adoption_assessments` |
| `apps/web/lib/server/backpack-store.ts` | Read-only database/API payload; tolerates absent new schema |
| `apps/web/components/backpack/adoption-overview.tsx` | Active adoption headline, baseline, chart and rules |
| `apps/web/components/backpack/growth-overview.tsx` | Earlier financial overview component; no longer imported by monitor |
| `apps/web/components/backpack/monitor.tsx` | Securities table, supporting evidence, BP and provenance |
| `tests/test_backpack_adoption.py` | Price-independent calculation regressions |
| `tests/test_backpack_integration.py` | Real PostgreSQL persistence, retention, API SQL and immutability tests |

Summaries and assessments are insert-only. Original snapshots are never updated on rerun. For an actual methodological correction, design an explicit version/revision policy; do not silently overwrite historical assessments or relabel older results as newly calculated.

## Configuration and operations

GitHub Actions secrets already present: `DATABASE_URL`, `HELIUS_API_KEY`, `JUPITER_API_KEY`. The user cannot provide Alpaca credentials. **Do not make Alpaca a prerequisite again or add stock-price fallback work as a priority.**

`BACKPACK_REQUIRE_EQUITY_REFERENCE=0` is the workflow default. All other readiness checks still apply. Local preflight defaults to strict mode, so set this variable explicitly for local limited-data operation. `ready_for_capture` controls capture readiness; `ready_for_security_capture` still includes optional full financial readiness and can remain false.

Optional secrets:

- `SOLANA_VALIDATION_RPC_URL`: public Solana RPC fallback currently works.
- `BACKPACK_REVALIDATE_SECRET`: absent. Collection succeeds; authenticated immediate cache refresh is unavailable. One-hour cache fallback remains, with `stale-while-revalidate=86400` permitting older responses while refresh runs. If configured later, use the same value in Actions and Vercel.

Assessment repository variables, with defaults:

```text
BACKPACK_ADOPTION_HOLDER_PCT_30D=1
BACKPACK_ADOPTION_SUPPLY_PCT_30D=0.1
BACKPACK_ADOPTION_RAPID_PCT_30D=10
BACKPACK_ADOPTION_BREADTH_PCT=60
BACKPACK_ADOPTION_MOMENTUM_PP=0.25
```

For normal production operation, dispatch the existing workflow instead of copying production credentials locally:

```sh
gh workflow run backpack-monitor.yml -R themvf/Speeches --ref main
gh run list -R themvf/Speeches --workflow backpack-monitor.yml --limit 5
```

The worker's command sequence is migration/preflight, verified starter registration, preflight, capture, audit, research/maintenance/cost reporting. `python backpack_monitor.py --research` derives missing summaries from retained stored evidence without provider requests. `--execute` only captures the current UTC date; it cannot reconstruct past balances. Repeated runs preserve captured assets.

Download the `backpack-readiness-<RUN_ID>` artifact and inspect its readiness, registry, audit and cost JSONL files. A green workflow means execution completed, not that every metric exists or the research thesis has been validated. Inspect `adoption`, `adoption_assessments`, holder reconciliation and per-metric limitations.

## Validation completed

- PR #134: 70 Python tests including PostgreSQL 16 integration; 8 TypeScript tests; all final CI gates passed.
- PR #135: 72 Python unit/PostgreSQL integration tests and 8 TypeScript tests passed on its final commit; all PR checks and Vercel preview passed. This includes dormant-registration and full-redemption regressions.
- Type checking, targeted ESLint, local production build and Vercel preview/production builds passed.
- Local browser fixtures checked baseline counts, 7/30-day switching, rapid-but-slowing states, runtime errors and mobile overflow. Screenshots inspected; fixtures were never persisted to production.
- Production run verified the real baseline above and immutable reuse of 15 snapshots.
- An existing `next/font/google` failure blocked both local and Vercel builds. PR #134 bundles the same IBM Plex Sans and Space Grotesk weights using pinned Fontsource packages and `next/font/local`. No font mock is committed or required in production.

Run relevant verification:

```sh
python -m pytest tests/test_backpack.py tests/test_backpack_adoption.py -q
# Set BACKPACK_TEST_DATABASE_URL only to a disposable PostgreSQL database:
python -m pytest tests/test_backpack_integration.py -q
cd apps/web
npm ci
npm run test:backpack
npm run typecheck
npm run build
```

Integration tests skip without the explicit disposable database variable. CI provisions PostgreSQL 16. Do not aim those tests at production.

## Remaining steps, in priority order

### 1. Establish trustworthy daily adoption history

- Verify successive scheduled runs actually arrive; GitHub schedules are best effort.
- Check no missing/incomplete day or provider budget failure interrupts the series.
- Watch the first real issuance of any currently unissued security: confirm comparison windows wait for a positive baseline and do not treat new coverage as comparable growth. The zero-supply handling and redemption regressions are already on `main`.
- At the first eligible 7-day window, manually reconcile deduplicated holders, per-security supply changes, breadth and the classified state against retained observations.
- Show insufficient evidence until requirements are satisfied. Do not manufacture earlier history from today's balances or stock-price history.

### 2. Improve ownership interpretation before trusting rapid-growth labels

- Review why 14,251 nonzero owners reduce to 240 one-token holders. Investigate token distributions, dust, wallet splitting and custody structure.
- Add source-backed treasury/custody/pool/vesting labels through the existing admin flow. Never infer a named owner from balance size alone.
- Be explicit that changes in the exclusion set temporarily suppress comparable windows.
- Consider a reviewed retention/new-lost ecosystem cohort measure using historical evidence; current aggregate counts alone cannot establish retention. Do not sum per-asset arrivals/exits to create ecosystem arrivals/exits.
- Calibrate classification thresholds against observed periods and retain methodology versions. Do not silently change historical labels when tuning thresholds.

### 3. Improve provider robustness

- The first capture recorded Jupiter HTTP 429s and some unavailable token prices/routes. Add plan-aware pacing and bounded Retry-After handling; distinguish an unavailable market from throttling.
- Keep Helius request/time budgets and holder reconciliation intact. No provider upgrade is assumed authorized or necessary for the current headline.
- Optional cache-secret setup improves freshness but is not a prerequisite for adoption capture.

### 4. Finish separate thesis dimensions only after daily adoption is reliable

- **Complete DEX activity:** the existing current-wallet sample is not token-wide volume. Implement an auditable index covering relevant token accounts/programs, historical participants and complete UTC windows; deduplicate routed swaps and reconcile independently. Never promote sampled volume to complete totals.
- **DeFi:** verify protocol/vault/pool identities and implement protocol-specific balances/utilization. Empty schema is not a working collector.
- **BP circulation/flows:** obtain verifiable treasury/vesting/circulating evidence and classifications. Total supply is not circulating supply; FDV is not market cap; large associated balances are not necessarily staking.
- **Competitors:** verify and collect comparable Ondo/xStocks observations through the existing issuer-neutral model. Registry labels alone are not competitor history or a market-share denominator.
- **Launch dates/milestones:** source verified announcements; do not equate the registry entry or first observation with launch.

### 5. Operational cleanup

- Add explicit daily-capture freshness alerting if requested; none was created by this change.
- Review API payload size as adoption history grows (reader currently caps summaries at 5,000 days).
- Maintain raw-data retention and permanent summaries; do not retain duplicate daily wallet lists just to draw charts.
- Use billing exports before making cost claims. Provider request counts are not billed credits.

## Guardrails for the next engineer

Keep BP and securities separate. Keep observed zero distinct from unavailable. Keep system labels evidence-backed. Keep wallet counts distinct from investors. Keep financial metrics optional for the headline. Preserve exact asset identities, decimal supply arithmetic, immutable observations, historical label provenance and same-cohort comparisons. Acknowledge remaining phases rather than presenting all original thesis requirements as complete.
