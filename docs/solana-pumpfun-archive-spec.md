# Solana / Pump.fun Graduation Archive — feasibility assessment and V1 plan (2026-09-19)

Extends the Robinhood Chain archive (`docs/graduation-archive-spec.md`, `launchpad_archive.py`) to
Solana. Same philosophy: record the population first, derive thresholds from outcomes later, make
censoring structural, never treat graduation as success.

The research question is **what happens after migration** — which graduates retain momentum, which
fade, and whether the difference is visible in the first minutes.

---

## 1. Technical feasibility assessment

**Verdict: feasible, but not as a complete census of launches.** A complete *launch* archive is out
of reach with the free public feed; a complete or near-complete *graduate* archive is reachable at a
two-minute cadence; and a statistically sound answer to every question asked needs only an unbiased
**sample** of graduates, which is comfortably affordable.

All of the following was measured live on 2026-09-19, not assumed.

| Measured | Solana | Robinhood (for contrast) |
|---|---|---|
| New pools per minute (all DEXes) | **25/min** | 8-13/min |
| `new_pools` depth, all 10 pages | **5.8 minutes** | 11-14 minutes |
| Pump.fun share of new pools | 36 of 60 (60%) | n/a |
| PumpSwap pools in the same sample | 5 of 60 | n/a |
| `launchpad_details` on curve tokens | **present and identical** | present |
| Holder count via `/info` | **partial** — 1 of 3 sampled tokens had it | present |

**The good news: the adapter pattern transfers.** Solana has the same curve/graduate DEX pairing the
Robinhood design rests on — `pump-fun` is the bonding curve, `pumpswap` is where graduates land —
and `launchpad_details` is byte-identical in shape (`graduation_percentage`, `completed`,
`completed_at`, `migrated_destination_pool_address`). A live pump.fun token read
`{"graduation_percentage": 6.32, "completed": false, ...}`. Parsing, graduation detection by
arrival, and the rung ladder all work unchanged.

**The hard constraint: the feed is 5.8 minutes deep.** Ten pages × 20 pools is 200 pools, and at 25
pools/minute that is under six minutes of history. Sweeping every five minutes — the Robinhood
cadence — would leave essentially no margin, and any delayed run would lose launches permanently.
There is **no DEX-scoped newest-first feed** to escape this with: `/dexes/pumpswap/pools` sorts only
by `h24_volume_usd_desc` or `h24_tx_count_desc` (a `pool_created_at_desc` sort returns HTTP 400),
and `new_pools?dex=pumpswap` **silently ignores the filter** — it returned a mixed-DEX page. So the
global feed at 10 pages is the only newest-first view available.

### The finding that would have silently corrupted the dataset

`migrated_destination_pool_address` **must not be trusted on Solana.** For a token that graduated at
19:18:02, the documented destination pool and the pool where trading actually happened were
different addresses with wildly different activity:

| Pool | Source | Liquidity | Trades (m5) | Buyers (m5) |
|---|---|---|---|---|
| `FramDv5…` | `migrated_destination_pool_address` | **$0.43** | 3 buys | 2 |
| `ALPZYXZB…` | the token's deepest pool (pumpswap) | **$30,562** | 1,964 buys | **1,088** |

Following the documented field would have measured a dead pool for every rung and produced a
dataset showing that essentially all graduates die instantly — a confident, completely wrong
finding. Listing the token's pools showed three: the live pumpswap pool, plus two empty ones on
`meteora-damm-v2` and `meteora-dbc` created seconds earlier.

**Rule: after graduation, measure the token's deepest pool by liquidity, and record which pool was
chosen and why.** The destination field is kept as metadata, never as the measurement target.

That example also shows why "Pump.fun" cannot be assumed from a token's presence alone: this token
carried a `meteora-dbc` curve pool, so its `launchpad_details` describes Meteora's lifecycle, not
Pump.fun's. Solana hosts several launchpads with their own curve→graduate pairings (`pump-fun` →
`pumpswap`, `meteora-dbc` → `meteora-damm-v2`, `raydium-launchlab`, `boop-fun`, `moonshot`).
**Launchpad identity must come from the curve pool's DEX id**, and be stored per token.

### BOOST is visible in the data

That same graduate took **1,964 buys from 1,088 distinct buyers in its first five minutes**. Whatever
the mechanism's current parameters, the first rung is clearly not a free market reading, which is
exactly why it must be recorded as its own rung and treated as structurally different in analysis —
see §6.

---

## 2. Recommended data sources

**GeckoTerminal for V1**, for the same reasons it worked on Robinhood: free, no key, no credential
to manage, already wrapped in tested code, and it carries the fields the questions need — price,
liquidity, volume at m5/m15/m30/h1/h6/h24, and `transactions` with **distinct buyers and sellers**
per interval, which is what "still actively traded" actually means.

What it does **not** reliably give, and where a second source is eventually needed:

| Need | GeckoTerminal | Alternative |
|---|---|---|
| Holder count / concentration | **partial** (1 of 3 sampled) | Solana RPC `getTokenLargestAccounts` + `getTokenSupply`, or Helius/Bitquery |
| Complete launch census | no (5.8-minute feed) | Bitquery streaming, or a Solana RPC log subscription on the Pump.fun program |
| Exact migration events | derived, not subscribed | same |

**Bitquery is the correct upgrade path if completeness turns out to matter**, as it subscribes to
migration events rather than polling for them. I have **not verified its current pricing, rate
limits or free-tier terms**, so that claim should be checked before any commitment. The same applies
to the published Pump.fun statistics cited for this work (832,941 launches, 0.20% graduating,
BOOST-era rates of 4.7-6.7%): those are third-party figures, not measurements of ours, and the
archive exists precisely to replace them.

**Solana RPC for holders** is the cheap first addition: `getTokenLargestAccounts` returns the top 20
holders in one call, which gives a top-10 concentration figure directly and needs no paid service.
It is deliberately out of V1 scope but is the first thing to add.

---

## 3. Schema

**No new tables.** The existing schema already keys on `network` and generalises cleanly; the
Solana adapter needs four additive columns, all nullable, so the Robinhood archive is untouched.

```sql
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS launchpad text;          -- curve pool's DEX id
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS measure_pool text;       -- pool the ladder reads
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS measure_pool_reason text;-- why that pool
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS cohort_sampled boolean NOT NULL DEFAULT true;
```

`measure_pool` is the finding in §1 made structural: the ladder reads the pool we chose, the choice
is recorded, and `migrated_destination_pool_address` stays in `graduation_pool` as metadata. If the
selection rule is ever revised, `measure_pool_reason` says which rows were gathered under which rule
instead of leaving a silent mixture.

`cohort_sampled` marks whether a graduate was drawn into the ladder cohort (§4), so the sampling
fraction is a recorded property of the data rather than a fact about the code at the time.

`launchpad_observations` gains one rung — **+5 minutes** — for the BOOST window. Rungs become
`5, 10, 30, 60, 180, 360, 720, 1440, 2880, 10080` minutes. The existing partial unique index on
`(network, token_address, rung_minutes)` already guarantees one row per rung.

### Censoring, made structural

A rung's denominator is a property of the query, not a caveat in prose:

```sql
-- "+24h: 381 of 612 graduated tokens have reached this observation window."
SELECT r.rung,
       count(*) FILTER (WHERE now() >= t.graduated_at + make_interval(mins => r.rung)) AS eligible,
       count(o.rung_minutes)                                                           AS observed,
       count(*)                                                                        AS cohort
FROM launchpad_tokens t
CROSS JOIN unnest(ARRAY[5,10,30,60,180,360,720,1440,2880,10080]) AS r(rung)
LEFT JOIN launchpad_observations o
  ON o.network = t.network AND o.token_address = t.token_address AND o.rung_minutes = r.rung
WHERE t.network = 'solana' AND t.graduated AND t.cohort_sampled
GROUP BY r.rung ORDER BY r.rung;
```

`eligible` is how many *could* have reached the rung; `observed` is how many we actually have.
A token that graduated six hours ago is absent from the +24h denominator entirely — it cannot be
counted as a failure, because the query never asks it the question. `observed < eligible` is a
collection defect and shows up as its own number.

---

## 4. Collection and scheduling architecture

Same shape as Robinhood, three differences.

**Cadence: every 2 minutes**, dispatched by the existing Vercel cron pressing `workflow_dispatch`
(GitHub drops most `schedule` fires, and a 5.8-minute feed cannot absorb that). The workflow's own
cron stays sparse as a genuine fallback. Discovery is 10 pages; the early-stop on meeting known
ground applies, so a healthy sweep usually costs far fewer than 10 calls.

**Graduate cohort sampling.** Observed PumpSwap pool creations run at roughly 0.9/minute (~1,250/day).
A full ladder on every graduate is ~12,500 pool calls/day on top of ~7,200 discovery calls — about
14 requests/minute against an unkeyed ceiling of ~30/minute, with no headroom for retries. V1
therefore draws a **fixed random fraction of graduates into the ladder cohort** (start at 25%,
recorded per row in `cohort_sampled`), which answers every question in §5 with confidence intervals
while leaving room to widen later or to lift the cap with an API key. The sampling draw must be made
**at graduation time and independently of any token property** — never "the most liquid ones" —
or the cohort stops being a random sample of graduates and every downstream statistic inherits the
bias.

**Measurement pool selection.** At graduation, list the token's pools, pick the deepest by
liquidity, store it in `measure_pool` with a reason, and read every rung from it.

Gap detection, the advisory lock, one-write-per-sweep, `ON CONFLICT DO NOTHING`, and the daily
health row all carry over unchanged from the Robinhood implementation.

---

## 5. Expected costs

**API.** Free tier, no key. Discovery ~5 calls/minute; ladder ~3/minute at 25% sampling; total well
under the ~30/minute ceiling with retry headroom. An API key would remove the need to sample.

**GitHub Actions.** 720 runs/day at ~1 minute each. The repository is public, so Actions minutes are
free; on a private repo this alone would exceed the monthly allowance and the design would have to
change.

**Neon compute.** A 2-minute write cadence holds the database awake continuously — modelled at
~180 CU-hours/month, about $19 at 0.25 CU, against a measured existing baseline of 43-77% awake, so
the true increment is roughly **$5-11/month**. Sampling cadence and write cadence remain separate
decisions; if this is unwelcome, the sweep can buffer and flush every 6 minutes at the cost of
detection lag.

**Neon storage.** ~1,250 graduates/day, ~25% sampled × 10 rungs ≈ 3,100 observation rows/day plus
curve rows — low hundreds of MB per year. Pennies.

---

## 6. Known limitations and data-quality risks

1. **The launch census is incomplete by construction.** At 25 pools/minute against a 5.8-minute feed,
   a 2-minute cadence has ~3 minutes of margin, and every missed run is recorded as a gap. Treat
   `launchpad_tokens` on Solana as a **sample of launches**, never as the population — the denominator
   for "what fraction of launches graduate" is not ours to compute from this data alone.
2. **BOOST contaminates the +5m and +10m rungs.** A mechanism that automatically buys after migration
   means early price and buyer counts are partly structural, not demand. Record those rungs, never
   interpret them as organic, and prefer *decay from* +5m rather than *level at* +5m when comparing
   tokens. If the mechanism's parameters change mid-collection, the comparison breaks across that
   date — so the collection date must be carried into any analysis, and a known change should be
   recorded in the spec when it happens.
3. **Holder data is partial and possibly missing-not-at-random.** One of three sampled tokens had
   holder counts. If availability correlates with a token being more established, then "holder
   concentration predicts failure" could be an artifact of which tokens have holder data at all.
   Report coverage alongside any holder finding, and prefer the RPC source once added.
4. **Survivorship in the destination-pool choice.** Picking the deepest pool is right for measuring
   where trading happens, but a token whose pools are *all* empty has no meaningful measurement pool.
   Record it with `measure_pool_reason = 'no liquid pool'` rather than dropping it — those are
   outcomes, not missing data, and dropping them would bias every survival rate upward.
5. **Ticker and name collisions are rampant**; identity is the mint address, never the symbol.
6. **`twitter_handle` is user-supplied** and frequently a status URL (two of three sampled).
   The existing `normalize_handle` already handles this; social-signal findings must also account
   for the fact that a *declared* account is not a *real* or *active* one.
7. **GeckoTerminal's schema is not contractual** and the `migrated_destination_pool_address`
   behaviour shows its semantics can differ per chain. Re-verify before trusting any field on a new
   chain, and prefer fields we can cross-check.

---

## 7. Minimal V1 implementation plan

Adapter-first, so Pump.fun specifics never leak into the shared code.

1. **Extract a chain adapter.** Move the Robinhood-specific constants out of `launchpad_archive.py`
   into an adapter object: network id, curve DEXes, graduate DEXes, sweep cadence, rungs, and a
   `choose_measure_pool(token_pools)` hook. The sweep, gap detection, persistence and reporting stay
   shared and chain-agnostic. This is a refactor with no behaviour change, verifiable by the existing
   18 tests.
2. **Add the Solana adapter**: `pump-fun`/`meteora-dbc`/`raydium-launchlab`/`boop-fun`/`moonshot` as
   curve DEXes, `pumpswap`/`meteora-damm-v2` as graduate DEXes, 2-minute cadence, the +5m rung, and
   `choose_measure_pool` = deepest pool by liquidity with the reason recorded.
3. **Additive migration** for the four columns and the +5m rung.
4. **Cohort sampling** at graduation time, recorded per row.
5. **Censoring-aware rung report** (`--cohort`), the query in §3, one row per rung with `eligible`,
   `observed` and `cohort`.
6. **A separate workflow** (`solana-archive.yml`) and dispatch target, so a Solana failure can never
   take the Robinhood archive down with it.
7. **Tests**: the destination-pool trap (a token whose `migrated_destination_pool_address` is an
   empty pool must be measured on the deep one), sampling independence, censoring denominators, and
   the existing gap/idempotence suite re-run for both chains.

No dashboard, no radar, no thresholds.

---

## 8. Measurements to collect before deciding on a faster radar

The decision should be made from data this archive produces, not from intuition:

1. **Detection lag** (`graduated_detected_at - graduated_at`), median and p95. At a 2-minute cadence
   the floor is ~60 seconds; if p95 is far above that, the collector, not the cadence, is the problem.
2. **Gap rate**: sweeps with `gap_seconds > 0` per day, and launches lost per gap.
3. **How fast the separation appears.** Compare +5m/+10m/+30m metrics of tokens that are still
   actively traded at +24h against those that are not. If the separation is already visible at +30m,
   a minute-resolution radar buys little; if it only appears at +5m, it buys a lot.
4. **Decay curves** for volume, distinct buyers and liquidity, per cohort — the shape says which rung
   is the earliest honest verdict.
5. **Base rates at every rung**, with denominators: still-traded, above graduation price, and both.
6. **Sampling sufficiency**: confidence interval width on the +24h survival rate at 25% sampling.
   If it is already tight, sampling harder is not needed; if not, an API key is cheaper than a radar.
7. **Whether the deepest-pool rule ever picks wrong** — how often the chosen pool is later overtaken
   by another pool for the same token.
8. **Holder coverage**, and whether it correlates with survival, which determines whether the RPC
   source is worth adding.

Only once 3 and 5 have answers does a faster collector have a case. Until then, more history beats
more resolution.

---

# OSINT layers — what is reachable, measured 2026-09-19

Framing this as early warning rather than selection changes what the archive must capture. Five
layers were proposed: creator, early buyers, social, narrative, momentum. Their feasibility differs
enormously, and two findings below decide the collection design.

## Layer 1 (creator) is free, and already one call away

`/tokens/{address}/info` — the call the archive already makes for every graduate — carries far more
than holders and socials:

`developer_address`, `developer_holding_percentage`, `is_honeypot`, `mint_authority`,
`freeze_authority`, `description`, `categories`, `telegram_handle`, `websites`, `discord_url`,
`gt_score`, `gt_score_details`.

Populated on all three Solana tokens sampled, with `developer_holding_percentage` varying
meaningfully (0.0, 19.99, 26.01). So the creator layer needs **no Solana RPC, no Bitquery and no new
credential**, and once `developer_address` is stored per token, creator history — prior launches,
graduation rate, survivor rate — is a **self-join on our own archive**, costing nothing per token
and getting better the longer we collect.

Caveat: `developer_address` is the address that deployed, which is exactly what a sophisticated
operator rotates. Treat a creator's record as evidence about *that address*, never about a person,
and expect the honest finding to be that most addresses are single-use.

## Layer 2 (early buyers) is free at wallet level — but only if captured live

`/networks/solana/pools/{pool}/trades` returns 300 trades, each with `tx_from_address`. A live call
returned 300 trades from **281 distinct wallets** — so buyer-level cohort analysis needs no paid
service either.

**The constraint that decides everything: those 300 trades covered 33 seconds.** On a token taking
1,964 trades in five minutes, the endpoint's window is under a minute, and `?page=2` returned an
overlapping, partly *newer* window rather than an older one — so paging backwards is not reliable.

**Consequence: early-buyer data cannot be reconstructed retrospectively for the tokens that matter.**
A hot graduate's first minutes are gone from the endpoint within about a minute. Any experiment that
proposes to "take every graduated token for the last 30 days and reconstruct the first 30 minutes"
is impossible for precisely the successful tokens it most needs — the data no longer exists to be
fetched. This layer must be collected **forward**, within seconds of graduation, or not at all.

That is also the first genuine argument for a fast lane, and it is narrow: not a minute-resolution
radar over the whole firehose, but a burst of trade captures on a token **at the moment it
graduates**. Cost is bounded by graduations (~0.9/minute), not by launches.

## Layer 3 (social) splits into a free half and an expensive half

**Free, already collected:** the declared handles, Telegram, website and description from `/info`.
Note that declared handles are frequently not handles — two of three sampled tokens had a status URL
in `twitter_handle` — so the existing `normalize_handle` matters here too, and *whether a creator
declared a clean, real handle* is itself a candidate feature.

**Expensive: propagation and sequence.** Reconstructing who mentioned a token first, and in what
order, needs X search per token. At ~1,250 graduates/day, 30 days is ~37,500 tokens; at roughly 100
posts each and the pay-per-use rates recorded in CLAUDE.md (~$0.005/post read), that is **~$18,750**
— not a budget question, a non-starter. A random sample of ~400 graduates costs ~$200 at the same
rates and answers the same question with confidence intervals. Account-age and follower lookups for
declared handles are ~$0.01 each, so ~$4 for that sample: the cheapest genuinely new social signal
available.

**The feature that actually matters is not available.** "Has a Telegram" is weak — it is confounded
by creator effort, and manipulators fill socials in too. "Has a Telegram that existed before the
token" is the real signal, and neither GeckoTerminal nor a cheap X call provides community age.
Telegram history is rated medium/low feasibility in this repo's own source guidance, and nothing
here changes that.

## Layer 4 (narrative) is the only one needing an LLM, and both provider accounts are unfunded

Classifying what a token references, and whether the narrative predates it, is an LLM task over the
`description`, name and social text we already collect. It is cheap per token but blocked today —
see the credit note at the top of CLAUDE.md. It is also the layer most exposed to circularity: a
model asked "is this a copycat?" after the fact will happily rationalise either answer, so the
classification must be made from launch-time text only, stored immutably with its model version.

## Layer 5 (momentum) is what the archive already does

Nothing new needed beyond the ladder.

## The methodological trap worth stating plainly

The two published findings cited for this work answer **different questions**:

- "Tokens advertising socials graduated 1.9% vs 0.11%" predicts **graduation**, and needs the
  non-graduated population as its denominator.
- "What distinguishes durable graduates" predicts **survival among graduates**, and conditions on
  graduation.

The proposed first experiment conditions on graduation, so it can only answer the second. That is
fine — it is the archive's own question — but it cannot confirm or refute the social-links result,
and combining the two would produce a collider-style artifact. Keep them as separate studies with
separate cohorts, and remember that on Solana our launch census is itself a sample (§6), so the
graduation-rate denominator is not ours to compute at all.

The same caution applies to early-buyer cohorts: repeated co-occurrence of wallets across launches
may reflect those wallets selecting into launches that were already attracting attention, rather
than causing anything. Cohort membership is a **description of the flow**, not an explanation of it.

## Why the manipulation layer is the strongest place to start

It is the one layer that can be validated **without predicting anything**. Wash-trading signatures,
buyer cohorts that co-occur across unrelated tokens, developer holding concentration and
instant-shill patterns are all measurable against ground truth that does not depend on future price:
either the same wallets keep appearing together or they do not. Everything in the opportunity
direction requires waiting for outcomes; the manipulation direction is checkable the week it is
built. It is also the half that stays useful even if no predictive signal is ever found.

## Revised first experiment

Not "30 days backwards" — that data is gone. Instead, **forward from now**:

1. Extend the Solana adapter's enrichment to store the fields already available for free:
   `developer_address`, `developer_holding_percentage`, `is_honeypot`, `mint_authority`,
   `freeze_authority`, `telegram_handle`, `websites`, `description`, `gt_score`.
2. Capture **trades at graduation**: on detecting a graduation, pull the measurement pool's trades
   immediately and again shortly after, storing wallet-level rows. Bounded by graduations, not
   launches.
3. Let it run until there are enough graduates with complete +24h and +7d outcomes.
4. Then run the comparison: creator history (self-join), early-buyer diversity and co-occurrence,
   declared-social quality — against survival. Sample ~400 for the paid X propagation arm only if
   the free features look like they carry signal.

Posture: public on-chain records and public posts only, research context, never advice. Wallet
analysis stays at the level of addresses and behaviour; nothing here attempts to identify the people
behind them, and a creator's record is a statement about an address, not a person.
