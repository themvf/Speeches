# Graduation Archive — program overview

Read this first if you are picking up any work on the launchpad archive. It explains what the
program is for and the rules it is built on. The implementation details live in
[`docs/graduation-archive-spec.md`](graduation-archive-spec.md) (Robinhood Chain) and
[`docs/solana-pumpfun-archive-spec.md`](solana-pumpfun-archive-spec.md) (Solana, the chain adapter,
the enrichment worker, and the live-smoke gate every new chain must pass).

## What we are building

A **cross-chain graduation archive**: a historical record of newly launched tokens as they move from
launchpad/bonding-curve trading into ordinary liquidity pools, and of what happens to them
afterwards.

The first two ecosystems are **Robinhood Chain** and **Solana**, with Solana initially focused on
Pump.fun and the related launchpad activity that shares its data shape.

## The question it exists to answer

**What separates tokens that maintain momentum after graduation from those that quickly collapse?**

Graduation itself is not success. It is the starting line. So the archive measures performance at
intervals — **+5 minutes, +10, +30, +1 hour, +3, +6, +12, +24 hours, +48 hours, +7 days** — rather
than treating the migration event as an outcome.

The expected product is **not a "buy score"**. It is a research dataset capable of identifying
patterns associated with post-graduation survival, sustained attention, creator track record and
manipulation risk. Whether a real-time early-warning capability is ever justified, and which signals
should drive it, is a question the accumulated history is meant to answer — not an assumption we
start from.

## What we capture

- **Graduation events** — when a token migrates, detected from the arrival of its destination pool,
  cross-checked against the launchpad's own completion record.
- **Creator information** — developer address, holding percentage, mint and freeze authority. Once
  the address is stored, creator track record (does this wallet repeatedly launch tokens that
  graduate, or that survive longer than average?) is a self-join on our own archive.
- **Opening trading activity** — wallet-level trades immediately after graduation, on Solana.
- **Liquidity, volume, buyers and sellers** at each interval, with **distinct buyers** rather than
  trade counts, because "is it still attracting new people" is the actual question.
- **Social links** declared at launch — X, Telegram, website, description.
- **Post-graduation performance** at every rung above.

### Why the opening wallet cohort is captured immediately

On Solana the trades endpoint returns a recent window that, on a busy graduate, spanned **33
seconds**. Paging returns an overlapping rather than an older window. So the opening cohort **cannot
be reconstructed later at any price** — it is captured within seconds of graduation or it is gone.
This is the single hardest constraint in the design, and most of the scheduling architecture exists
to protect it.

## How organic activity is separated from coordinated activity

The archive stores raw wallet-level rows, not just summaries, so the question can be asked later in
ways we have not thought of yet. Alongside them it records distinct wallets, buyers, sellers,
repeat-wallet counts and top-wallet share. That supports looking for **recurring early-wallet
groups across unrelated launches** and **concentrated trading behaviour** — activity that looks
organic in aggregate and looks very different once you ask who was actually trading.

This half of the work has a property the rest does not: it can be **validated without waiting for
outcomes**. Whether the same wallets keep appearing together is checkable immediately, whereas any
claim about "success" needs weeks of history first.

## Rules the program is built on

**No arbitrary thresholds.** We do not encode "+50% is good" or "100 buyers is strong". Roughly one
launch in sixty graduates, and a threshold chosen before outcomes are labelled produces a confident
list that is mostly wrong in a way nobody notices. Collect first; derive thresholds from the data.

**"Not observed" must never read as "observed and absent".** Every gap in the record is recorded as
its own state with its own reason, and every rung reports its own denominator, so a token that
graduated six hours ago is absent from the +24h denominator rather than counted as a failure.

**Data-quality controls are part of the product, not instrumentation on top of it.** The system
detects and records missed collection windows, delayed measurements, incomplete trade captures and
late pool selection, because the failure mode that matters in an archive is not a crash — it is
silently producing a plausible, misleading dataset that nobody questions for months.

**Health is separated by failure mode.** Losing perishable capture data while every durable metric
reads perfect is the specific failure we most need to see, so capture health, enrichment health and
measurement-selection health are tracked apart and a small sample reports `unknown` rather than
being forced to a verdict.

**Sampling is a recorded property of the data.** Solana volume is far larger than Robinhood Chain,
so graduates are drawn into the measurement cohort by a hash of the mint address — stable across
re-runs and incapable of correlating with liquidity, buyers or anything later treated as an outcome
— and whether a row was drawn is stored on the row.

## How it is kept affordable and timely

Solana runs at roughly **3.4 graduations per minute** (~4,900/day across all launchpads, of which
Pump.fun lineage is about 30%), against a feed that reaches back only about six minutes. The design
therefore splits work by whether it expires:

- A **short-cadence sweep** protects only what cannot be recovered: discovery, graduation detection,
  measurement-pool selection, and the opening trade capture.
- A **separate enrichment worker** owns everything durable — creator and social metadata, holders,
  and the post-graduation ladder — on its own schedule, with its own failure isolation. Being slow
  there costs latency; being slow in the sweep costs data.

Live testing shows the architecture keeps up: enrichment sustains roughly **twice the observed
arrival rate**, and the sweep completes comfortably inside its cadence.

## Three things every analysis must respect

**Solana `first_pool_created` is not a launch time.** The curve pool is frequently indexed at or
near migration — measured median gap to graduation of 0s for Pump.fun — so any "time to graduation"
or "minutes since launch" statistic computed from it will be confidently wrong. This is a
**graduation and post-graduation archive**, not a launch-time archive. The caveat is also a
`COMMENT ON COLUMN`, because the columns make the invalid calculation possible.

**Scope every read to one chain.** `launchpad_sweeps` is shared by both archives, and until
2026-09-20 it had no `network` column at all. An unscoped read of it blends them, which was true of
the gap watermark inside the collector itself and of the continuity verdict — the latter read
healthy straight through an outage, because the other chain's sweeps filled the expected count.
Sweep rows from before that date are unattributable and are reported separately rather than folded
into either chain.

**Filter on `launchpad_family`, never on `launchpad`.** A fast graduator is often only ever observed
arriving at its destination, so the curve-DEX column is NULL for it; filtering on it returned 9 of
16 real Pump.fun graduates on the commissioning sample. Family spans both sides of a launchpad's
pairing and is the column a launchpad-specific analysis should use.

## Posture

Public on-chain records and public posts only. Research context, never investment advice.
Wallet analysis stays at the level of addresses and behaviour; a creator's record is a statement
about an address, not about a person.
