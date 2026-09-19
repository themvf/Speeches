# Graduation Archive — Robinhood Chain launchpad recorder (spec, 2026-09-19)

## Purpose

Build a complete historical record of Robinhood Chain launchpad activity so that we can determine,
**empirically**, what distinguishes a successful graduate from the thousands of tokens that vanish.

This is deliberately **not a radar**. It surfaces nothing, alerts on nothing, and sets no thresholds.
Roughly one launch in sixty graduates, and picking a "watch it above 70%" rule before we have
labelled outcomes would produce a confident list that is mostly wrong in a way nobody would notice.
The archive is what makes the threshold question answerable in two to four weeks. The radar is a
later change that reads this table.

Association only, never attribution or advice — same posture as the rest of the crypto workspace.

## What was verified live (2026-09-19)

Re-verify before assuming any of this still holds; GeckoTerminal's schema is not contractual.

| Fact | Measured |
|---|---|
| New pool rate, Robinhood Chain | **8/minute**, then **13.4/minute** an hour later — it varies |
| `new_pools` pagination depth | **10 pages × 20 = 200 pools**, page 11 returns 401 |
| History covered by a full sweep | **11-14 minutes**, measured; falls as the chain gets busier |
| Share of new pools that are Pons curve pools | 57 of 80 (71%) |
| Curve pools vs graduated pools | 57 vs 1 → **1.8%** (published figure elsewhere: 1.55%) |
| Rate limit (no API key) | ~30 requests/minute; exceeded easily with per-token calls |
| `tokens/multi` batch size | **15+ tokens in one call** (documented max 30) |

**The bonding curve and the graduated pool are different DEXes.** `pons-v2` is the curve;
`pons-v2-dex` is where graduates land. ASKR's `migrated_destination_pool_address`
(`0x90b75d6e…`) is on `pons-v2-dex`, and its `pool_created_at` equals the token's
`launchpad_details.completed_at` to the second (2026-09-18T18:36:55Z).

**Consequence: graduation detection is free.** A graduation announces itself as a new pool on a
graduate DEX inside the feed we already sweep, carrying the exact graduation timestamp. No
per-token polling is required to notice one. `launchpad_details.completed_at` remains the
authoritative confirmation, read on the token record we fetch anyway.

Other launchpad DEXes present on the chain: `hoodit`, `o1-launchpad-robinhood`,
`bankr-robinhood`, `clanker-robinhood`, `virtuals-robinhood`, `easya-kickstart-robinhood`,
`mint-club-robinhood`. V1 records all of them; only Pons's curve/graduate DEX pairing has been
confirmed, so the others are recorded by DEX id without assuming their lifecycle.

## Endpoints and what each one costs

| Endpoint | Batched? | Gives us | When we call it |
|---|---|---|---|
| `networks/robinhood/new_pools?page=1..10` | 20/call | pool address, name, DEX, `pool_created_at`, liquidity, volume, `transactions` (buys/sells/**buyers/sellers** at m5/m15/m30/h1/h6/h24), `price_change_percentage` | 10 calls, every sweep |
| `networks/robinhood/tokens/multi/{addrs}` | **up to 30/call** | `launchpad_details` (`graduation_percentage`, `completed`, `completed_at`, `migrated_destination_pool_address`), price, FDV, volume | every sweep, for new + live candidates |
| `networks/robinhood/pools/{addr}` | 1/call | the full pool object above, current | post-graduation snapshot ladder |
| `networks/robinhood/tokens/{addr}/info` | **1/call** | `holders.count`, `holders.distribution_percentage`, **`twitter_handle`**, telegram, websites, `gt_score` | **graduated tokens only** |

`tokens/multi` does **not** carry holders. Holder count and concentration need the per-token
`/info` call, which is why it is reserved for graduates (~1.8% of launches, ~200/day, ~8/hour) and
never run against the firehose.

`/info` is also where the project's **X handle** comes from. That is the bridge to the social side:
a graduate arrives with its own declared X account, without a search.

## Budget

Per sweep (every 5 minutes; see the build note at the end): 10 discovery calls + ~2-4 `tokens/multi` calls + a handful of
snapshot/info calls for graduates in flight. Comfortably inside ~30/minute even allowing retries.
Free public data; **no X credits are involved** — this is entirely separate from the collection
ledger that meters post ingestion.

## Schema

Python-owned (`neon_feeds`/a new module), consistent with `crypto_market_*`. Two tables: one row
per token (mutable state) and one immutable row per observation.

```sql
-- One row per launchpad token, updated in place as its state advances.
CREATE TABLE IF NOT EXISTS launchpad_tokens (
  network            text        NOT NULL,
  token_address      text        NOT NULL,
  symbol             text,
  name               text,
  dex                text        NOT NULL,          -- launchpad that created the curve pool
  curve_pool         text,                          -- bonding-curve pool address
  first_pool_created timestamptz,                   -- pool_created_at of the curve pool = launch
  first_seen_at      timestamptz NOT NULL,          -- when WE first observed it
  first_seen_pct     double precision,              -- graduation % at first observation
  graduated          boolean     NOT NULL DEFAULT false,
  graduated_at       timestamptz,                   -- launchpad_details.completed_at
  graduated_detected_at timestamptz,                -- when WE saw it (lag is a data-quality measure)
  graduation_pool    text,                          -- migrated_destination_pool_address
  last_pct           double precision,
  last_seen_at       timestamptz,
  state              text        NOT NULL DEFAULT 'live',  -- live | graduated | stalled | dead
  holders            integer,                       -- graduates only, from /info
  top10_share        double precision,              -- graduates only
  twitter_handle     text,                          -- graduates only
  PRIMARY KEY (network, token_address)
);

-- Immutable. One row per (token, observation). Never rewritten, same rule as crypto_price_events.
CREATE TABLE IF NOT EXISTS launchpad_observations (
  network        text        NOT NULL,
  token_address  text        NOT NULL,
  observed_at    timestamptz NOT NULL,
  phase          text        NOT NULL,   -- 'curve' | 'post'
  graduation_pct double precision,       -- curve phase
  price_usd      double precision,
  fdv_usd        double precision,
  liquidity_usd  double precision,
  volume_m30     double precision,
  volume_h1      double precision,
  volume_h24     double precision,
  buyers_m30     integer,
  sellers_m30    integer,
  buyers_h1      integer,
  sellers_h1     integer,
  txns_h1        integer,
  price_change_h1 double precision,
  holders        integer,                -- post phase, when fetched
  PRIMARY KEY (network, token_address, observed_at)
);

-- Sweep-level record, so gaps are visible rather than inferred. See "Gap detection".
CREATE TABLE IF NOT EXISTS launchpad_sweeps (
  id              bigserial PRIMARY KEY,
  started_at      timestamptz NOT NULL,
  finished_at     timestamptz,
  pages_fetched   integer     NOT NULL DEFAULT 0,
  pools_seen      integer     NOT NULL DEFAULT 0,
  oldest_pool_at  timestamptz,           -- pool_created_at of the oldest pool on the last page
  newest_pool_at  timestamptz,
  new_tokens      integer     NOT NULL DEFAULT 0,
  graduations     integer     NOT NULL DEFAULT 0,
  gap_seconds     integer,               -- see below; NULL until a prior sweep exists
  complete        boolean     NOT NULL DEFAULT true,
  note            text
);
```

Indexes: `launchpad_tokens (state, last_seen_at)` for candidate selection,
`launchpad_tokens (graduated_at)` for cohort queries, `launchpad_observations (token_address, observed_at)`.

## The sweep, every 10 minutes

1. Fetch `new_pools` pages 1-10. Stop early once a page's oldest `pool_created_at` predates the
   previous sweep's `newest_pool_at` — the rest is already recorded.
2. Deduplicate by pool address; split into curve pools (launchpad DEXes) and graduate pools.
3. **Graduation by arrival**: a pool on a graduate DEX (`pons-v2-dex`, and any other pairing we
   later confirm) whose base token we already hold marks that token graduated, with
   `graduated_at = pool_created_at`.
4. Batch the new tokens plus all `state = 'live'` candidates through `tokens/multi` (30 at a time)
   for `graduation_percentage` / `completed`. Confirm or correct step 3 from `completed_at`.
5. For tokens that graduated in this sweep, one `/info` call each: holders, top-10 share, X handle.
6. Post-graduation ladder: for each graduate, if the elapsed time since `graduated_at` has crossed
   a rung it has no observation for, fetch its pool and write one. Rungs:
   **+10m, +30m, +1h, +3h, +6h, +12h, +24h, +48h, +7d**. A missed rung is recorded late with its
   true `observed_at`, never backdated.
7. Adaptive sampling on the curve side: a token at 0% keeps its discovery row and nothing more;
   one that has moved is observed every sweep; one with no movement for 24h goes `dead` and is
   dropped from the candidate set. (Published data says a token still on the curve after 24 hours
   almost never graduates — worth re-testing against our own data once we have it.)
8. **One write.** Everything gathered in the sweep — token upserts, observations, the sweep row —
   goes to Neon in a single connection with multi-row `execute_values` inserts.

## Why the write pattern matters (Neon cost)

Existing scheduled writers already wake this database at roughly **:00, :05, :17, :20, :30** every
hour, plus :35/:37 every two hours and a 3-hourly cluster. With a 5-minute scale-to-zero timeout
that is ~43% awake in a quiet hour and ~77% with everything firing (web traffic raises it further).

A 10-minute sweep that writes **once** lands inside windows the database is largely awake for
already; the modelled increment is a few dollars a month at 0.25 CU, and at a 10-minute cadence the
awake fraction sits *below* the all-jobs baseline. A recorder that wrote every 60 seconds would
pin the database at 100% awake for no analytical gain.

The rule to keep: **sampling cadence and write cadence are separate decisions.** If a fast lane is
added later, it samples in memory and still flushes on the slow clock.

Storage is not the concern — ~11,500 discovery rows/day plus observations only for movers is
hundreds of MB per year, pennies at Neon's storage rate. Retention: keep full resolution around
every graduation permanently; thin `dead` tokens' observations after 90 days, keeping the token row.

## Gap detection (required, not optional)

The feed reaches back ~19 minutes. A sweep that is late, fails, or gets rate-limited **loses
launches permanently**, and the failure is invisible — the feed simply moves on.

Every sweep therefore records `oldest_pool_at` from its deepest page and compares it with the
previous sweep's `newest_pool_at`. If the previous newest is **older** than this sweep's oldest,
launches happened in between that we never saw: set `complete = false`, store
`gap_seconds`, and surface it. Any analysis run over a window containing an incomplete sweep must
say so rather than presenting a partial population as the population — this is the same posture as
`corpus_source` and the `warnings` arrays elsewhere in the app.

Watch the launch rate too: at ~8 pools/minute the 10-page window is ~19 minutes, but if the chain
gets busier that window shrinks. If `oldest_pool_at` is consistently less than ~12 minutes back,
the sweep interval needs shortening — the recorder should flag that rather than silently degrade.

Also record `graduated_detected_at` alongside `graduated_at`: the difference is our observation
lag, and it is the honest measure of what the 10-minute cadence costs us.

## What this makes answerable

After two to four weeks, with outcomes labelled:

- Of tokens that graduate, how many still gain holders after an hour? After a day?
- Does buying in the first 30 minutes after graduation predict 24-hour survival?
- Does holder concentration at graduation separate durable graduates from pump-and-dumps?
- Where do ASKR, ZCAT and the rest sit against the whole population — and is "+10,445% since
  graduation" typical of survivors or an outlier? (Published figures elsewhere put the median
  graduate 85% below its first-hour price, with fewer than one in ten above it. Our own number
  should replace that citation.)
- Is pre-graduation velocity predictive at all? If the 10-minute resolution shows a signal, a
  60-second fast lane is justified by evidence. If it does not, we saved the worker.

And the one that connects to the rest of this workspace:

- Were the accounts we already track posting about these tokens **before** they graduated?

## The social bridge, and its honest limit

Each graduate arrives with a declared `twitter_handle` from `/info`, and its contract address is
the exact string to search saved posts for.

The limit to state plainly: the rolling collector searches **per tracked coin**, so for the ~600
accounts in People we only ever saved posts that already mentioned a registry coin. An account that
called forty Pons coins at 14% would show zero pre-graduation calls, and that would be an artifact
of collection, not a finding. Only the ten **watcher** accounts have full timelines, so an
"early voice" study is honest over those ten and misleading over the rest — until timeline
collection widens.

## Explicitly out of scope for V1

- The radar: any surfacing, ranking, threshold or alert. Thresholds come from the data.
- The 60-second fast lane and the persistent worker it needs.
- Any UI. When it arrives it belongs inside `/market/crypto` as a tab, pane or command token —
  not a new route.
- Chains other than Robinhood, and any judgement about which launchpads are "good".

## Open questions

- The 4.2 ETH Pons threshold, the 1.55% graduation rate and the 4-minute median graduation time
  come from published third-party analysis and are **not independently verified here**, beyond our
  own 1.8% curve-to-graduate ratio on a sample of 80 pools. The archive replaces all three with
  measurements.
- Curve/graduate DEX pairing is confirmed for Pons only.
- Whether `new_pools` lists every launchpad's curve pools, or only those whose curve is an indexed
  pool. A launchpad whose curve is not pool-shaped would be invisible to this sweep.
- Scheduling mechanism: GitHub `schedule` fires are dropped often (see the Vercel-trigger section
  of docs/crypto-social-tracking.md), so the 10-minute cadence likely needs the existing
  `/api/cron/dispatch-workflows` pattern rather than a bare cron.


## Built 2026-09-19 — what the implementation changed

`launchpad_archive.py`, `sql/launchpad_archive.sql`, `.github/workflows/launchpad-archive.yml`,
`tests/test_launchpad_archive.py`. Four things the build learned that the spec above had wrong or
did not cover.

**The cadence is five minutes, not ten.** A live sweep fetched 149 unique pools spanning 11.1
minutes — 13.4 pools/minute, well above the 8/minute measured an hour earlier. A ten-minute sweep
would have run on about one minute of margin, so any delayed run would lose launches. The window is
a function of how busy the chain is, so it is re-measured every sweep rather than trusted.

**Feed depth can only be read from a sweep that went full depth.** A healthy sweep stops early once
it meets pools it already recorded, which makes its reach short precisely when everything is
working — measuring that raises an alarm exactly when nothing is wrong. The continuity report
therefore takes its depth figure only from sweeps with `pages_fetched >= 10`, and reports depth as
unknown (not healthy) when no sweep in the window went full depth.

**`twitter_handle` is whatever the launcher typed.** A live sweep returned
`Na1_N1ako/status/2101346622135230543` in that field. Handles are normalised (URL prefixes and
path segments stripped, `@` removed, 15-character and character-set check) and anything that is
not a handle is stored as NULL rather than as a value the social join would silently miss on.

**Observations carry `rung_minutes`.** The ladder rung is stored rather than inferred from
timestamps, with a partial unique index on `(network, token_address, rung_minutes)`, so a rung is
filled exactly once even if a sweep retries.

Live verification, three sweeps against the real API into a local Postgres: 149/80/20 pools,
`gap_seconds` 0 on both sweeps that had a predecessor, 4 graduations detected from graduate-DEX
arrivals, holders and X handles enriched, zero errors. First real measurements from our own data:
**2.2% graduation rate** (4 of 181) and a **median detection lag of 86 seconds** (worst 764) — that
lag is the number that will eventually decide whether the 60-second fast lane is worth building.
