# Telegram OSINT collector — implementation contract

Built 2026-09-19. `telegram_collector.py`, `telegram_extract.py`, `telegram_mentions.py`,
`telegram_analytics.py`, `sql/telegram_osint.sql`, `telegram_channels.json`,
`.github/workflows/telegram-collect.yml`, `.github/workflows/telegram-mentions.yml`,
`tests/test_telegram_{extract,mentions,collector,analytics}.py`.

The objective is not an archive of Telegram messages. It is to measure **which channels surface
tokens early, how a token propagates between them, and what actually happened afterwards** — joined
to the Solana graduation archive (`docs/solana-pumpfun-archive-spec.md`), which is the only place
this repository holds outcomes.

## 1. Library: Kurigram, with Telethon as the alternative

Verified live against PyPI on 2026-09-19: **`kurigram` 2.2.26**, source
`github.com/KurimuzonAkuma/pyrogram`. **Telethon** is 1.45.0. Both are MTProto clients; neither is
the Bot API.

Kurigram is the pick. It is the maintained Pyrogram fork, which means current layer support against
a server API that changes without asking, and a Pyrogram-shaped API that anyone who has used the
original already knows. Nothing new is started on the original `pyrogram` package: its repository
was archived in 2024, and an archived MTProto client is a client that will eventually stop being
able to talk to Telegram at all.

**The gotcha worth writing down: Kurigram installs as `kurigram` and imports as `pyrogram`.** It is
a fork that kept the module name, so `from pyrogram import Client` is the correct import *and* would
silently pick up the archived original if that were also installed. The workflow installs exactly
one of them.

The library choice is made reversible rather than trusted: `normalize_pyrogram` and
`normalize_telethon` produce the same row, and that row shape is the only thing any later stage
depends on. `TELEGRAM_LIBRARY=telethon` switches backends without touching the schema or the
analytics. Node was not considered for this collector — the archive, the price layer and every join
here are Python/Postgres — but for the record, GramJS is archived in favour of teleproto, so a Node
implementation would start on the successor, not on GramJS.

## 2. Authentication: a dedicated user account, not a bot

`messages.getHistory` is a **user-account method**. A Bot API bot cannot walk arbitrary channel
history even where it can see new posts, which makes backfill — the thing that gives this dataset
any depth at all on day one — impossible. So:

- A dedicated Telegram user account, used for nothing else and **never a personal everyday
  account**: the session string this collector holds is full read access to whatever that account
  can see, so the blast radius of a leaked runner secret should be a throwaway account's public
  channel memberships and nothing more. It joins only public channels.
- `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` from my.telegram.org, and `TELEGRAM_SESSION`, a session
  string generated once **interactively on a workstation** (the login sends a code to the account;
  an unattended runner cannot answer it).
- The session string is **equivalent to the account**: full read access to everything it can see.
  It lives in GitHub Actions secrets only, and the client runs `in_memory=True` so no session file
  is ever written to the runner's disk.
- Peer ids are never guessed. `telegram_channels.json` may carry a username with a null
  `channel_id`; `--resolve` asks Telegram and records the real id. A wrong id does not fail, it
  collects the wrong channel.
- Rate limits are Telegram's to set: a FloodWait is recorded as its own access state with its
  duration, not folded into a generic error, because it means "you are fine, wait" while a
  `forbidden` means "a human needs to look at this".

## 3. Schema

`sql/telegram_osint.sql`, Python-owned, applied on every run. Three layers kept apart:

| layer | tables | property |
|---|---|---|
| raw | `telegram_channels`, `telegram_messages`, `telegram_collection_runs` | what Telegram said; never rewritten by a classifier |
| derived | `telegram_token_mentions`, `telegram_price_points`, `telegram_mention_outcomes`, `telegram_mention_summary` | recomputable from raw + the archive |
| rollup | `telegram_channel_stats`, `telegram_channel_pairs` | every rate stores its own denominator |

Collection is **idempotent by construction**: `(channel_id, message_id)` is Telegram's own identity,
so re-reading a window writes the same rows. Duplicate ids are therefore impossible, which is why
the health report measures the opposite signal — gaps in a channel's id sequence, which are
deletions or windows we missed.

## 4. Collection

`telegram_collector.py --execute` polls every active channel for messages newer than its cursor;
`--backfill` walks down from `backfill_floor_id` a page at a time until Telegram stops returning
older history, and records that it stopped. Cursors only move forward (`GREATEST`/`LEAST` in SQL),
so a partial or out-of-order read can never rewind and skip the window in between. Errors are
per channel: the window a poll covers is not recoverable once the feed moves on, so one dead channel
must not cost the run.

Every run writes a `telegram_collection_runs` row — the same reasoning as `launchpad_sweeps`: a run
that never happened leaves no trace anywhere else.

## 5. Extraction and resolution

`telegram_extract.py` is pure. A Solana mint is identified by **decoding base58 to 32 bytes**, not by
matching a shape, and not by requiring a `pump` suffix — that is a Pump.fun convention, and the
archive holds Meteora, Raydium, Boop and Moonshot graduates too. Addresses inside Dexscreener and
Birdeye links are collected, because a link is one of the commonest ways a channel posts a contract.

Resolution (`telegram_mentions.py`):

- **contract** → resolved, confidence 1.0. Whether the archive holds the token is a separate
  question; conflating them makes "we never archived this" read as "the channel posted junk".
- **ticker** → resolved only when the archive holds **exactly one** token with that symbol within
  ±72h of the message (`ticker_unique`, confidence 0.7). Otherwise `unresolved_ambiguous` or
  `unresolved_not_in_archive`, recorded as such. A wrong resolution is worse than no resolution: it
  puts another channel's token into this channel's denominator.
- A message carrying a contract has its cashtags **dropped**. The contract is the subject.

### Relay attribution — original vs forward vs repost

The largest semantic risk in this layer, and the one that decides whether it measures discovery or
just stores text. A Telegram **forward** announces itself in the message header. A **copy-paste
repost** does not, and in call channels the repost is the commoner shape — so without a rule for it,
an unattributed copy becomes a second independent sighting: the token looks discovered twice, and
the relay channel's "first among monitored" count is inflated by exactly the tokens it was slowest
on.

`mention_origin` is therefore three-valued, set by `attribute_relays()` before the sequence is
computed, per token, in time order (a repost can only ever be attributed to a post that came
*before* it):

| value | rule | counts as discovery |
|---|---|---|
| `forward` | Telegram forward header | no |
| `repost` | ≥0.60 Jaccard over word 5-grams with an earlier post about the same token from another channel, inside 24h; below 8 words only an exact copy counts | no, and `relay_of_channel_id` names what it relayed |
| `original` | everything else | yes |

The word floor matters: these channels legitimately post a bare contract address, and without it
every terse channel would be labelled a relay of every other terse channel. Only `original` mentions
take a sequence position, so `is_first_monitored_mention` and every timing statistic derived from it
are counts of discovery, not of posting.

## 6. Archive join, and why there is a second price layer

Mentions copy the archive's view at derivation time: `graduated_at`, `launchpad_family`,
`measure_pool`, `measure_pool_timing`, and `seconds_to_graduation` = `graduated_at − mentioned_at`
(**positive means the channel posted before graduation**).

Timing is measured against **graduation, never against `first_pool_created`**. On Solana that column
is indexing/migration-adjacent — median gap to graduation 0s for Pump.fun, 52s for Meteora DBC — so
any "minutes since launch" computed from it is confidently wrong. The caveat lives as a
`COMMENT ON COLUMN` on the archive table itself.

Analysis filters on **`launchpad_family`, never `launchpad`**: the latter is only ever a curve DEX
and was NULL for the fastest graduators, undercounting Pump.fun by 44% on the commissioning sample.

The archive's own ladder is anchored to **graduation** and samples **a quarter** of Solana
graduates, so it cannot answer "what happened after this channel posted". Hence a separate,
mention-anchored price layer: GeckoTerminal OHLCV is **historical and can be fetched after the
fact**, unlike the opening trade window, which expires. Minute candles carry the +10m/+30m/+1h/+3h
rungs; hourly candles carry +24h and the peak.

## 7. The methodological rules, and where each is enforced in code

| rule | enforcement |
|---|---|
| no cherry-picked winners; every resolved call stays in the denominator | `channel_stats` counts all resolved mentions; nothing filters on outcome |
| not observed ≠ not yet eligible ≠ observed and failed | `telegram_mention_outcomes.status` ∈ `observed`/`pending`/`unobserved`, with `unobserved_reason`; a pending rung carries no return at all, so it cannot be averaged into one |
| numerator and denominator for every statistic | `rungs` JSONB carries all three counts per rung; `first_among_monitored`/`graduated_tokens` are stored as counts, not shares |
| forwards are not independent discovery | `is_forward` excludes a mention from the sequence entirely; pairs count `forward_pairs` separately |
| claimed multiples are never results | `claimed_multiple` is carried beside the measured return, never substituted into it — `mention_outcome` cannot see message text |
| no subjective "good channel" score | `typology()` returns a measured label and its reason, or `None` below 10 graduated mentions. There is no weighted quality number, for the same reason the archive sets no thresholds in V1 |

Typologies, in fixed precedence (coordination first, because a channel posting in lockstep is not
independently early however early it is): `coordinated_cluster` → `smart_wallet_relay` → `relay` →
`originator` → `early_amplifier` → `late_promoter` → `momentum_follower` → `unclassified`.
`unclassified` is a real answer and better than stretching a rule to cover a channel.

## 8. Propagation graph and the questions it answers

`telegram_analytics.py --token <mint>` returns the token's path: each hop's channel, time, whether
it was a forward and from where, its position among monitored channels, its lag behind the first,
market cap and price at that moment, and every rung's outcome. `--pairs` returns co-mention edges
(shared tokens, how many landed inside 120s, median gap, who led, how many involved a forward),
which is the coordination evidence — counted, not pronounced upon.

"First among monitored channels" is exactly that. The monitored set is a convenience sample of
Telegram and says nothing about who was first in the world; the column comment says so.

## 9. Health

`--health` reports collection and derivation separately, because a perfectly collected channel whose
tokens the archive never saw produces zero measurable calls — a coverage fact about the archive, not
a collection failure. It carries: run cadence and longest silence, per-channel silence and access
state, channels losing access, id-sequence gaps, edited/deleted counts, the resolution mix, archive
coverage, and the outcome status split.

## 10. Live-smoke gate — REQUIRED BEFORE MERGE, NOT YET RUN

Nothing in this build has run against real Telegram data: this session has no API credentials, no
session string and no database. **The tests prove the rules, not the pipeline.** The Solana adapter
learned this the expensive way — a mocked suite passed green against an address normaliser that
404'd every live call, because a mock answers whatever address it is asked. The same class of defect
is available here in at least three places (peer resolution, history paging, FloodWait handling).

What *was* validated locally: both SQL files apply cleanly to a real PostgreSQL 16, and
`tests/test_telegram_pipeline_db.py` runs collection → derivation → outcomes → analytics → health
against a disposable schema (set `CRYPTO_SOCIAL_TEST_DATABASE_URL`, same convention as the archive's
tests). That pass caught a query whose parameters were never passed — it had been green against
every mocked test. It still proves only the SQL, not Telegram.

Before this is merged and scaled past the pilot channel set, one live run must confirm all six,
plus the manual inspection. `telegram_gate.py --run --markdown` executes them in order and prints
PR-ready evidence; it exits non-zero if any fails.

1. **Peer resolution** — 3–5 real channels resolved from usernames into peer ids, the identity
   confirmed against what the client reports, and the same ids returned on a re-check.
2. **Backfill idempotence** — a backfill page writes rows, `backfill_floor_id` decreases, and a
   re-read of the same window writes **zero** new rows. Pyrogram's history iterator has no `min_id`;
   the stop condition is ours, and if it is wrong the poll silently re-reads whole histories forever.
3. **A real contract resolves to the correct archive token** — and the stored address round-trips
   out of the stored message text, which is what would catch a normalisation or truncation bug
   between extraction and storage.
4. **Contract beats ticker** — a real message carrying both derives exactly one mention, of kind
   `contract`.
5. **One known case reconstructs** — FLEX or PHILANCAT, from the original call through
   mention-anchored outcomes, as a single line:
   `@channel → contract → first observed mention → market cap at mention → +30m/+1h/+3h/+24h`.
6. **Health separates a lost channel from a quiet one** — the gate deliberately polls a peer that
   does not exist: it must appear in `channels_losing_access` while a genuinely quiet channel stays
   `access_state='ok'` with zero new messages. This is the distinction that decides whether a silent
   dashboard means everything is fine or that collection died.
7. **Relay attribution, read by eye** — a token posted by at least two monitored channels, whose
   chain reads *original post → forwarded/reposted alert → later independent mention*, with only the
   originals holding sequence positions. This is the one item that cannot be a pass/fail assertion,
   and it is the one that would catch a plausible-looking pipeline that is crediting relays with
   discovery.

## 11. Pilot scope and what is deliberately not built

Start at 10–20 public channels (`telegram_channels.json` ships empty: the operator supplies the
list, and a channel is added by username, resolved, then backfilled). Backfill as far as Telegram
allows, then poll every 15 minutes.

Not built, deliberately:

- **No dashboard.** The brief asks for the dataset and the analytics validated first.
- **No LLM anywhere in this pipeline.** Extraction is deterministic, and both provider accounts in
  this repository are out of funds anyway (see CLAUDE.md).
- **No paid market data.** OHLCV is the same free GeckoTerminal source the archive already uses, at
  the same unkeyed ~30 requests/minute ceiling.
- **No channel quality score.** See §7.
- **No private groups, no paywalled channels, no bypassing access controls** — the account joins
  public channels, and a lost channel is recorded as lost rather than worked around.
