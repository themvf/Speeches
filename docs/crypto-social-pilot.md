# ZCAT / Zcash social research pilot

This adds saved X research to Market → Crypto, backed by the existing Neon
DATABASE_URL. The dashboard is read-only and never spends TwitterAPI.io credits.
It works independently of the CoinGecko market-price request.

## Activation

1. Add TWITTERAPI_IO_API_KEY to the repository's GitHub Actions secrets. Reuse
   the existing DATABASE_URL secret, pointing to the same Neon database as Vercel.
   Never paste a key into a commit, workflow input, or chat.
2. Run **Crypto Social Research Pilot (Manual)** with execute=false to inspect
   the plan without database access or paid calls.
3. Run execute=true, max_requests=14 for the smoke test. It reserves at most
   4,200 credits ($0.042 at 100,000 credits/USD). Inspect request outcomes and
   source posts before running more. For a new workflow on a feature branch,
   use GitHub's workflow dispatch API with the branch ref, or merge first.
4. Subsequent manual runs resume the same seven-day period and ledger. The
   dashboard appears once the web branch is deployed; it does not start collection.

Local equivalents:

```bash
python crypto_social_pilot.py
python crypto_social_pilot.py --execute --max-requests 14
```

Schema is created additively on first execution using sql/crypto_social.sql.
No existing speech, RSS, or market tables are changed. Credentials are required
before initialization. Default CLI mode makes no network requests or writes.

## Budget and collection

The persistent singleton `zcat-zec-v1` is capped at 50,000 credits ($0.50).
Each attempted search reserves 300 credits before any network call, in a locked
transaction: the documented 20-post page maximum × 15 credits per post. Reruns
never reset the budget/date range, and reservations are not refunded even for
empty pages. Estimated usage is recorded separately. The ceiling applies to this
collector under the published rates/page contract, not to manual MCP or other
API use on the account. Recheck rates before activation if pricing changes.

Seven complete UTC days are frozen at initialization. Each coin has four
six-hour windows/day. Collection cycles through all windows once before fetching
second pages, at most two pages/window. That bounds this baseline to 112 calls,
33,600 reserved credits, up to 2,240 post retrievals; it does not force spending
all 50,000 credits. Searches use Latest with since_time/until_time epoch filters.
No real-time streams, profile/follower crawling, LLM calls, or recurring schedule
are enabled. Account details and interaction edges come from returned posts.

Requests with uncertain outcomes retain their reservation and block further
collection, including simultaneous invocations. Timeouts and rate-limit errors
are not retried. A killed process leaves a reserved record that also blocks.
An operator must inspect the provider and database ledger before resolving a
reserved/uncertain row; do not remove the reservation or automatically replay.
Invalid timestamps, out-of-window posts, stalled pagination, duplicate IDs inside
a page, and changed response shapes stop collection. Successful pages, matches,
metrics, graph edges and cursor progress commit atomically.

## Data and interpretation

- ZCAT is Anonymous Cat, using the user-supplied address
  HcRLc9VDgjLeK154xDawfb1dmVJ98DoSqcwTHGqiDeJR. Identity has not yet been
  independently verified; no price/on-chain mapping is asserted.
- ZEC uses Zcash and its cashtag. No language filter is applied.
- Posts have stable text IDs, source links, UTC timestamps and original text.
  A post can match both coins; it is stored once with separate query/window links.
- Profile observations store handle/name/follower count from search responses.
  Engagement snapshots preserve observed-at timestamps separately from posting time.
- Daily counts are observed matches, not platform totals. A searched window with
  no results is distinct from an unsearched window. Even exhausted pagination is
  not evidence that X search returned every matching post.
- Candidate rankings use distinct observed quote/repost actors, then sampled post
  volume. Observed posts/day divides the sample count by all seven calendar days.
  This is neither total account posting frequency nor a population estimate.
  Median likes are age-confounded snapshots, not normalized influence scores.
- Graph arrows represent observed reply, quote, repost or mention relationships;
  all stored edges retain their source post. Likes supply counts only, not identities.
  Quotes/replies can be criticism. Neither interaction nor repetition proves
  endorsement, common control, coordination, or manipulation.
- Network display shows up to 20 nodes drawn from the strongest 60 aggregated
  connections. The evidence list supplies accessible links, including connections
  omitted from the visual. The database retains the full observed edge set.

## Next phase after inspecting pilot results

Verify ZCAT identity and sample query precision; inspect completeness and rejection
rates; select an account shortlist. Total account posts/day requires separately
bounded timeline collection. Longer baseline, account-role review, targeted
retweeter/reply enrichment, historical profile snapshots, alerts and recurring
collection are follow-on work, not silently activated by this pilot.

## Validation

```bash
python -m pytest tests/test_crypto_social_pilot.py -q
cd apps/web && npm run typecheck
```

Three optional Postgres integration tests use CRYPTO_SOCIAL_TEST_DATABASE_URL.
Use only a disposable test database: they recreate the crypto_test schema.
