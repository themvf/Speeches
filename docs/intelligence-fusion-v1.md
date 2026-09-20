# Intelligence fusion V1

This slice turns saved social and chain records into a source-neutral dossier. It does not collect
new data and does not call X, Telegram, an LLM, or a market provider while reading a dossier.

## Model

Raw source records are preserved by their owning archives. The fusion layer creates immutable,
hash-addressed observations from those records, then relates observations to canonical assets,
claims, and independently observed events. A claim is something a source asserted. An event is a
stored occurrence accepted under a named, versioned verification policy. An assessment connects the
two without rewriting either.

Asset identity is `network + contract address` when a contract exists. EVM addresses are normalized
to lowercase; case-sensitive chain addresses are preserved. Native assets use a chain-scoped native
symbol and never receive an invented contract. Tickers, names, and aliases are discovery metadata.

V1 extracts only graduation, listing, pool-creation, and transfer claims. A post is labeled `repeats`
only when the saved source metadata shows a repost dependency; chronological order alone does not
prove propagation. A second assertion is not promoted to `independently_corroborates` merely because
two accounts made similar statements. Subjective claims can be stored later but must resolve as
`not_verifiable`, not as factual successes or failures.

## Sources

- X observations come only from saved `crypto_social_*` rows.
- On-chain observations and events come only from saved `launchpad_*` rows. Graduation, pool
  creation, and exact-transaction-hash transfer verification are supported.
- Telegram is represented in the source contracts and API as `not_configured`. There are no
  credentials, provider calls, or assumed account mappings.
- Exchange, official-web, and GitHub source values are reserved for later adapters.

## Time and outcomes

Observations preserve publication and collection times. Events preserve occurrence and observation
times. Assessments preserve verification time. Saved market measurements are linked to explicit
`first_social_observation`, `event_occurred`, and `verified_at` anchors with exact timestamps and
horizons. These are reproducible associations; the model makes no causal claim.

## Operation

Preview the materialization plan (the default makes no writes):

```powershell
python intelligence_fusion.py
```

Apply the additive schema and derive dossiers from saved archives:

```powershell
$env:DATABASE_URL = "postgresql://..."
python intelligence_fusion.py --execute
```

Limit a run with repeatable `--coin SYMBOL` arguments. The read API is
`GET /api/market/crypto/dossier?coin=SYMBOL`. If the schema or asset has not been materialized, it
returns an explicit status rather than collecting data as a side effect.

The production materializer runs every 15 minutes in `intelligence-fusion.yml`. It only reads saved
database archives and records zero external-provider calls in its run metadata.

## Deliberate V1 limits

The first slice does not claim platform recall, infer account ownership across platforms, assign a
composite influencer score, infer propagation independence, fetch reply conversations, or perform
OCR. Collection diagnostics, propagation graphs, selective expansion, and source calibration build
on this schema without changing raw observations.
