# Decision: Use GDELT DOC 2.0 for Intel Beta Evidence Links

Date: 2026-04-21

## Decision

Use the GDELT DOC 2.0 API as the first live article source for Intel Beta evidence retrieval.

The initial integration should query DOC 2.0 article-list JSON results, map returned article URLs into the Intel Beta evidence model, and fall back to existing seed evidence when GDELT is unavailable or returns no useful articles.

## Why

Intel Beta needs source-level evidence behind each signal. The UI should not display search URLs as if they were articles.

DOC 2.0 is the best first source because it directly supports targeted article retrieval for recent news coverage and is simpler to integrate than bulk GKG files or BigQuery-backed historical processing.

## Alternatives Considered

- GKG / BigQuery: stronger for historical baselines, source diversity, and large-scale theme analytics, but heavier for first implementation.
- Event Mentions: useful when tracking a specific structured event and all mentions of that event, but less direct for theme-driven evidence streams.
- Article List RSS: useful for firehose ingestion, but not ideal for targeted signal-specific retrieval.

## Implementation Direction

1. Convert each normalized signal theme set into a GDELT DOC query.
2. Request recent articles with `mode=artlist`, `format=json`, a bounded `timespan`, and a bounded `maxrecords`.
3. Map returned articles into `IntelligenceEvidenceArticle`, preserving real article URLs.
4. Do not generate search fallback URLs.
5. Keep seed evidence as a graceful fallback only.

## Follow-Up

Later phases can add GKG or BigQuery for historical baselines, anomaly scoring, clustering, and source-diversity analytics.
