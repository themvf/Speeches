# Decision: Use GDELT GKG for Intel Beta Evidence Links

Date: 2026-04-21

## Decision

Use the GDELT Global Knowledge Graph update feed as the first live article source for Intel Beta evidence retrieval.

The production integration should read recent GKG update archives, map source URLs and GDELT theme fields into the Intel Beta evidence model, and fall back to existing seed evidence when GDELT is unavailable or returns no useful articles.

## Why

Intel Beta needs source-level evidence behind each signal. The UI should not display search URLs as if they were articles.

DOC 2.0 was the first prototype because it directly supports targeted article retrieval. In testing, targeted DOC queries were too slow and inconsistent for an on-demand UI path. The GKG update feed is a better production first source because it is fast, includes article source URLs, and carries GDELT theme fields that can be mapped through the normalized finance taxonomy.

## Alternatives Considered

- DOC 2.0 Article List: useful for title enrichment and targeted search, but too slow for synchronous evidence loading.
- BigQuery: stronger for historical baselines, source diversity, and large-scale analytics, but heavier for first implementation.
- Event Mentions: useful when tracking a specific structured event and all mentions of that event, but less direct for theme-driven evidence streams.
- Article List RSS: useful for firehose ingestion, but not ideal for targeted signal-specific retrieval.

## Implementation Direction

1. Read the latest GDELT GKG archive URL from `lastupdate.txt`.
2. Parse recent GKG CSV ZIP archives and extract document URLs, source domains, timestamps, and raw theme fields.
3. Run raw GKG themes through the normalized theme engine.
4. Rank records by overlap with the selected signal's primary and secondary themes.
5. Map returned articles into `IntelligenceEvidenceArticle`, preserving real article URLs.
6. Do not generate search fallback URLs.
7. Keep seed evidence as a graceful fallback only.

## Follow-Up

Later phases can add DOC 2.0 title enrichment and BigQuery-backed historical baselines, anomaly scoring, clustering, and source-diversity analytics.
