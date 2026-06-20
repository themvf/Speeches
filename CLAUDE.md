# Speeches — Developer Guide

## GCS Cost Best Practices

Google Cloud Storage is the primary data store for this project. GCS egress and operation costs have historically been a significant expense. Follow these rules when writing code that touches GCS.

### Always go through the cache

All GCS reads must go through the module-level in-process cache in `apps/web/lib/server/data-store.ts` (for corpus/enrichment/settings blobs) or the equivalent cache in `apps/web/lib/server/vector-state.ts`. Never call `downloadGcsJson` directly from an API route or component — always call the typed loader functions (`loadCorpusDocuments`, `loadEnrichmentState`, `loadVectorStoreState`, etc.).

The current cache TTL is **5 minutes**. Do not lower it. Raising it further is fine for blobs that change only on workflow runs.

### Invalidate on write, not on read

When you add a new GCS write path (e.g. `uploadGcsJson`), call `clearCacheKey` (in `data-store.ts`) immediately after a successful write. Never clear the cache before reading — that defeats the purpose.

### Do not bypass the cache for "freshness"

Do not add `cache: "no-store"` to `fetch` calls for GCS-backed API routes, and do not set `export const revalidate = 0` on routes that serve GCS data unless there is a hard real-time requirement. Most data in this app (speeches, enrichment, vector state) is updated by GitHub Actions workflows, not by users — stale-by-5-minutes is always acceptable.

### Add CDN caching for public API routes

Routes that serve GCS-backed data to browsers should include `Cache-Control: public, s-maxage=3600, stale-while-revalidate=86400` (or similar) on responses that do not depend on the authenticated user's identity. See `apps/web/app/api/intel/feed/route.ts` for the pattern: skip the CDN header only when the caller explicitly requests a refresh.

### Conditional workflow steps

GitHub Actions workflows that extract data and then enrich it should gate the enrichment step on whether extraction produced new or changed documents. See `.github/workflows/bloomberg-public-hourly.yml` and `substack-public-3hour.yml` for the pattern: write an extraction summary JSON, read `saved_new + saved_updates`, and set a step output; downstream enrichment steps use `if: steps.extraction_summary.outputs.changed_count != '0'`.

### Scheduled workflow frequency

Before adding or increasing the frequency of a scheduled workflow, consider the downstream cost:
- Extraction workflows trigger `knowledge-index-sync`, which does concurrent OpenAI file uploads.
- Enrichment workflows make LLM API calls per document.
- Running hourly instead of daily is a 24x cost multiplier.

Use the minimum frequency that meets the freshness requirement.

### New GCS blobs

When introducing a new GCS blob:
1. Add a typed loader function in `data-store.ts` using `loadFromSource` — this gives it the 5-minute cache automatically.
2. Add an `invalidate*` or `clearCacheKey` call alongside any write function for that blob.
3. Do not load the blob inside a loop or inside a function that is called per-document during enrichment.
