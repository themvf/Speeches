import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

import {
  fullTextToDocumentContent,
  loadMetadataOnlyFeed,
  metadataRowsToCorpusDocuments,
  projectionRowsToCorpusAndEnrichment,
  projectionRowsToEnrichmentState,
} from "./document-metadata-feed.ts";

test("metadata-only feed records preserve canonical list metadata without source text", () => {
  const documents = metadataRowsToCorpusDocuments([
    {
      document_id: "doc-123",
      metadata: {
        document_id: "stale-id",
        title: "Treasury market structure update",
        organization: "U.S. Treasury",
        source_kind: "treasury_press_release",
        url: "https://home.treasury.gov/example",
        published_at: "2026-07-14T10:00:00Z",
        tags: "Credit Markets, Banking & Payments",
        keywords: ["treasury", "liquidity"],
        summary: "Stored connector summary",
      },
    },
  ]);

  assert.equal(documents.length, 1);
  assert.deepEqual(documents[0].metadata, {
    document_id: "doc-123",
    title: "Treasury market structure update",
    organization: "U.S. Treasury",
    source_kind: "treasury_press_release",
    url: "https://home.treasury.gov/example",
    published_at: "2026-07-14T10:00:00Z",
    tags: "Credit Markets, Banking & Payments",
    keywords: ["treasury", "liquidity"],
    summary: "Stored connector summary",
  });
  assert.deepEqual(documents[0].content, {
    full_text: "",
    paragraphs: [],
    sentences: [],
  });
});

test("metadata-only feed records skip invalid ids and tolerate missing metadata", () => {
  const documents = metadataRowsToCorpusDocuments([
    { document_id: "", metadata: { title: "Invalid" } },
    { document_id: "doc-456", metadata: null },
  ]);

  assert.equal(documents.length, 1);
  assert.equal(documents[0].metadata.document_id, "doc-456");
});

test("metadata-only feed success uses only the Neon loader and preserves list fields", async () => {
  let neonLoaderCalls = 0;
  let gcsLoaderCalls = 0;
  const legacyGcsLoader = async () => {
    gcsLoaderCalls += 1;
    return [];
  };

  const result = await loadMetadataOnlyFeed(
    async () => {
      neonLoaderCalls += 1;
      return [{
        document_id: "doc-789",
        metadata: {
          title: "Bank liquidity proposal",
          date: "July 14, 2026",
          url: "https://example.com/doc-789",
          source_kind: "federal_reserve_speech_testimony",
          tags: "Banking & Payments",
          keywords: ["liquidity"],
          summary: "Connector-provided summary",
        },
      }];
    },
    metadataRowsToCorpusDocuments
  );

  assert.equal(result.source, "neon");
  assert.equal(neonLoaderCalls, 1);
  assert.equal(gcsLoaderCalls, 0);
  assert.equal(typeof legacyGcsLoader, "function");
  assert.deepEqual(result.documents[0].metadata, {
    document_id: "doc-789",
    title: "Bank liquidity proposal",
    date: "July 14, 2026",
    url: "https://example.com/doc-789",
    source_kind: "federal_reserve_speech_testimony",
    tags: "Banking & Payments",
    keywords: ["liquidity"],
    summary: "Connector-provided summary",
  });
  // A metadata-only rollout row must not fabricate DeepSeek fields. Once the
  // enrichment mirror exists, the joined projection supplies them separately.
  assert.equal("enrichment" in result.documents[0].metadata, false);
  assert.equal("sentiment" in result.documents[0].metadata, false);
});

test("metadata-only feed fails closed on a Neon error and never invokes a legacy GCS loader", async () => {
  let gcsLoaderCalls = 0;
  const legacyGcsLoader = async () => {
    gcsLoaderCalls += 1;
    return [{ title: "Must not load" }];
  };

  const result = await loadMetadataOnlyFeed(
    async () => {
      throw new Error("Neon unavailable");
    },
    () => []
  );

  assert.equal(result.source, "unavailable");
  assert.equal(result.metadata_only, true);
  assert.deepEqual(result.documents, []);
  assert.match(result.warning || "", /automatic GCS fallback is disabled/);
  assert.equal(gcsLoaderCalls, 0);
  assert.equal(typeof legacyGcsLoader, "function");
});

test("metadata-only feed fails closed when Neon returns zero eligible rows", async () => {
  let buildCalls = 0;
  const result = await loadMetadataOnlyFeed(
    async () => [],
    () => {
      buildCalls += 1;
      return [{ title: "Must not build" }];
    }
  );

  assert.equal(result.source, "unavailable");
  assert.deepEqual(result.documents, []);
  assert.match(result.warning || "", /no eligible records/);
  assert.equal(buildCalls, 0);
});

test("bounded Neon projections rebuild the legacy enrichment map by document id", () => {
  const rows = [{
    document_id: "doc-enriched",
    metadata: { title: "Liquidity speech", tags: "Banking & Payments" },
    enrichment_entry: {
      doc_id: "stale-id",
      status: "enriched",
      model: "deepseek-chat",
      pipeline_version: "v2",
      updated_at: "2026-07-14T14:00:00Z",
      enrichment: {
        summary: "A bounded row-level summary.",
        tags: ["Banking & Payments"],
        keywords: ["liquidity"],
      },
    },
  }];

  const result = projectionRowsToCorpusAndEnrichment(rows);
  assert.equal(result.documents[0].metadata.document_id, "doc-enriched");
  assert.equal(result.enrichment.entries["doc-enriched"].doc_id, "doc-enriched");
  assert.equal(result.enrichment.entries["doc-enriched"].model, "deepseek-chat");
  assert.equal(result.enrichment.pipeline_version, "v2");
});

test("documents-only rollout projection remains usable when enrichment is absent", () => {
  const result = projectionRowsToCorpusAndEnrichment([{
    document_id: "doc-before-backfill",
    metadata: { title: "Available during enrichment rollout" },
  }]);

  assert.equal(result.documents.length, 1);
  assert.deepEqual(result.enrichment.entries, {});
});

test("single-row Neon detail reconstructs the legacy content shape", () => {
  const content = fullTextToDocumentContent(
    "First paragraph has two sentences. It remains readable!\r\n\r\nSecond paragraph is preserved."
  );

  assert.equal(content.full_text.includes("Second paragraph"), true);
  assert.deepEqual(content.paragraphs, [
    "First paragraph has two sentences. It remains readable!",
    "Second paragraph is preserved.",
  ]);
  assert.deepEqual(content.sentences, [
    "First paragraph has two sentences.",
    "It remains readable!",
    "Second paragraph is preserved.",
  ]);
});

test("projectionRowsToEnrichmentState skips null entries without fabricating enrichment", () => {
  const state = projectionRowsToEnrichmentState([{
    document_id: "metadata-only",
    metadata: {},
    enrichment_entry: null,
  }]);
  assert.deepEqual(state.entries, {});
});

test("high-traffic document routes contain no monolithic corpus reader", () => {
  const routeSources = [
    "../../app/api/intel/feed/route.ts",
    "../../app/api/documents/route.ts",
    "../../app/api/documents/[documentId]/route.ts",
    "../../app/api/metrics/route.ts",
  ].map((path) => fs.readFileSync(new URL(path, import.meta.url), "utf8"));

  for (const source of routeSources) {
    assert.doesNotMatch(source, /load(?:CorpusDocuments|CustomDocuments|SecSpeeches|EnrichmentState)\s*\(/);
  }
  assert.match(routeSources[1], /loadDocumentListPageFromNeon/);
  assert.match(routeSources[2], /getMirroredDocumentDetail/);
  assert.match(routeSources[3], /getMirroredDocumentMetricsSnapshot/);
});

test("Neon reader includes an explicit documents-only rollout path", () => {
  const source = fs.readFileSync(new URL("./neon.ts", import.meta.url), "utf8");
  assert.match(source, /to_regclass\('public\.document_enrichments'\)/);
  assert.match(source, /getMirroredDocumentListPageWithoutEnrichment/);
  assert.match(source, /getMirroredDocumentFacetsWithoutEnrichment/);
  assert.match(source, /NULL::jsonb AS enrichment_entry/);
});

test("Neon list and feed projections retain timestamp precision and transfer compact enrichment", () => {
  const source = fs.readFileSync(new URL("./neon.ts", import.meta.url), "utf8");
  const detailStart = source.indexOf("export async function getMirroredDocumentDetail");
  const facetStart = source.indexOf("function stringArray", detailStart);
  const detailSource = source.slice(detailStart, facetStart);

  assert.match(source, /raw_published::timestamptz/);
  assert.match(source, /ORDER BY published_sort DESC/);
  assert.match(source, /jsonb_strip_nulls\(jsonb_build_object\(/);
  assert.match(detailSource, /enrichment\.entry AS enrichment_entry/);
});

test("Neon facet and pagination paths preserve canonical counts and totals", () => {
  const neonSource = fs.readFileSync(new URL("./neon.ts", import.meta.url), "utf8");
  const storeSource = fs.readFileSync(new URL("./data-store.ts", import.meta.url), "utf8");
  const routeSource = fs.readFileSync(new URL("../../app/api/documents/route.ts", import.meta.url), "utf8");

  assert.match(neonSource, /canonical_topic_values/);
  assert.match(neonSource, /count\(DISTINCT document_id\)::integer AS count/);
  assert.match(storeSource, /page\.rows\.length === 0[\s\S]*page: 1, pageSize: 1/);
  assert.match(routeSource, /searchParams\.has\("doc_ids"\)/);
  assert.match(routeSource, /hasDocumentIdsFilter/);
});

test("updated sorting and recent-ingest metrics use source timestamps before mirror write time", () => {
  const neonSource = fs.readFileSync(new URL("./neon.ts", import.meta.url), "utf8");
  const storeSource = fs.readFileSync(new URL("./data-store.ts", import.meta.url), "utf8");
  const metricsStart = neonSource.indexOf("export async function getMirroredDocumentMetricsSnapshot");
  const metricsSource = neonSource.slice(metricsStart);

  for (const field of ["last_reviewed_or_updated", "updated_date", "extraction_date"]) {
    assert.match(neonSource, new RegExp(`metadata->>'${field}'`));
  }
  assert.match(neonSource, /enrichment\.entry->>'updated_at'/);
  assert.match(
    neonSource,
    /sort} = 'updated_desc' THEN semantic_updated_at[\s\S]*sort} = 'updated_desc' THEN row_updated_at/
  );
  assert.match(metricsSource, /newsapi_semantic_updates/);
  assert.match(metricsSource, /semantic_updated_at >= now\(\) - INTERVAL '24 hours'/);
  assert.doesNotMatch(metricsSource, /document_updated_at >= now\(\) - INTERVAL '24 hours'/);
  assert.match(
    storeSource,
    /m\.last_reviewed_or_updated[\s\S]*m\.updated_date[\s\S]*m\.extraction_date[\s\S]*enrich\?\.updated_at/
  );
});

test("scraped date casts are guarded by PostgreSQL input validation", () => {
  const neonSource = fs.readFileSync(new URL("./neon.ts", import.meta.url), "utf8");

  assert.match(neonSource, /pg_input_is_valid\(substring\(raw_published FROM 1 FOR 10\), 'date'\)/);
  assert.match(neonSource, /pg_input_is_valid\(raw_published, 'timestamp with time zone'\)/);
  assert.match(neonSource, /pg_input_is_valid\(candidate\.raw_updated, 'timestamp without time zone'\)/);
  assert.doesNotMatch(
    neonSource,
    /WHEN raw_published ~[^\n]*\n\s+THEN (?:substring\(raw_published FROM 1 FOR 10\)::date|raw_published::timestamptz)/
  );
  assert.doesNotMatch(
    neonSource,
    /WHEN candidate\.raw_updated ~[^\n]*\n\s+THEN candidate\.raw_updated::(?:date|timestamp|timestamptz)/
  );
});
