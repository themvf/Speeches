import assert from "node:assert/strict";
import test from "node:test";

import { loadMetadataOnlyFeed, metadataRowsToCorpusDocuments } from "./document-metadata-feed.ts";

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
  // DeepSeek enrichment status/model/sentiment are intentionally absent from
  // this automatic list projection; the detail path still loads them.
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
