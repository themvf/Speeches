import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

import {
  ENFORCEMENT_HEATMAP_CACHE_CONTROL,
  ENFORCEMENT_HEATMAP_REVALIDATE_SECONDS,
  enforcementHeatmapRowsToCorpus,
  normalizeEnforcementSourceKinds,
} from "./enforcement-heatmap-corpus.ts";

test("enforcement heatmap rows retain full text and canonical row identity", () => {
  const documents = enforcementHeatmapRowsToCorpus([
    {
      document_id: "finra-123",
      source_kind: "finra_awc",
      metadata: {
        document_id: "stale-id",
        title: "AWC action",
        date: "2026-07-01",
      },
      full_text: "The firm violated FINRA Rule 2010.",
    },
    {
      document_id: "",
      source_kind: "finra_awc",
      metadata: {},
      full_text: "invalid row",
    },
  ]);

  assert.equal(documents.length, 1);
  assert.equal(documents[0].metadata.document_id, "finra-123");
  assert.equal(documents[0].metadata.source_kind, "finra_awc");
  assert.equal(documents[0].content.full_text, "The firm violated FINRA Rule 2010.");
});

test("source filters are stable, unique, and cannot become an unrestricted query", () => {
  assert.deepEqual(
    normalizeEnforcementSourceKinds([
      " sec_enforcement_litigation ",
      "finra_awc",
      "finra_awc",
      "",
    ]),
    ["finra_awc", "sec_enforcement_litigation"]
  );
  assert.deepEqual(normalizeEnforcementSourceKinds([" "]), []);
});

test("heatmap routes cannot regress to monolithic GCS corpus reads", () => {
  const webRoot = path.resolve(import.meta.dirname, "../..");
  const finraRoute = fs.readFileSync(
    path.join(webRoot, "app/api/finra/heatmap/route.ts"),
    "utf8"
  );
  const enforcementRoute = fs.readFileSync(
    path.join(webRoot, "app/api/enforcement/heatmap/route.ts"),
    "utf8"
  );
  const loader = fs.readFileSync(
    path.join(webRoot, "lib/server/enforcement-heatmap-corpus.ts"),
    "utf8"
  );

  for (const route of [finraRoute, enforcementRoute]) {
    assert.doesNotMatch(route, /loadCorpusDocuments/);
    assert.doesNotMatch(route, /force-dynamic/);
    assert.match(route, /loadEnforcementHeatmapDocuments/);
    assert.match(route, /export const revalidate = 3600/);
    assert.match(route, /ENFORCEMENT_HEATMAP_CACHE_CONTROL/);
    assert.match(route, /headers\.set\("Cache-Control"/);
  }
  assert.doesNotMatch(loader, /from ["']@\/lib\/server\/(?:gcs-loader|data-store)["']/);
  assert.doesNotMatch(loader, /downloadGcsJson|loadCorpusDocuments/);
  assert.match(loader, /FROM documents/);
  assert.match(loader, /source_kind = ANY/);
  assert.equal(ENFORCEMENT_HEATMAP_REVALIDATE_SECONDS, 3600);
  assert.equal(
    ENFORCEMENT_HEATMAP_CACHE_CONTROL,
    "public, s-maxage=3600, stale-while-revalidate=300"
  );
});
