import { neon } from "@neondatabase/serverless";

import type { CustomDocumentMetadata, CustomDocumentRecord } from "@/lib/server/types";

/**
 * Enforcement data is updated by batch ingestion, not continuously. Keeping
 * the derived heatmaps for an hour avoids repeatedly waking Neon while still
 * refreshing much faster than the normal enforcement-source cadence.
 */
export const ENFORCEMENT_HEATMAP_REVALIDATE_SECONDS = 60 * 60;
export const ENFORCEMENT_HEATMAP_CACHE_CONTROL =
  "public, s-maxage=3600, stale-while-revalidate=300";

export type EnforcementHeatmapDocumentRow = {
  document_id: string;
  source_kind: string;
  metadata: Record<string, unknown> | null;
  full_text: string | null;
};

type HeatmapSql = ReturnType<typeof neon>;

let heatmapSql: HeatmapSql | null = null;
const warmCache = new Map<
  string,
  { expiresAt: number; documents: CustomDocumentRecord[] }
>();
const inFlight = new Map<string, Promise<CustomDocumentRecord[]>>();

function getHeatmapSql(): HeatmapSql {
  if (!heatmapSql) {
    const databaseUrl = String(process.env.DATABASE_URL || "").trim();
    if (!databaseUrl) {
      throw new Error("DATABASE_URL env var is not set");
    }
    heatmapSql = neon(databaseUrl);
  }
  return heatmapSql;
}

export function normalizeEnforcementSourceKinds(sourceKinds: string[]): string[] {
  return [...new Set(
    sourceKinds
      .map((value) => String(value || "").trim())
      .filter(Boolean)
  )].sort();
}

export function enforcementHeatmapRowsToCorpus(
  rows: EnforcementHeatmapDocumentRow[]
): CustomDocumentRecord[] {
  const documents: CustomDocumentRecord[] = [];
  for (const row of rows) {
    const documentId = String(row.document_id || "").trim();
    if (!documentId) continue;
    const sourceKind = String(row.source_kind || "").trim();
    const metadata = row.metadata && typeof row.metadata === "object"
      ? row.metadata
      : {};
    documents.push({
      metadata: {
        ...metadata,
        document_id: documentId,
        source_kind: String(metadata.source_kind || sourceKind).trim(),
      } as CustomDocumentMetadata,
      content: {
        full_text: String(row.full_text || ""),
        paragraphs: [],
        sentences: [],
      },
    });
  }
  return documents;
}

async function queryEnforcementHeatmapDocuments(
  sourceKinds: string[]
): Promise<CustomDocumentRecord[]> {
  const sql = getHeatmapSql();
  const rows = (await sql`
    SELECT document_id, source_kind, metadata, full_text
    FROM documents
    WHERE source_kind = ANY(${sourceKinds}::text[])
    ORDER BY updated_at DESC, document_id ASC
  `) as unknown as EnforcementHeatmapDocumentRow[];
  return enforcementHeatmapRowsToCorpus(rows);
}

/**
 * Load only the document families consumed by the enforcement heatmaps.
 * This intentionally fails closed if Neon is unavailable: silently falling
 * back to all_speeches.json + custom_documents.json would restore the large,
 * recurring GCS egress this reader is designed to eliminate.
 */
export async function loadEnforcementHeatmapDocuments(
  requestedSourceKinds: string[]
): Promise<CustomDocumentRecord[]> {
  const sourceKinds = normalizeEnforcementSourceKinds(requestedSourceKinds);
  if (!sourceKinds.length) return [];

  const key = sourceKinds.join("\n");
  const now = Date.now();
  const cached = warmCache.get(key);
  if (cached && cached.expiresAt > now) {
    return cached.documents;
  }

  const existing = inFlight.get(key);
  if (existing) return existing;

  const pending = queryEnforcementHeatmapDocuments(sourceKinds)
    .then((documents) => {
      warmCache.set(key, {
        expiresAt: Date.now() + ENFORCEMENT_HEATMAP_REVALIDATE_SECONDS * 1000,
        documents,
      });
      return documents;
    })
    .finally(() => {
      inFlight.delete(key);
    });
  inFlight.set(key, pending);
  return pending;
}
