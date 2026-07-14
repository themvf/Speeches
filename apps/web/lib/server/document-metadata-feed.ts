import type {
  CustomDocumentMetadata,
  CustomDocumentRecord,
  EnrichmentEntry,
  EnrichmentStatePayload,
} from "./types";

export type MirroredDocumentMetadataRow = {
  document_id: string;
  metadata: Record<string, unknown> | null;
};

export type MirroredDocumentProjectionRow = MirroredDocumentMetadataRow & {
  enrichment_entry?: Record<string, unknown> | null;
};

export type MetadataOnlyFeedLoadResult<T> = {
  documents: T[];
  source: "neon" | "unavailable";
  metadata_only: true;
  warning?: string;
};

/**
 * Rebuild the corpus-record shape expected by the existing list-item mapper
 * without materializing source text. The document detail endpoint remains the
 * only place that needs the full content payload.
 */
export function metadataRowsToCorpusDocuments(
  rows: MirroredDocumentMetadataRow[]
): CustomDocumentRecord[] {
  const documents: CustomDocumentRecord[] = [];

  for (const row of rows) {
    const documentId = String(row.document_id || "").trim();
    if (!documentId) continue;

    const metadata = row.metadata && typeof row.metadata === "object"
      ? row.metadata
      : {};

    documents.push({
      metadata: {
        ...metadata,
        document_id: documentId,
      } as unknown as CustomDocumentMetadata,
      content: {
        full_text: "",
        paragraphs: [],
        sentences: [],
      },
    });
  }

  return documents;
}

/**
 * Rebuild the legacy enrichment-state shape from the per-document Neon
 * projection. The caller may provide the compact list/feed projection or the
 * complete detail entry; both retain the legacy shape used by serializers.
 */
export function projectionRowsToEnrichmentState(
  rows: MirroredDocumentProjectionRow[]
): EnrichmentStatePayload {
  const entries: Record<string, EnrichmentEntry> = {};
  let updatedAt = "";
  let pipelineVersion = "";

  for (const row of rows) {
    const documentId = String(row.document_id || "").trim();
    const raw = row.enrichment_entry;
    if (!documentId || !raw || typeof raw !== "object" || Array.isArray(raw)) {
      continue;
    }

    const entry = {
      ...raw,
      doc_id: documentId,
    } as unknown as EnrichmentEntry;
    entries[documentId] = entry;

    const entryUpdatedAt = String(entry.updated_at || "").trim();
    if (entryUpdatedAt && (!updatedAt || Date.parse(entryUpdatedAt) > Date.parse(updatedAt))) {
      updatedAt = entryUpdatedAt;
    }
    if (!pipelineVersion) {
      pipelineVersion = String(entry.pipeline_version || "").trim();
    }
  }

  return {
    version: 1,
    pipeline_version: pipelineVersion,
    updated_at: updatedAt,
    entries,
  };
}

export function projectionRowsToCorpusAndEnrichment(
  rows: MirroredDocumentProjectionRow[]
): { documents: CustomDocumentRecord[]; enrichment: EnrichmentStatePayload } {
  return {
    documents: metadataRowsToCorpusDocuments(rows),
    enrichment: projectionRowsToEnrichmentState(rows),
  };
}

/** Reconstruct the legacy detail content shape from Neon's canonical text. */
export function fullTextToDocumentContent(fullTextValue: unknown): CustomDocumentRecord["content"] {
  const fullText = String(fullTextValue || "");
  const paragraphs = fullText
    .split(/(?:\r?\n){2,}/)
    .map((value) => value.trim())
    .filter(Boolean);
  const sentences = (fullText.match(/[^.!?]+(?:[.!?]+(?=\s|$)|$)/g) || [])
    .map((value) => value.trim())
    .filter(Boolean);
  return { full_text: fullText, paragraphs, sentences };
}

/**
 * Automatic feed reads fail closed. In particular, this helper has no GCS
 * loader dependency: a Neon outage cannot turn hourly polling back into
 * repeated whole-corpus downloads.
 */
export async function loadMetadataOnlyFeed<T, TRow extends MirroredDocumentMetadataRow = MirroredDocumentMetadataRow>(
  loadRows: () => Promise<TRow[]>,
  buildDocuments: (rows: TRow[]) => T[]
): Promise<MetadataOnlyFeedLoadResult<T>> {
  try {
    const rows = await loadRows();
    if (rows.length === 0) {
      return {
        documents: [],
        source: "unavailable",
        metadata_only: true,
        warning: "Neon document feed projection returned no eligible records; automatic GCS fallback is disabled",
      };
    }

    return {
      documents: buildDocuments(rows),
      source: "neon",
      metadata_only: true,
    };
  } catch (error) {
    console.error("[document-projection] Neon feed read failed", error);
    return {
      documents: [],
      source: "unavailable",
      metadata_only: true,
      warning: "Neon document feed read failed; automatic GCS fallback is disabled",
    };
  }
}
