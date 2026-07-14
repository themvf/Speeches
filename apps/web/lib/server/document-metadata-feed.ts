import type { CustomDocumentMetadata, CustomDocumentRecord } from "./types";

export type MirroredDocumentMetadataRow = {
  document_id: string;
  metadata: Record<string, unknown> | null;
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
 * Automatic feed reads fail closed. In particular, this helper has no GCS
 * loader dependency: a Neon outage cannot turn hourly polling back into
 * repeated whole-corpus downloads.
 */
export async function loadMetadataOnlyFeed<T>(
  loadRows: () => Promise<MirroredDocumentMetadataRow[]>,
  buildDocuments: (rows: MirroredDocumentMetadataRow[]) => T[]
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
    const message = error instanceof Error ? error.message : "Unknown error";
    return {
      documents: [],
      source: "unavailable",
      metadata_only: true,
      warning: `Neon document feed read failed; automatic GCS fallback is disabled: ${message}`,
    };
  }
}
