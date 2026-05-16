import { NextResponse } from "next/server";
import { downloadGcsJson, uploadGcsJson } from "@/lib/server/gcs-loader";
import { invalidateDocumentCaches } from "@/lib/server/data-store";
import type { CustomDocumentsPayload } from "@/lib/server/types";

export const dynamic = "force-dynamic";

const BLOBS = ["custom_documents.json", "all_speeches.json"];

export async function DELETE(
  _req: Request,
  context: { params: Promise<{ documentId: string }> }
): Promise<NextResponse> {
  const { documentId } = await context.params;
  if (!documentId?.trim()) {
    return NextResponse.json({ ok: false, error: "documentId required" }, { status: 400 });
  }

  for (const blob of BLOBS) {
    const payload = await downloadGcsJson<CustomDocumentsPayload>(blob);
    if (!payload?.documents) continue;

    const before = payload.documents.length;
    const filtered = payload.documents.filter(
      (d) => String(d.metadata?.document_id ?? "").trim() !== documentId
    );

    if (filtered.length === before) continue; // not found in this blob

    const updated: CustomDocumentsPayload = {
      ...payload,
      documents: filtered,
      updated_at: new Date().toISOString(),
    };

    const saved = await uploadGcsJson(blob, updated);
    if (!saved) {
      return NextResponse.json({ ok: false, error: "Failed to write to GCS" }, { status: 500 });
    }

    invalidateDocumentCaches();
    return NextResponse.json({ ok: true, deleted_from: blob });
  }

  return NextResponse.json({ ok: false, error: "Document not found" }, { status: 404 });
}
