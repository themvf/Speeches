import { createHash } from "node:crypto";
import { NextResponse } from "next/server";
import { downloadGcsJson, uploadGcsJson } from "@/lib/server/gcs-loader";
import type { CustomDocumentsPayload } from "@/lib/server/types";

export const dynamic = "force-dynamic";

const CUSTOM_DOCS_BLOB = "custom_documents.json";

function generateDocId(title: string, organization: string, date: string): string {
  const key = `custom|${organization}|${title}|${date}`.trim();
  return createHash("sha256").update(key).digest("hex").slice(0, 24);
}

function countWords(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

export async function POST(req: Request): Promise<NextResponse> {
  let body: {
    title?: string;
    organization?: string;
    source_kind?: string;
    doc_type?: string;
    speaker?: string;
    date?: string;
    url?: string;
    content?: string;
  };

  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON" }, { status: 400 });
  }

  const title = (body.title ?? "").trim();
  const content = (body.content ?? "").trim();
  if (!title) return NextResponse.json({ ok: false, error: "title is required" }, { status: 400 });
  if (!content) return NextResponse.json({ ok: false, error: "content is required" }, { status: 400 });

  const organization = (body.organization ?? "Custom").trim() || "Custom";
  const source_kind = (body.source_kind ?? "custom_document").trim() || "custom_document";
  const doc_type = (body.doc_type ?? "Document").trim() || "Document";
  const speaker = (body.speaker ?? "").trim();
  const date = (body.date ?? "").trim();
  const url = (body.url ?? "").trim();
  const now = new Date().toISOString();

  const document_id = generateDocId(title, organization, date || now);

  const paragraphs = content.split(/\n{2,}/).map((p) => p.trim()).filter(Boolean);
  const word_count = countWords(content);

  const newDoc = {
    metadata: {
      document_id,
      title,
      speaker,
      date,
      url,
      word_count,
      organization,
      doc_type,
      source_filename: "",
      source_format: "text",
      source_local_path: "",
      source_gcs_path: "",
      tags: "",
      source_kind,
      source_family: "custom",
      source_index_url: "",
      published_date: date,
      updated_date: now,
      last_reviewed_or_updated: now,
    },
    content: {
      full_text: content,
      paragraphs,
      sentences: [] as string[],
    },
  };

  const existing = (await downloadGcsJson<CustomDocumentsPayload>(CUSTOM_DOCS_BLOB)) ?? {
    updated_at: "",
    documents: [],
  };

  const isDuplicate = (existing.documents ?? []).some(
    (d) => String(d.metadata?.document_id ?? "").trim() === document_id
  );
  if (isDuplicate) {
    return NextResponse.json(
      { ok: false, error: "A document with this title/org/date already exists" },
      { status: 409 }
    );
  }

  const updated: CustomDocumentsPayload = {
    ...existing,
    documents: [...(existing.documents ?? []), newDoc],
    updated_at: now,
  };

  const saved = await uploadGcsJson(CUSTOM_DOCS_BLOB, updated);
  if (!saved) {
    return NextResponse.json({ ok: false, error: "Failed to write to GCS" }, { status: 500 });
  }

  return NextResponse.json({ ok: true, document_id });
}
