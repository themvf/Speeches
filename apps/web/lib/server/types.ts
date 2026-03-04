export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

export interface ApiErrorPayload {
  ok: false;
  error: string;
  code: string;
  request_id?: string;
}

export interface ApiSuccessPayload<T> {
  ok: true;
  data: T;
  request_id?: string;
}

export interface CustomDocumentMetadata {
  document_id: string;
  title: string;
  speaker: string;
  date: string;
  url: string;
  word_count: number;
  organization: string;
  doc_type: string;
  source_filename: string;
  source_format: string;
  source_local_path: string;
  source_gcs_path: string;
  tags: string;
  source_kind: string;
  source_family: string;
  source_index_url: string;
  published_date: string;
  updated_date: string;
  last_reviewed_or_updated: string;
}

export interface CustomDocumentContent {
  full_text: string;
  paragraphs: string[];
  sentences: string[];
}

export interface CustomDocumentRecord {
  metadata: CustomDocumentMetadata;
  content: CustomDocumentContent;
  validation?: Record<string, JsonValue>;
}

export interface CustomDocumentsPayload {
  updated_at: string;
  documents: CustomDocumentRecord[];
}

export interface EnrichmentReviewPayload {
  decision: string;
  notes: string;
  reviewed_at: string;
}

export interface EnrichmentPayload {
  summary: string;
  tags: string[];
  keywords: string[];
  entities: string[];
  stance: Record<string, JsonValue>;
  evidence_spans: Array<Record<string, JsonValue>>;
  confidence: number;
}

export interface EnrichmentEntry {
  doc_id: string;
  organization: string;
  org_key: string;
  title: string;
  speaker: string;
  date: string;
  url: string;
  doc_type: string;
  word_count: number;
  status: string;
  error: string;
  model: string;
  pipeline_version: string;
  updated_at: string;
  enrichment: EnrichmentPayload;
  review: EnrichmentReviewPayload;
  reward?: Record<string, JsonValue>;
  auto_review?: Record<string, JsonValue>;
}

export interface EnrichmentStatePayload {
  version: number;
  pipeline_version: string;
  updated_at: string;
  entries: Record<string, EnrichmentEntry>;
}

export interface NewsConnectorSettingsPayload {
  updated_at: string;
  query: string;
  lookback_days: number;
  max_pages: number;
  page_size: number;
  target_count: number;
  sort_by: string;
  organization_label: string;
  domains: string;
  exclude_domains: string;
  tags_csv: string;
}

export interface DocumentListItem {
  document_id: string;
  title: string;
  organization: string;
  source_kind: string;
  doc_type: string;
  speaker: string;
  url: string;
  date: string;
  published_at: string;
  word_count: number;
  ingest_status: string;
  enrichment_status: string;
  review_decision: string;
  updated_at: string;
}
