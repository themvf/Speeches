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
  notice_type?: string;
  notice_number?: string;
  notice_title?: string;
  notice_url?: string;
  file_number?: string;
  release_numbers?: string[];
  rule_type?: string;
  sec_issue_date?: string;
  federal_register_publish_date?: string;
  source_notice_url?: string;
  comment_url?: string;
  comments_url?: string;
  commenter_name?: string;
  commenter_org?: string;
  letter_type?: string;
  effective_date?: string;
  comment_deadline?: string;
  pdf_url?: string;
  release_no?: string;
  case_id?: string;
  subject_text?: string;
  case_summary?: string;
  action_type?: string;
  forum?: string;
  outcome_status?: string;
  alleged_violations?: string[];
  entities?: string[];
  respondents?: string[];
  sanctions?: string[];
  sanctions_text?: string;
  detail_url?: string;
  discovery_source?: string;
  input_url?: string;
  docket_id?: string;
  docket_url?: string;
  document_url?: string;
  rule_url?: string;
  comment_id?: string;
  comment_page_url?: string;
  resolved_content_url?: string;
  attachment_urls?: string[];
  extraction_mode?: string;
  extraction_warnings?: string[];
  summary?: string;
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
  comment_position: Record<string, JsonValue>;
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
  doj_usao_exclude_terms: string;
}

export interface RuleSummaryOverviewTopic {
  label: string;
  count: number;
  share: number;
}

export interface RuleSummaryOverview {
  total_comments: number;
  enriched_comments: number;
  position_counts: Record<string, number>;
  top_topics: RuleSummaryOverviewTopic[];
}

export interface RuleSummaryGroup {
  notice_key: string;
  source_kind: string;
  source_family: string;
  source_family_label: string;
  group_type_label: string;
  group_identifier_label: string;
  group_identifier: string;
  notice_document_id: string;
  notice_number: string;
  docket_id: string;
  title: string;
  summary: string;
  organization: string;
  url: string;
  pdf_url: string;
  published_at: string;
  effective_date: string;
  comment_deadline: string;
  tags: string[];
  keywords: string[];
  enrichment_status: string;
  review_decision: string;
  comment_count: number;
  latest_comment_at: string;
  overview: RuleSummaryOverview;
  comment_document_ids: string[];
}

export interface RuleSummariesPayload {
  version: number;
  updated_at: string;
  generated_at: string;
  custom_documents_updated_at: string;
  enrichment_state_updated_at: string;
  totals: {
    notices: number;
    comments: number;
    enriched_comments: number;
    pending_review_comments: number;
  };
  groups: RuleSummaryGroup[];
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
  tags: string[];
  keywords: string[];
  topics: string[];
  ingest_status: string;
  enrichment_status: string;
  review_decision: string;
  updated_at: string;
}

export interface DocumentsFacets {
  sources: string[];
  organizations: string[];
  topics: string[];
  key_topics: string[];
  keywords: string[];
  statuses: string[];
}

export interface DocumentsListResponseData {
  items: DocumentListItem[];
  page: number;
  page_size: number;
  total: number;
  facets: DocumentsFacets;
}

export interface TimelineBucketSourceCount {
  source_kind: string;
  count: number;
}

export interface TimelineBucket {
  key: string;
  label: string;
  start: string;
  end: string;
  count: number;
  source_counts: TimelineBucketSourceCount[];
}

export interface TimelineSummary {
  matching_documents: number;
  dated_documents: number;
  undated_documents: number;
  bucket_count: number;
  peak_bucket_key: string;
  peak_bucket_label: string;
  peak_bucket_count: number;
  start_date: string;
  end_date: string;
}

export interface TimelineResponseData {
  grain: "month" | "quarter" | "year";
  buckets: TimelineBucket[];
  totals: TimelineSummary;
  facets: DocumentsFacets;
}

export type GraphNodeKind = "document" | "organization" | "speaker" | "topic" | "keyword" | "entity";

export type GraphEdgeKind =
  | "published_by"
  | "spoken_by"
  | "has_topic"
  | "has_keyword"
  | "mentions_entity"
  | "org_topic"
  | "org_keyword"
  | "org_entity"
  | "speaker_topic"
  | "topic_entity";

export interface GraphNode {
  id: string;
  kind: GraphNodeKind;
  label: string;
  document_count: number;
  degree: number;
  metadata: Record<string, JsonValue>;
}

export interface GraphEdge {
  id: string;
  kind: GraphEdgeKind;
  source: string;
  target: string;
  weight: number;
  document_count: number;
  evidence_doc_ids: string[];
  metadata: Record<string, JsonValue>;
}

export interface GraphSummary {
  matching_documents: number;
  node_count: number;
  edge_count: number;
  returned_nodes: number;
  returned_edges: number;
  include_documents: boolean;
  nodes_by_kind: Record<string, number>;
  edges_by_kind: Record<string, number>;
  start_date: string;
  end_date: string;
}

export interface GraphResponseData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  summary: GraphSummary;
  facets: DocumentsFacets;
}

export interface Neo4jStatusResponseData {
  configured: boolean;
  database: string;
  url: string;
  missing_required_env: string[];
}

export interface TrendSparklinePoint {
  date: string;
  count: number;
}

export interface TrendDocItem {
  id: string;
  title: string;
  date: string;
  source_kind: string;
  url: string;
  summary: string;
}

export interface TrendItem {
  id: string;
  label: string;
  canonical_tag: string;
  cluster_tags: string[];
  description: string;
  total_mentions: number;
  recent_mentions: number;
  growth_pct: number;
  first_seen: string;
  last_seen: string;
  sparkline: TrendSparklinePoint[];
  top_doc_ids: string[];
  top_docs: TrendDocItem[];
  sources: string[];
}

export interface TrendsPayload {
  version: number;
  generated_at: string;
  trend_count: number;
  trends: TrendItem[];
}

/* ── Market page types ──────────────────────────────────────────────────── */

export type MarketStatus = "OPEN" | "CLOSED" | "PRE" | "AFTER";
export type FearGreedLabel = "GREED" | "CALM" | "CONCERN" | "PANIC";

export interface IndexPcts {
  d1: number;
  w1: number;
  m1: number;
  ytd: number;
}

export interface MarketIndexQuote {
  symbol: string;
  name: string;
  price: number;
  change: number;
  pct: number;
  pcts: IndexPcts;
  sparkline: number[];
  up: boolean;
  status: MarketStatus;
}

export interface VixQuote {
  value: number;
  change: number;
  pct: number;
  label: FearGreedLabel;
  gradientPct: number;
}

export interface MarketOverviewData {
  indices: MarketIndexQuote[];
  vix: VixQuote | null;
  globalIndices: MarketIndexQuote[];
  generatedAt: string;
}

export interface SectorStock {
  symbol: string;
  name: string;
  price: number;
  pct: number;
  change: number;
  up: boolean;
}

export interface SectorPcts {
  d1: number;
  w1: number;
  m1: number;
  m3: number;
  ytd: number;
}

export interface SectorData {
  name: string;
  pcts: SectorPcts;
  stocks: SectorStock[];
}

export interface MarketSectorsData {
  sectors: SectorData[];
  generatedAt: string;
}

export interface MoverQuote {
  rank: number;
  symbol: string;
  name: string;
  price: number;
  pct: number;
  change: number;
  up: boolean;
}

export interface MarketMoversData {
  gainers: MoverQuote[];
  losers: MoverQuote[];
  generatedAt: string;
}

export interface CryptoCoin {
  rank: number;
  id: string;
  symbol: string;
  name: string;
  price: number;
  pct24h: number;
  pct7d: number;
  pct30d: number;
  marketCap: number;
  volume24h: number;
  up: boolean;
}

export interface MarketCryptoData {
  coins: CryptoCoin[];
  generatedAt: string;
}

export interface ExchangeInfo {
  code: string;
  name: string;
  timezone: string;
  status: MarketStatus;
}

export interface ExchangeRegionGroup {
  region: "Americas" | "Europe" | "Asia Pacific";
  exchanges: ExchangeInfo[];
}

export interface MarketExchangesData {
  regions: ExchangeRegionGroup[];
  generatedAt: string;
}

export type CommodityCategory = "metals" | "energy" | "agriculture";

export interface CommodityQuote {
  symbol: string;
  name: string;
  price: number;
  change: number;
  pct: number;
  up: boolean;
  category: CommodityCategory;
}

export interface MarketCommoditiesData {
  commodities: CommodityQuote[];
  generatedAt: string;
}

export interface TreasuryYield {
  label: string;
  rate: number;
  change: number;
  pct: number;
  up: boolean;
}

export interface MarketBondsData {
  yields: TreasuryYield[];
  dxy: { price: number; change: number; pct: number; up: boolean } | null;
  generatedAt: string;
}

export interface Neo4jPathResponseData {
  projection_key: string;
  synced_node_count: number;
  synced_edge_count: number;
  path_found: boolean;
  hops: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
}
