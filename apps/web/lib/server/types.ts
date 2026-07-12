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
  extraction_quality?: string;
  full_text_available?: boolean;
  source_local_path: string;
  source_gcs_path: string;
  tags: string;
  source_kind: string;
  source_family: string;
  source_index_url: string;
  published_at?: string;
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
  source_name?: string;
  authors?: string[];
  keywords?: string[];
  apify_actor_id?: string;
  apify_raw_keys?: string[];
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

export interface SentimentPayload {
  score: number;
  label: "positive" | "negative" | "neutral";
  rationale: string;
  model: string;
  status: string;
  error: string;
  updated_at: string;
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
  sentiment?: SentimentPayload;
  enforcement_analysis?: Record<string, JsonValue>;
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
  source_format?: string;
  extraction_quality?: string;
  full_text_available?: boolean;
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
  enrichment_summary?: string;
  enrichment_model?: string;
  enrichment_confidence?: number;
  review_decision: string;
  updated_at: string;
  sentiment_label: "positive" | "negative" | "neutral" | "";
  sentiment_score: number;
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

// Stock attention tracker (docs/stock-attention-spec.md §7)
export interface AttentionSource {
  title: string;
  permalink: string;
  subreddit: string;
  author: string;
  kind: string;
  mood: string;
}

export interface AttentionRow {
  rank: number;
  ticker: string;
  company: string;
  mentionCount: number;              // total (reddit + news) - see enhancement item 1
  redditCount: number;
  newsCount: number;
  prevMentionCount: number | null;   // prior day's total; null = no prior row
  sourceCount: number;
  subredditCount: number;
  weightedScore: number;
  mood: string;
  price: number | null;              // request-time live quote; null = no quote
  pricePct: number | null;
  // Stored market context from the rollup (item 2) - distinct from the
  // request-time price/pricePct pairing above, which reflects "right now"
  // vs. this row's "as of the rollup". Both are shown; they can differ.
  storedPriceClose: number | null;
  storedPricePct: number | null;
  volume: number | null;
  volumeVs20d: number | null;        // ratio, e.g. 2.5 = 2.5x the 20-day average
  divergence: string;                // '' | 'attention_spike_no_price_move' | 'price_move_no_attention'
  weightedMentionCount: number;      // credibility/subreddit-weighted (items 4-5); ranks the board
  qualityFlags: string[];            // item 6 manipulation-pattern annotations
  sparkline: number[];               // last N days' total_mention_count, oldest first
  topSources: AttentionSource[];
}

export interface MarketAttentionData {
  date: string | null;               // UTC day the rollup covers; null = nothing aggregated yet
  rows: AttentionRow[];
  warning?: string;
  generatedAt: string;
}

export interface AttentionHistoryPoint {
  date: string;
  mentionCount: number;
  redditCount: number;
  newsCount: number;
  priceClose: number | null;
  pricePct: number | null;
}

export interface MarketAttentionHistoryData {
  ticker: string;
  company: string;
  points: AttentionHistoryPoint[];
  warning?: string;
}

export interface IntradayAttentionRow {
  rank: number;
  ticker: string;
  decayedMentionCount: number; // freshness-weighted, unique-author count
  rawMentionCount: number;     // unique authors, no decay
}

export interface MarketAttentionIntradayData {
  hoursBack: number;
  rows: IntradayAttentionRow[];
  warning?: string;
  generatedAt: string;
}

// Activity + Authors views (see CLAUDE.md plan, 2026-07-12)
export interface AttentionActivityItem {
  sourceId: string;
  kind: string;              // 'post' | 'comment'
  subreddit: string;
  author: string;
  title: string;
  permalink: string;
  createdUtc: string;
  score: number;
  mood: string;
  tickers: string[];
}

export interface MarketAttentionActivityData {
  hoursBack: number;
  items: AttentionActivityItem[];
  warning?: string;
  generatedAt: string;
}

export interface AttentionAuthorRow {
  rank: number;
  author: string;
  itemsTotal: number;
  tickersDistinct: number;
  subredditsDistinct: number;
  topTicker: string;
  topTickerShare: number;    // 0-1
  accountCreated: string | null;
  linkKarma: number | null;
  firstSeen: string | null;
  lastSeen: string | null;
  discounted: boolean;       // currently trips the item-5 credibility discount
}

export interface MarketAttentionAuthorsData {
  rows: AttentionAuthorRow[];
  warning?: string;
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
