export const DEFAULT_LIST_ID = "general";

export interface SavedItemMetadata {
  documentId?: string;
  organization?: string;
  sourceKind?: string;
  docType?: string;
  speaker?: string;
  date?: string;
  publishedAt?: string;
  wordCount?: number;
  keywords?: string[];
  topics?: string[];
  sentimentLabel?: "positive" | "negative" | "neutral" | "";
  sentimentScore?: number;
  feedKey?: string;
  author?: string;
  toneLabel?: "positive" | "negative" | "neutral" | null;
}

export interface SavedItem {
  id: string;
  type: "article" | "doc";
  title: string;
  url?: string;
  source: string;
  topic?: string;
  savedAt: string;
  listIds: string[];
  metadata?: SavedItemMetadata;
}

export interface SavedList {
  id: string;
  name: string;
  createdAt: string;
}

export interface SavedItemsPayload {
  items: SavedItem[];
  lists: SavedList[];
}
