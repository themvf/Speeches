import type { CoinConfig } from "./crypto-coins.ts";

export type CryptoBriefWindow = "24h" | "7d" | "30d";
export type CryptoBriefMatchMethod = "contract" | "cashtag" | "name_context" | "declared_account";

export type BriefEvidence = {
  id: string;
  source: "crypto_archive" | "rss_x";
  sourceRecordIds: string[];
  authorId: string | null;
  handle: string | null;
  text: string;
  url: string | null;
  publishedAt: string | null;
  observedAt: string | null;
  kind: "original" | "reply" | "quote" | "repost" | "unknown";
  match: { coin: string; method: CryptoBriefMatchMethod; provisional: boolean; reason: string };
};

export type BriefCoverage = {
  status: "available" | "partial" | "unavailable";
  windowStart: string;
  windowEnd: string;
  latestPublishedAt: string | null;
  lastCollectedAt: string | null;
  matchingPostCount: number | null;
  includedPostCount: number;
  includedAuthorCount: number | null;
  evidenceLimit: number;
  truncated: boolean;
  excludedUndatedCount: number;
  collectionComplete: boolean | null;
  comparableBaseline: boolean;
  reasons: string[];
};

export type BriefTakeaway = {
  id: string;
  category: string;
  text: string;
  basis: "post_excerpt" | "stored_analysis" | "named_rule";
  whyItMatters: string | null;
  watchNext: string | null;
  evidenceIds: string[];
  ruleId: string | null;
  limitations: string[];
};

export type CryptoBrief = {
  schemaVersion: "crypto-brief-v1";
  asset: Pick<CoinConfig, "symbol" | "name" | "network" | "networkLabel" | "address" | "identityNote">;
  window: CryptoBriefWindow;
  asOf: string;
  coverage: BriefCoverage;
  overview: string;
  takeaways: BriefTakeaway[];
  evidence: BriefEvidence[];
};

export type CryptoBriefSourcePost = {
  id: string;
  author_id: string | null;
  handle: string | null;
  text: string;
  url: string | null;
  posted_at: string | Date | null;
  first_seen_at?: string | Date | null;
  kind: string | null;
};
