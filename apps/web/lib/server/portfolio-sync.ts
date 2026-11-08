import { createHash } from "node:crypto";

export const PORTFOLIO_SYNC_HEADER = "x-portfolio-sync-token";
const TOKEN_PATTERN = /^[A-Za-z0-9_-]{43}$/;

export function normalizePortfolioSyncToken(value: string | null): string | null {
  const token = value?.trim() ?? "";
  return TOKEN_PATTERN.test(token) ? token : null;
}

export function portfolioWorkspaceId(token: string): string {
  return `workspace:${createHash("sha256").update(token).digest("hex")}`;
}
