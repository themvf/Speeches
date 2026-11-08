import assert from "node:assert/strict";
import test from "node:test";
import { normalizePortfolioSyncToken, portfolioWorkspaceId } from "./portfolio-sync.ts";

const TOKEN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ";

test("accepts only 256-bit base64url-style workspace tokens", () => {
  assert.equal(normalizePortfolioSyncToken(TOKEN), TOKEN);
  assert.equal(normalizePortfolioSyncToken(`${TOKEN}x`), null);
  assert.equal(normalizePortfolioSyncToken("not-a-token"), null);
});

test("derives a stable identifier without storing the workspace token", () => {
  const id = portfolioWorkspaceId(TOKEN);
  assert.match(id, /^workspace:[a-f0-9]{64}$/);
  assert.equal(id, portfolioWorkspaceId(TOKEN));
  assert.ok(!id.includes(TOKEN));
});
