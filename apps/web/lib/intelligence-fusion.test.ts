import assert from "node:assert/strict";
import test from "node:test";
import { assetIdentityKey, claimResolution } from "./intelligence-fusion.ts";

test("asset identity is chain scoped and address-case aware", () => {
  assert.equal(assetIdentityKey("base", "0xAbC", "ABC"), "asset:base:0xabc");
  assert.equal(assetIdentityKey("solana", "AbC", "ABC"), "asset:solana:AbC");
  assert.equal(assetIdentityKey("ethereum", null, "ETH"), "asset:ethereum:native:ETH");
});

test("claim resolution preserves unresolved and non-verifiable states", () => {
  assert.equal(claimResolution("objective", []), "unresolved");
  assert.equal(claimResolution("subjective", []), "not_verifiable");
  assert.equal(claimResolution("objective", [
    { status: "confirmed", assessedAt: "2026-01-01T00:00:00Z" },
    { status: "contradicted", assessedAt: "2026-01-02T00:00:00Z" },
  ]), "contradicted");
});
