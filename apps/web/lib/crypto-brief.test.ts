import test from "node:test";
import assert from "node:assert/strict";
import { COINS, coinConfig } from "./crypto-coins.ts";
import { CRYPTO_BRIEF_COINS, cryptoBriefConfig } from "./crypto-brief-config.ts";
import { buildCryptoBrief, resolveCryptoBriefCoin } from "./crypto-brief.ts";
import type { CryptoBriefSourcePost } from "./crypto-brief-types.ts";

const NOW = "2026-09-20T12:00:00.000Z";
const post = (id: string, text: string, at = "2026-09-20T10:00:00.000Z", author = `a${id}`): CryptoBriefSourcePost => ({ id, author_id: author, handle: `user${author}`, text, url: `https://x.com/user${author}/status/${id}`, posted_at: at, first_seen_at: "2026-09-20T10:05:00.000Z", kind: "original" });

test("every current and future registry entry receives the reusable default config", () => {
  assert.equal(CRYPTO_BRIEF_COINS.length, COINS.length);
  for (const coin of COINS) {
    const config = cryptoBriefConfig(coin.symbol);
    assert.equal(config.coin, coin.symbol);
    assert.equal(config.modules.savedPosts, true);
    assert.ok(config.maxTakeaways <= 3);
  }
});

test("resolver accepts symbol, name, and exact case-sensitive contract without guessing", () => {
  assert.equal(resolveCryptoBriefCoin("$backpack").status, "resolved");
  assert.equal(resolveCryptoBriefCoin("Backpack").status, "resolved");
  assert.equal(resolveCryptoBriefCoin(coinConfig("BACKPACK").address!).status, "resolved");
  assert.equal(resolveCryptoBriefCoin(coinConfig("BACKPACK").address!.toLowerCase()).status, "untracked");
  assert.equal(resolveCryptoBriefCoin("not-a-coin").status, "untracked");
});

test("Backpack rejects luggage, keeps provenance, and applies the contract anchor", () => {
  const address = coinConfig("BACKPACK").address!;
  const brief = buildCryptoBrief({
    coin: "BACKPACK", window: "24h", asOf: NOW, archiveStatus: "available", contractAnchorAt: "2026-09-20T09:00:00.000Z",
    posts: [
      post("10000000001", "My travel backpack has a broken zipper."),
      post("10000000002", "$BACKPACK token volume is rising."),
      post("10000000003", `Contract ${address} launched on Solana.`),
      post("10000000004", "$BACKPACK token before anchor.", "2026-09-20T08:00:00.000Z"),
    ],
  });
  assert.equal(brief.coverage.matchingPostCount, 2);
  assert.deepEqual(brief.evidence.map((item) => item.match.method).sort(), ["cashtag", "contract"]);
  assert.ok(brief.evidence.every((item) => item.id.startsWith("x:")));
});

test("brief deduplicates tweet IDs, avoids three takeaways from one author, and reports truncation", () => {
  const posts = [post("10000000001", "$ZEC ships update one", undefined, "same"), post("10000000001", "$ZEC ships update one", undefined, "same"), post("10000000002", "$ZEC ships update two", undefined, "same"), post("10000000003", "$ZEC ships update three", undefined, "other")];
  const brief = buildCryptoBrief({ coin: "ZEC", window: "24h", asOf: NOW, archiveStatus: "partial", posts, coverageReasons: ["sample"] });
  assert.equal(brief.evidence.length, 3);
  assert.equal(brief.takeaways.length, 3);
  assert.equal(brief.takeaways[1].evidenceIds[0], "x:10000000003", "the first pass prefers another author");
  assert.equal(brief.coverage.status, "partial");
  assert.deepEqual(brief.coverage.reasons, ["sample"]);
});
