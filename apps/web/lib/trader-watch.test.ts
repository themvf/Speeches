import assert from "node:assert/strict";
import test from "node:test";
import { TRADER_WATCH_PROFILES, walletTokenChanges } from "./trader-watch.ts";

const address = "64w4qRu9VGio7U1Asc6B68QDpS8L1McmSn2yyExC6Fii";

test("candidate wallet remains explicitly unverified", () => {
  assert.equal(TRADER_WATCH_PROFILES[0].wallet?.address, address);
  assert.equal(TRADER_WATCH_PROFILES[0].wallet?.attribution, "unverified");
  assert.equal(TRADER_WATCH_PROFILES[0].wallet?.evidenceUrl, null);
});

test("token changes use only balances owned by the watched address", () => {
  const changes = walletTokenChanges(address,
    [
      { owner: address, mint: "mint-a", uiTokenAmount: { amount: "1500000", decimals: 6 } },
      { owner: "someone-else", mint: "mint-a", uiTokenAmount: { amount: "999999999", decimals: 6 } },
    ],
    [
      { owner: address, mint: "mint-a", uiTokenAmount: { amount: "2500001", decimals: 6 } },
      { owner: "someone-else", mint: "mint-a", uiTokenAmount: { amount: "0", decimals: 6 } },
    ],
  );
  assert.deepEqual(changes, [{ mint: "mint-a", delta: "+1.000001", decimals: 6 }]);
});

test("missing owner metadata is not treated as wallet exposure", () => {
  const changes = walletTokenChanges(address,
    [{ mint: "mint-b", uiTokenAmount: { amount: "400", decimals: 2 } }],
    [{ mint: "mint-b", uiTokenAmount: { amount: "100", decimals: 2 } }],
  );
  assert.deepEqual(changes, []);
});

test("unchanged token balances are omitted", () => {
  const balance = { owner: address, mint: "mint-b", uiTokenAmount: { amount: "400", decimals: 2 } };
  assert.deepEqual(walletTokenChanges(address, [balance], [balance]), []);
});

test("a decrease is reported as a balance change, including for a closed token account", () => {
  const changes = walletTokenChanges(address,
    [{ owner: address, mint: "mint-c", uiTokenAmount: { amount: "1000", decimals: 3 } }],
    [],
  );
  assert.deepEqual(changes, [{ mint: "mint-c", delta: "-1", decimals: 3 }]);
});
