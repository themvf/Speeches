import assert from "node:assert/strict";
import test from "node:test";
import { TRADER_WATCH_PROFILES, labelWalletObservations, walletHoldingsFromAccounts, walletTokenChanges } from "./trader-watch.ts";

const address = "64w4qRu9VGio7U1Asc6B68QDpS8L1McmSn2yyExC6Fii";

test("wallet watch identifies the address and its Pump.fun profile", () => {
  assert.equal(TRADER_WATCH_PROFILES[0].wallet?.address, address);
  assert.equal(TRADER_WATCH_PROFILES[0].wallet?.profileUrl, `https://pump.fun/profile/${address}`);
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
  assert.deepEqual(changes, [{ mint: "mint-a", delta: "+1.000001", decimals: 6, symbol: null, name: null, labelSource: null }]);
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
  assert.deepEqual(changes, [{ mint: "mint-c", delta: "-1", decimals: 3, symbol: null, name: null, labelSource: null }]);
});

test("current holdings combine token accounts by mint and keep exact decimal quantities", () => {
  const account = (owner: string, mint: string, amount: string, decimals: number) => ({
    account: { data: { parsed: { info: { owner, mint, tokenAmount: { amount, decimals } } } } },
  });
  assert.deepEqual(walletHoldingsFromAccounts(address, [
    account(address, "mint-a", "1000001", 6),
    account(address, "mint-a", "2", 6),
    account(address, "mint-b", "0", 9),
    account("someone-else", "mint-c", "9999", 0),
  ]), [{ mint: "mint-a", amount: "1.000003", decimals: 6, symbol: null, name: null, labelSource: null }]);
});

test("recent activity gets a readable token name without losing its mint or direction", () => {
  const change = walletTokenChanges(address, [], [
    { owner: address, mint: "mint-a", uiTokenAmount: { amount: "3000000", decimals: 6 } },
  ])[0];
  const observation = { signature: "signature", timestamp: null, status: "confirmed" as const, tokenChanges: [change], url: "https://explorer.solana.com/tx/signature" };
  const labeled = labelWalletObservations([observation], new Map([["mint-a", { symbol: "COIN", name: "Example Coin" }]]));
  assert.deepEqual(labeled[0].tokenChanges[0], { ...change, symbol: "COIN", name: "Example Coin", labelSource: "dexscreener" });
  assert.equal(labeled[0].tokenChanges[0].delta, "+3");
  assert.equal(labeled[0].tokenChanges[0].mint, "mint-a");
});
