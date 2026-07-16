import assert from "node:assert/strict";
import test from "node:test";

import { mergeWalletIntelligence, type BasePredictionWallet, type MacroWalletStatInput } from "./prediction-wallet-intelligence.ts";

const earnings: BasePredictionWallet = {
  wallet: "0xABC",
  name: "Earnings Ace",
  archetype: "early_sharp",
  markets: 12,
  wins: 8,
  winRate: 8 / 12,
  pnlUsd: 500,
  roi: 0.5,
  avgWinnerEntry: 0.42,
  openPositions: [],
};

function macro(overrides: Partial<MacroWalletStatInput> = {}): MacroWalletStatInput {
  return {
    wallet: "0xabc",
    cohort: "headline_cpi",
    name: "Macro Ace",
    events: 10,
    wins: 7,
    pnl: 300,
    cost: 600,
    predictive_cost: 450,
    timing_cost: 500,
    win_entry_avg: 0.38,
    archetype: "early_sharp",
    ...overrides,
  };
}

test("merges wallet identity case-insensitively and retains specialty evidence", () => {
  const [wallet] = mergeWalletIntelligence([earnings], [macro()], 8);
  assert.equal(wallet.wallet, "0xABC");
  assert.equal(wallet.specialties.length, 2);
  assert.equal(wallet.qualifiedSpecialties, 2);
  assert.equal(wallet.markets, 22);
  assert.equal(wallet.wins, 15);
  assert.equal(wallet.pnlUsd, 800);
  assert.deepEqual(wallet.specialties.map((item) => item.label).sort(), ["Earnings", "Headline CPI"]);
});

test("adds macro-only wallets and excludes generalist aggregates from totals", () => {
  const rows = [
    macro({ wallet: "0xmacro", name: "Macro Only", cohort: "fed_decision", events: 10, wins: 6, pnl: 200 }),
    macro({ wallet: "0xmacro", name: "Macro Only", cohort: "macro_generalist", events: 20, wins: 13, pnl: 900 }),
  ];
  const [wallet] = mergeWalletIntelligence([], rows, 8);
  assert.equal(wallet.name, "Macro Only");
  assert.equal(wallet.markets, 10);
  assert.equal(wallet.pnlUsd, 200);
  assert.equal(wallet.qualifiedSpecialties, 2);
  assert.equal(wallet.archetype, "early_sharp");
});

test("keeps sparse histories visible without presenting them as proven specialties", () => {
  const [wallet] = mergeWalletIntelligence([], [macro({ archetype: "unclassified", events: 4, wins: 4 })], 8);
  assert.equal(wallet.qualifiedSpecialties, 0);
  assert.equal(wallet.archetype, "unclassified");
  assert.equal(wallet.specialties[0].classLabel, "Building sample");
});

test("orders proven multi-specialists before higher-PnL single-specialists", () => {
  const rows = [
    macro({ wallet: "0xmulti", cohort: "headline_cpi", pnl: 100 }),
    macro({ wallet: "0xmulti", cohort: "fed_decision", pnl: 100 }),
    macro({ wallet: "0xsingle", cohort: "us_gdp", pnl: 2000 }),
  ];
  const wallets = mergeWalletIntelligence([], rows, 8);
  assert.equal(wallets[0].wallet, "0xmulti");
});
