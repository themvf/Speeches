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

test("each specialty carries its own qualifying bar, so a high combined total cannot masquerade as depth", () => {
  // A wallet spread thin across four cohorts: 18 events in total, but the
  // deepest single cohort is 7 against a 10-event bar. The combined figure
  // shown on the row is a sum; qualification is per specialty. Without a
  // per-specialty bar the UI reads as if 18 events should already qualify.
  const rows = [
    macro({ wallet: "0xspread", cohort: "unemployment", events: 7, wins: 7, archetype: "unclassified" }),
    macro({ wallet: "0xspread", cohort: "nonfarm_payrolls", events: 5, wins: 5, archetype: "unclassified" }),
    macro({ wallet: "0xspread", cohort: "us_gdp", events: 5, wins: 5, archetype: "unclassified" }),
    macro({ wallet: "0xspread", cohort: "headline_cpi", events: 1, wins: 1, archetype: "unclassified" }),
  ];
  const [wallet] = mergeWalletIntelligence([], rows, 8);
  assert.equal(wallet.markets, 18);
  assert.equal(wallet.qualifiedSpecialties, 0);

  const bar = (id: string) => wallet.specialties.find((s) => s.id === id)!.minEvents;
  assert.equal(bar("unemployment"), 10, "monthly cohorts use the default 10-event bar");
  // us_gdp is quarterly and gets a lower bar - mirrors
  // COHORT_MIN_EVENTS_OVERRIDES in polymarket_macro_sync.py.
  assert.equal(bar("us_gdp"), 5);
});

test("the earnings specialty carries the earnings gate, not the macro one", () => {
  const [wallet] = mergeWalletIntelligence(
    [{ ...earnings, markets: 261, wins: 149, archetype: "unclassified" }], [], 8);
  const specialty = wallet.specialties.find((s) => s.id === "earnings")!;
  assert.equal(specialty.minEvents, 8);
  // 261 markets is far past the gate, so "Building sample" here is a
  // classifier verdict, not a sample-size one - the two must stay tellable
  // apart.
  assert.ok(specialty.events > specialty.minEvents);
  assert.equal(specialty.qualified, false);
});

test("trajectory is absent rather than fabricated when the Python columns have not shipped yet", () => {
  // Deploy-order: this reader goes live on Vercel instantly, but the
  // trajectory columns only exist after the next Python sync runs its ALTERs.
  // A row without them must yield no trajectory at all - never a default that
  // would render as a real (and wrong) "developing" verdict.
  const [wallet] = mergeWalletIntelligence([], [macro({ archetype: "unclassified" })], 8);
  assert.equal(wallet.trajectory, undefined);
});

test("developing status and loss-chasing survive the merge", () => {
  const [wallet] = mergeWalletIntelligence([], [macro({
    archetype: "unclassified",
    recent_events: 10, recent_wins: 8, recent_pnl: 400, recent_cost: 1000,
    chases_losses: true, chase_ratio: 2.4, watchlist_status: "developing",
  })], 8);
  assert.equal(wallet.trajectory?.status, "developing");
  assert.equal(wallet.trajectory?.recentWinRate, 0.8);
  assert.equal(wallet.trajectory?.recentRoi, 0.4);
  assert.equal(wallet.trajectory?.chasesLosses, true);
  // Still not qualified - the watchlist must never manufacture a verdict.
  assert.equal(wallet.qualifiedSpecialties, 0);
  assert.equal(wallet.archetype, "unclassified");
});

test("an unrecognised status string degrades to none instead of leaking through", () => {
  const [wallet] = mergeWalletIntelligence([], [macro({
    recent_events: 5, watchlist_status: "totally-made-up",
  })], 8);
  assert.equal(wallet.trajectory?.status, "none");
});
