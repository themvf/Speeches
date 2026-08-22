import assert from "node:assert/strict";

const round3 = (v: number) => Math.round(v * 1000) / 1000;
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

test("uses the generalist row as the macro record rather than only the visible cohorts", () => {
  // Semantics changed deliberately: the macro stats query ranks WITHIN each
  // cohort, so smaller cohorts can be missing from the payload entirely. The
  // generalist row spans all of them, so summing only the visible cohorts
  // undercounts a wallet's real history (observed live: 7 vs 27). It is used
  // wholesale for the macro portion - never added on top, which would
  // double-count.
  const rows = [
    macro({ wallet: "0xmacro", name: "Macro Only", cohort: "fed_decision", events: 10, wins: 6, pnl: 200 }),
    macro({ wallet: "0xmacro", name: "Macro Only", cohort: "macro_generalist", events: 20, wins: 13, pnl: 900 }),
  ];
  const [wallet] = mergeWalletIntelligence([], rows, 8);
  assert.equal(wallet.name, "Macro Only");
  assert.equal(wallet.markets, 20, "the full macro record, not just the visible cohort");
  assert.equal(wallet.pnlUsd, 900, "and its P&L, so the two cannot disagree");
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

test("edge measures the win rate against the price actually paid", () => {
  // Buying at 0.90 and winning 90% of the time is zero edge - the win rate was
  // already priced in. This is the case a raw win-rate leaderboard gets wrong.
  const [chalk] = mergeWalletIntelligence([], [macro({
    wallet: "0xchalk", events: 100, wins: 90,
    recent_events: 10, watchlist_status: "none", entry_avg: 0.90,
  })], 8);
  assert.equal(chalk.trajectory?.edge, 0);

  // Winning only 40% but buying at 0.15 is a large edge.
  const [tail] = mergeWalletIntelligence([], [macro({
    wallet: "0xtail", events: 100, wins: 40,
    recent_events: 10, watchlist_status: "none", entry_avg: 0.15,
  })], 8);
  assert.equal(tail.trajectory?.edge, 0.25);

  // Paying MORE than you win is negative edge, however high the win rate.
  const [overpay] = mergeWalletIntelligence([], [macro({
    wallet: "0xover", events: 100, wins: 80,
    recent_events: 10, watchlist_status: "none", entry_avg: 0.95,
  })], 8);
  assert.ok((overpay.trajectory?.edge ?? 0) < 0);
});

test("edge is null rather than guessed when entry price is not yet available", () => {
  const [w] = mergeWalletIntelligence([], [macro({ recent_events: 5, watchlist_status: "none" })], 8);
  assert.equal(w.trajectory?.edge, null);
  assert.equal(w.trajectory?.entryAvg, null);
});

test("a single lucky longshot does not produce an edge figure", () => {
  // One market, bought at 1c, won. The arithmetic says +99 points, which would
  // outrank every genuine trader; the sample says nothing at all.
  const [lucky] = mergeWalletIntelligence([], [macro({
    wallet: "0xlucky", events: 1, wins: 1,
    recent_events: 1, watchlist_status: "none", entry_avg: 0.01,
  })], 8);
  assert.equal(lucky.trajectory?.edge, null, "no edge below the sample floor");
  assert.equal(lucky.trajectory?.entryAvg, 0.01, "the price paid is still reported");
});

test("sample size counts cohorts that fell outside the per-cohort ranking limit", () => {
  // The stats query ranks within each cohort, so a wallet's smaller cohorts
  // may never reach the payload. The generalist row still spans all of them,
  // so the visible specialties alone can badly undercount the real history -
  // and the edge figure is computed from the generalist, so the two must agree.
  const rows = [
    macro({ wallet: "0xw", cohort: "core_cpi", events: 2, wins: 2 }),
    macro({ wallet: "0xw", cohort: "ppi", events: 2, wins: 1 }),
    macro({ wallet: "0xw", cohort: "macro_generalist", events: 27, wins: 20 }),
  ];
  const [w] = mergeWalletIntelligence([], rows, 8);
  assert.equal(w.markets, 27, "not 4 - the unseen cohorts are counted back in");
  assert.equal(w.wins, 20);
  assert.equal(w.winRate, round3(20 / 27));
});

test("direct timing evidence overrides the entry-price proxy", () => {
  // Observed live on prediction1997: badged "Early sharp" from its earnings
  // record while its macro record showed 9% pre-release share and a qualified
  // "Release scalper" label. Both were shown, asserting contradictory things
  // about one trader. Pre-release share MEASURES when capital moves relative
  // to public information; earnings has no such signal and infers it from
  // entry price, so the measurement wins.
  const [wallet] = mergeWalletIntelligence(
    [{ ...earnings, wallet: "0xboth", archetype: "early_sharp" }],
    [macro({ wallet: "0xboth", cohort: "macro_generalist", archetype: "release_scalper",
             events: 25, wins: 24, predictive_cost: 9, timing_cost: 100 })],
    8);
  assert.equal(wallet.archetype, "news_scalper",
    "the earnings-family name for the behaviour macro measured directly");
});

test("a wallet with no contradicting macro record keeps its earnings verdict", () => {
  const [wallet] = mergeWalletIntelligence(
    [{ ...earnings, wallet: "0xclean", archetype: "early_sharp" }],
    [macro({ wallet: "0xclean", cohort: "fed_decision", archetype: "early_sharp", events: 12 })],
    8);
  assert.equal(wallet.archetype, "early_sharp");
});

test("the edge figure comes from the same family as the badge", () => {
  // The Edge column was reading a macro row's entry price while the badge came
  // from earnings - a number describing a different specialty than its label.
  const [wallet] = mergeWalletIntelligence(
    [{ ...earnings, wallet: "0xmix", archetype: "early_sharp" }],
    [macro({ wallet: "0xmix", cohort: "macro_generalist", events: 25, wins: 12,
             recent_events: 10, watchlist_status: "none", entry_avg: 0.95 })],
    8);
  assert.notEqual(wallet.trajectory?.entryAvg, 0.95,
    "must not take its entry price from the macro row");
});
