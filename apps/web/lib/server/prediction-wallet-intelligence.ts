import type {
  MacroSharpArchetype,
  WalletTrajectory,
  MacroSharpCohort,
  PredictionArchetype,
  PredictionWallet,
  PredictionWalletSpecialty,
} from "./types";

export type BasePredictionWallet = Omit<PredictionWallet, "specialties" | "qualifiedSpecialties">;

/** Raw trajectory columns as they arrive from either stats table. All
 *  optional: they are Python-owned and absent for one deploy cycle. */
export interface WalletTrajectoryInput {
  recent_events?: number;
  recent_wins?: number;
  recent_pnl?: number;
  recent_cost?: number;
  chases_losses?: boolean;
  chase_ratio?: number | null;
  watchlist_status?: string;
}

export function toTrajectory(row: WalletTrajectoryInput): WalletTrajectory | undefined {
  const status = row.watchlist_status;
  if (!status || row.recent_events === undefined) return undefined;
  const events = row.recent_events ?? 0;
  const wins = row.recent_wins ?? 0;
  const cost = row.recent_cost ?? 0;
  return {
    status: (["proven", "developing", "watching", "none"].includes(status)
      ? status : "none") as WalletTrajectory["status"],
    recentEvents: events,
    recentWins: wins,
    recentWinRate: events > 0 ? round(wins / events) : null,
    recentRoi: cost > 0 ? round((row.recent_pnl ?? 0) / cost) : null,
    chasesLosses: Boolean(row.chases_losses),
    chaseRatio: row.chase_ratio ?? null,
  };
}

export interface MacroWalletStatInput extends WalletTrajectoryInput {
  wallet: string;
  cohort: string;
  name: string;
  events: number;
  wins: number;
  pnl: number;
  cost: number;
  predictive_cost: number;
  timing_cost: number;
  win_entry_avg: number | null;
  archetype: string;
}

const COHORT_LABELS: Record<MacroSharpCohort, string> = {
  fed_decision: "Fed",
  nonfarm_payrolls: "Payrolls",
  unemployment: "Unemployment",
  headline_cpi: "Headline CPI",
  core_cpi: "Core CPI",
  us_gdp: "GDP",
  core_pce: "Core PCE",
  ism_manufacturing: "ISM Manufacturing",
  ism_services: "ISM Services",
  ppi: "PPI",
  jolts: "JOLTS",
  macro_generalist: "Macro Generalist",
};

const MACRO_CLASS_LABELS: Record<MacroSharpArchetype, string> = {
  early_sharp: "Early sharp",
  release_scalper: "Release scalper",
  longshot: "Longshot",
  unclassified: "Building sample",
};

const EARNINGS_CLASS_LABELS: Record<PredictionArchetype, string> = {
  early_sharp: "Early sharp",
  news_scalper: "News scalper",
  longshot: "Longshot",
  unclassified: "Building sample",
};

function key(wallet: string): string {
  return wallet.trim().toLowerCase();
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}

// Mirrors cohort_min_events()/COHORT_MIN_EVENTS_OVERRIDES in
// polymarket_macro_sync.py and MIN_EVENTS_OVERRIDES in the macro-contracts
// route: 10 for ~monthly-or-better cadence, 5 for quarterly us_gdp.
const MACRO_DEFAULT_MIN_EVENTS = 10;
const MACRO_MIN_EVENTS: Partial<Record<MacroSharpCohort, number>> = { us_gdp: 5 };

function macroMinEvents(cohort: MacroSharpCohort): number {
  return MACRO_MIN_EVENTS[cohort] ?? MACRO_DEFAULT_MIN_EVENTS;
}

function macroArchetype(value: string): MacroSharpArchetype {
  if (value === "early_sharp" || value === "release_scalper" || value === "longshot") return value;
  return "unclassified";
}

function displayName(name: string, wallet: string): string {
  return name.trim() || `${wallet.slice(0, 8)}…`;
}

function earningsSpecialty(wallet: BasePredictionWallet, minMarkets: number): PredictionWalletSpecialty {
  return {
    id: "earnings",
    label: "Earnings",
    family: "earnings",
    qualified: wallet.markets >= minMarkets && wallet.archetype !== "unclassified",
    classLabel: EARNINGS_CLASS_LABELS[wallet.archetype],
    events: wallet.markets,
    minEvents: minMarkets,
    wins: wallet.wins,
    winRate: wallet.winRate,
    pnlUsd: wallet.pnlUsd,
    roi: wallet.roi,
    predictiveShare: null,
    avgWinnerEntry: wallet.avgWinnerEntry,
  };
}

function macroSpecialty(row: MacroWalletStatInput): PredictionWalletSpecialty | null {
  if (!(row.cohort in COHORT_LABELS)) return null;
  const cohort = row.cohort as MacroSharpCohort;
  const archetype = macroArchetype(row.archetype);
  return {
    id: cohort,
    label: COHORT_LABELS[cohort],
    family: "macro",
    qualified: archetype !== "unclassified",
    classLabel: MACRO_CLASS_LABELS[archetype],
    events: row.events,
    minEvents: macroMinEvents(cohort),
    wins: row.wins,
    winRate: row.events > 0 ? round(row.wins / row.events) : 0,
    pnlUsd: round(row.pnl),
    roi: row.cost > 0 ? round(row.pnl / row.cost) : null,
    predictiveShare: row.timing_cost > 0 ? round(row.predictive_cost / row.timing_cost) : null,
    avgWinnerEntry: row.win_entry_avg,
  };
}

function macroOnlyArchetype(specialties: PredictionWalletSpecialty[]): PredictionArchetype {
  const qualified = specialties.filter((item) => item.qualified && item.id !== "macro_generalist");
  if (qualified.some((item) => item.classLabel === "Early sharp")) return "early_sharp";
  if (qualified.some((item) => item.classLabel === "Longshot")) return "longshot";
  if (qualified.some((item) => item.classLabel === "Release scalper")) return "news_scalper";
  return "unclassified";
}

function finalize(wallet: BasePredictionWallet, specialties: PredictionWalletSpecialty[]): PredictionWallet {
  const distinct = specialties.filter((item) => item.id !== "macro_generalist");
  const events = distinct.reduce((total, item) => total + item.events, 0);
  const wins = distinct.reduce((total, item) => total + item.wins, 0);
  const pnl = distinct.reduce((total, item) => total + item.pnlUsd, 0);
  const sorted = [...specialties].sort((left, right) =>
    Number(right.qualified) - Number(left.qualified) || right.pnlUsd - left.pnlUsd || right.events - left.events
  );
  return {
    ...wallet,
    markets: events,
    wins,
    winRate: events > 0 ? round(wins / events) : 0,
    pnlUsd: round(pnl),
    specialties: sorted,
    qualifiedSpecialties: specialties.filter((item) => item.qualified).length,
  };
}

export function mergeWalletIntelligence(
  earningsWallets: BasePredictionWallet[],
  macroRows: MacroWalletStatInput[],
  earningsMinMarkets: number,
): PredictionWallet[] {
  const byWallet = new Map<string, { wallet: BasePredictionWallet; specialties: PredictionWalletSpecialty[]; hasEarnings: boolean }>();

  for (const wallet of earningsWallets) {
    byWallet.set(key(wallet.wallet), { wallet, specialties: [earningsSpecialty(wallet, earningsMinMarkets)], hasEarnings: true });
  }

  for (const row of macroRows) {
    const specialty = macroSpecialty(row);
    if (!specialty) continue;
    const normalized = key(row.wallet);
    const existing = byWallet.get(normalized);
    if (existing) {
      const duplicate = existing.specialties.findIndex((item) => item.id === specialty.id);
      if (duplicate >= 0) existing.specialties[duplicate] = specialty;
      else existing.specialties.push(specialty);
      if ((!existing.wallet.name || existing.wallet.name.startsWith("0x")) && row.name) existing.wallet.name = row.name;
      // macro_generalist spans every cohort, so it is the best cross-cohort
      // read; otherwise take the first macro row that carries one.
      if (!existing.wallet.trajectory || row.cohort === "macro_generalist") {
        existing.wallet.trajectory = toTrajectory(row) ?? existing.wallet.trajectory;
      }
      continue;
    }
    const wallet: BasePredictionWallet = {
      wallet: row.wallet,
      name: displayName(row.name, row.wallet),
      archetype: "unclassified",
      markets: 0,
      wins: 0,
      winRate: 0,
      pnlUsd: 0,
      roi: null,
      avgWinnerEntry: null,
      openPositions: [],
      trajectory: toTrajectory(row),
    };
    byWallet.set(normalized, { wallet, specialties: [specialty], hasEarnings: false });
  }

  return [...byWallet.values()]
    .map(({ wallet, specialties, hasEarnings }) => {
      if (!hasEarnings) {
        wallet.archetype = macroOnlyArchetype(specialties);
        const primary = specialties.find((item) => item.qualified) ?? specialties[0];
        wallet.roi = primary?.roi ?? null;
        wallet.avgWinnerEntry = primary?.avgWinnerEntry ?? null;
      }
      return finalize(wallet, specialties);
    })
    .sort((left, right) =>
      right.qualifiedSpecialties - left.qualifiedSpecialties || right.pnlUsd - left.pnlUsd || right.markets - left.markets
    );
}
