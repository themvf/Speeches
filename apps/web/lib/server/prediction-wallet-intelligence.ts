import type {
  MacroSharpArchetype,
  MacroSharpCohort,
  PredictionArchetype,
  PredictionWallet,
  PredictionWalletSpecialty,
} from "./types";

export type BasePredictionWallet = Omit<PredictionWallet, "specialties" | "qualifiedSpecialties">;

export interface MacroWalletStatInput {
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
