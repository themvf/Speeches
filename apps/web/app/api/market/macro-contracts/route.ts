import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import { fetchPolymarketMacroPredictions } from "@/lib/server/polymarket-macro";
import { getPolymarketMacroWalletStats } from "@/lib/server/neon";
import type { MacroSharpCohort, MacroSharpCohortSummary, MacroSharpWallet, MacroWalletTrackingData } from "@/lib/server/types";

export const runtime = "nodejs";
export const revalidate = 300;

const COHORTS: Array<{ id: MacroSharpCohort; label: string; cadence: string }> = [
  { id: "fed_decision", label: "Fed Sharp", cadence: "~8/year" },
  { id: "nonfarm_payrolls", label: "Payrolls Sharp", cadence: "monthly" },
  { id: "unemployment", label: "Unemployment Sharp", cadence: "monthly" },
  { id: "headline_cpi", label: "Headline CPI Sharp", cadence: "monthly" },
  { id: "core_cpi", label: "Core CPI Sharp", cadence: "monthly" },
  { id: "us_gdp", label: "GDP Sharp", cadence: "quarterly" },
  { id: "macro_generalist", label: "Macro Generalist", cadence: "cross-cohort" },
];

async function walletTracking(): Promise<MacroWalletTrackingData> {
  try {
    const rows = await getPolymarketMacroWalletStats();
    const wallets: MacroSharpWallet[] = rows.map((row) => ({
      wallet: row.wallet,
      name: row.name || `${row.wallet.slice(0, 8)}…`,
      cohort: row.cohort as MacroSharpCohort,
      cohortLabel: COHORTS.find((cohort) => cohort.id === row.cohort)?.label ?? row.cohort,
      archetype: row.archetype as MacroSharpWallet["archetype"],
      events: row.events,
      wins: row.wins,
      winRate: row.events > 0 ? row.wins / row.events : 0,
      pnlUsd: row.pnl,
      roi: row.cost > 0 ? row.pnl / row.cost : null,
      predictiveShare: row.timing_cost > 0 ? row.predictive_cost / row.timing_cost : null,
      timingCoverage: row.cost > 0 ? row.timing_cost / row.cost : null,
      avgWinnerEntry: row.win_entry_avg,
    }));
    const cohorts = COHORTS.filter((cohort) => cohort.id !== "macro_generalist").map((cohort) => {
      const members = wallets.filter((wallet) => wallet.cohort === cohort.id);
      return {
        ...cohort,
        id: cohort.id as MacroSharpCohortSummary["id"],
        qualifiedWallets: members.filter((wallet) => wallet.archetype !== "unclassified").length,
        observations: members.reduce((total, wallet) => Math.max(total, wallet.events), 0),
      };
    });
    return {
      isLive: rows.length > 0, minCohortEvents: 10, generalistMinEvents: 20,
      generalistMinCohorts: 3, cohorts, wallets,
      ...(rows.length ? {} : { warning: "Macro wallet history is initializing; run the backfill workflow to seed resolved releases." }),
    };
  } catch {
    return {
      isLive: false, minCohortEvents: 10, generalistMinEvents: 20, generalistMinCohorts: 3,
      cohorts: COHORTS.filter((cohort) => cohort.id !== "macro_generalist").map((cohort) => ({
        ...cohort, id: cohort.id as MacroSharpCohortSummary["id"], qualifiedWallets: 0, observations: 0,
      })),
      wallets: [], warning: "Macro wallet history is initializing; live contracts remain available.",
    };
  }
}

export async function GET() {
  const requestId = createRequestId();
  try {
    const [predictions, tracking] = await Promise.all([fetchPolymarketMacroPredictions(), walletTracking()]);
    return ok({ ...predictions, walletTracking: tracking }, requestId);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to load Polymarket macro contracts.";
    return fail(message, "POLYMARKET_UPSTREAM_ERROR", 502, requestId);
  }
}
