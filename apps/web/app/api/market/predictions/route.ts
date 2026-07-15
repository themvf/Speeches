import { createRequestId, ok } from "@/lib/server/api-utils";
import type {
  MarketPredictionsData,
  PredictionArchetype,
  PredictionCalendarRow,
  PredictionClosedMarket,
  PredictionWallet,
} from "@/lib/server/types";
import snapshot from "@/lib/server/prediction-markets-data.json" with { type: "json" };

export const dynamic = "force-static";

// Static pilot snapshot (SEC-25/SEC-28) - statically imported so this route
// has zero network/DB dependency until the live SEC-26/SEC-27 pipeline lands.
// isLive: false is load-bearing for the UI's "static snapshot" labeling;
// don't flip it without wiring a live data source.
export async function GET() {
  const requestId = createRequestId();

  const calendar: PredictionCalendarRow[] = snapshot.calendar.map((row) => ({
    conditionId: row.conditionId,
    ticker: row.ticker,
    question: row.question,
    reportDate: row.reportDate,
    eps: row.eps,
    impliedProbYes: row.impliedProbYes,
    volume: row.volume,
    consensus: {
      yes: row.consensus.yes,
      no: row.consensus.no,
      wallets: row.consensus.wallets.map((w) => ({
        name: w.name,
        wallet: w.wallet,
        archetype: w.archetype as PredictionArchetype,
        side: w.side,
        shares: w.shares,
      })),
    },
  }));

  const closed: PredictionClosedMarket[] = snapshot.closed.map((row) => ({
    conditionId: row.conditionId,
    ticker: row.ticker,
    question: row.question,
    resolvedDate: row.resolvedDate,
    outcome: row.outcome === "beat" ? "beat" : "miss",
    volume: row.volume,
    sharpCohort: {
      correct: row.sharpCohort.correct,
      total: row.sharpCohort.total,
      wallets: row.sharpCohort.wallets.map((w) => ({
        name: w.name,
        wallet: w.wallet,
        archetype: w.archetype as PredictionArchetype,
        pnlUsd: w.pnlUsd,
        correct: w.correct,
      })),
    },
  }));

  const wallets: PredictionWallet[] = snapshot.wallets.map((w) => ({
    wallet: w.wallet,
    name: w.name,
    archetype: w.archetype as PredictionArchetype,
    markets: w.markets,
    wins: w.wins,
    winRate: w.winRate,
    pnlUsd: w.pnlUsd,
    roi: w.roi,
    avgWinnerEntry: w.avgWinnerEntry,
    openPositions: w.openPositions,
  }));

  const data: MarketPredictionsData = {
    isLive: false,
    snapshotDate: snapshot.generatedAt,
    source: snapshot.source,
    archMinMarkets: snapshot.archMinMarkets,
    calendar,
    closed,
    wallets,
    warning:
      "Static pilot snapshot (SEC-25) - not a live feed. Live daily ingestion + scoring is scoped in JIRA SEC-26/SEC-27.",
  };
  return ok(data, requestId);
}
