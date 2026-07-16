import { createRequestId, ok } from "@/lib/server/api-utils";
import type {
  MarketPredictionsData,
  PredictionArchetype,
  PredictionCalendarRow,
  PredictionClosedMarket,
  PredictionConsensusWallet,
  PredictionWalletPosition,
} from "@/lib/server/types";
import {
  getPolymarketClosedMarkets,
  getPolymarketMacroWalletStats,
  getPolymarketOpenMarkets,
  getPolymarketOpenPositionsForWallets,
  getPolymarketSharpResults,
  getPolymarketWalletStats,
  type PolymarketOpenPositionRow,
  type PolymarketWalletStatsRow,
} from "@/lib/server/neon";
import {
  mergeWalletIntelligence,
  type BasePredictionWallet,
  type MacroWalletStatInput,
} from "@/lib/server/prediction-wallet-intelligence";
import snapshot from "@/lib/server/prediction-markets-data.json" with { type: "json" };

export const runtime = "nodejs";
// Live-first (SEC-26/27): reads the Neon tables the 3x-daily Python sync
// maintains, falling back to the committed static snapshot whenever those
// tables are missing/empty (pre-backfill deploy, DB outage) - the tab can
// never break. isLive tells the UI which source served the request.
export const dynamic = "force-dynamic";

const WALLET_LIMIT = 60;
const CLOSED_LIMIT = 50;
const ARCH_MIN_MARKETS = 8;
const MIN_SHARES = 0.5; // dust floor for a "position"

const SHARP_ARCHETYPES: ReadonlySet<string> = new Set(["early_sharp", "longshot"]);

type BaseMarketPredictionsData = Omit<MarketPredictionsData, "wallets"> & { wallets: BasePredictionWallet[] };

function dominantSide(position: PolymarketOpenPositionRow): { side: string; shares: number } | null {
  const yes = position.net_yes;
  const no = position.net_no;
  if (yes <= MIN_SHARES && no <= MIN_SHARES) return null;
  if (yes >= no) return { side: "Yes", shares: Math.round(yes * 10) / 10 };
  return { side: "No", shares: Math.round(no * 10) / 10 };
}

function walletLabel(row: PolymarketWalletStatsRow): string {
  return row.name || `${row.wallet.slice(0, 8)}…`;
}

async function buildLive(): Promise<BaseMarketPredictionsData> {
  const [openMarkets, closedMarkets, stats] = await Promise.all([
    getPolymarketOpenMarkets(),
    getPolymarketClosedMarkets(CLOSED_LIMIT),
    getPolymarketWalletStats(WALLET_LIMIT),
  ]);
  if (stats.length === 0) {
    throw new Error("wallet stats empty - backfill has not run yet");
  }

  const [positions, sharpResults] = await Promise.all([
    getPolymarketOpenPositionsForWallets(stats.map((s) => s.wallet)),
    getPolymarketSharpResults(closedMarkets.map((m) => m.condition_id)),
  ]);
  const statsByWallet = new Map(stats.map((s) => [s.wallet, s]));

  // Positions keyed both ways: per market (consensus) and per wallet (drawer).
  const consensusByMarket = new Map<string, PredictionConsensusWallet[]>();
  const positionsByWallet = new Map<string, PredictionWalletPosition[]>();
  const marketById = new Map(openMarkets.map((m) => [m.condition_id, m]));
  for (const position of positions) {
    const stat = statsByWallet.get(position.wallet);
    const market = marketById.get(position.condition_id);
    const reduced = dominantSide(position);
    if (!stat || !market || !reduced) continue;
    const walletPositions = positionsByWallet.get(position.wallet) ?? [];
    walletPositions.push({ ticker: market.ticker, question: market.question, side: reduced.side, shares: reduced.shares });
    positionsByWallet.set(position.wallet, walletPositions);
    if (!SHARP_ARCHETYPES.has(stat.archetype)) continue; // consensus counts sharps only
    const cohort = consensusByMarket.get(position.condition_id) ?? [];
    cohort.push({
      name: walletLabel(stat),
      wallet: stat.wallet,
      archetype: stat.archetype as PredictionArchetype,
      side: reduced.side,
      shares: reduced.shares,
    });
    consensusByMarket.set(position.condition_id, cohort);
  }

  const calendar: PredictionCalendarRow[] = openMarkets.map((market) => {
    const cohort = (consensusByMarket.get(market.condition_id) ?? []).sort((a, b) => b.shares - a.shares);
    return {
      conditionId: market.condition_id,
      ticker: market.ticker,
      question: market.question,
      reportDate: market.report_date,
      eps: market.eps,
      impliedProbYes: market.implied_prob_yes,
      volume: market.volume,
      consensus: {
        yes: cohort.filter((w) => w.side === "Yes").length,
        no: cohort.filter((w) => w.side === "No").length,
        wallets: cohort,
      },
    };
  });

  const resultsByMarket = new Map<string, typeof sharpResults>();
  for (const result of sharpResults) {
    const list = resultsByMarket.get(result.condition_id) ?? [];
    list.push(result);
    resultsByMarket.set(result.condition_id, list);
  }
  const closed: PredictionClosedMarket[] = closedMarkets.map((market) => {
    const cohort = resultsByMarket.get(market.condition_id) ?? [];
    return {
      conditionId: market.condition_id,
      ticker: market.ticker,
      question: market.question,
      resolvedDate: market.resolved_date,
      outcome: market.winner === "Yes" ? "beat" : "miss",
      volume: market.volume,
      sharpCohort: {
        correct: cohort.filter((w) => w.correct).length,
        total: cohort.length,
        wallets: cohort.map((w) => ({
          name: w.name || `${w.wallet.slice(0, 8)}…`,
          wallet: w.wallet,
          archetype: w.archetype as PredictionArchetype,
          pnlUsd: Math.round(w.pnl * 100) / 100,
          correct: w.correct,
        })),
      },
    };
  });

  const wallets: BasePredictionWallet[] = stats.map((s) => ({
    wallet: s.wallet,
    name: walletLabel(s),
    archetype: s.archetype as PredictionArchetype,
    markets: s.markets,
    wins: s.wins,
    winRate: s.markets > 0 ? Math.round((s.wins / s.markets) * 1000) / 1000 : 0,
    pnlUsd: Math.round(s.pnl * 100) / 100,
    roi: s.cost > 0 ? Math.round((s.pnl / s.cost) * 1000) / 1000 : null,
    avgWinnerEntry: s.win_entry_avg,
    openPositions: positionsByWallet.get(s.wallet) ?? [],
  }));

  return {
    isLive: true,
    snapshotDate: new Date().toISOString(),
    source: "Polymarket public data API - live Neon pipeline (SEC-26/27), synced 3x daily",
    archMinMarkets: ARCH_MIN_MARKETS,
    calendar,
    closed,
    wallets,
  };
}

function buildFromSnapshot(reason: string): BaseMarketPredictionsData {
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

  const wallets: BasePredictionWallet[] = snapshot.wallets.map((w) => ({
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

  return {
    isLive: false,
    snapshotDate: snapshot.generatedAt,
    source: snapshot.source,
    archMinMarkets: snapshot.archMinMarkets,
    calendar,
    closed,
    wallets,
    warning: `Serving the static pilot snapshot (${reason}). Live 3x-daily sync is JIRA SEC-26/SEC-27.`,
  };
}

export async function GET() {
  const requestId = createRequestId();
  const macroRowsPromise = getPolymarketMacroWalletStats(300)
    .catch((error) => {
      console.warn("[market/predictions] macro wallet intelligence unavailable:", error);
      return [] as MacroWalletStatInput[];
    });
  try {
    const [data, macroRows] = await Promise.all([buildLive(), macroRowsPromise]);
    return ok({ ...data, wallets: mergeWalletIntelligence(data.wallets, macroRows, ARCH_MIN_MARKETS) }, requestId);
  } catch (err) {
    // Missing tables (sync not deployed/backfilled yet), empty stats, or any
    // DB failure -> static snapshot, visibly labeled. Never a 500.
    const reason = err instanceof Error ? err.message : "live data unavailable";
    console.warn("[market/predictions] falling back to static snapshot:", reason);
    const macroRows = await macroRowsPromise;
    const data = buildFromSnapshot(reason);
    return ok({ ...data, wallets: mergeWalletIntelligence(data.wallets, macroRows, ARCH_MIN_MARKETS) }, requestId);
  }
}
