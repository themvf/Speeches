import { createRequestId, ok } from "@/lib/server/api-utils";
import type {
  EarningsWeekCompany,
  MarketEarningsWeekData,
  PredictionArchetype,
  PredictionConsensusWallet,
} from "@/lib/server/types";
import {
  getDailyStockAttention,
  getLatestStockAttentionDate,
  getPolymarketOpenMarkets,
  getPolymarketOpenPositionsForWallets,
  getPolymarketWalletStats,
} from "@/lib/server/neon";
import kpiSnapshot from "@/lib/server/kpi-pilot-data.json" with { type: "json" };
import predictionsSnapshot from "@/lib/server/prediction-markets-data.json" with { type: "json" };

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// SEC-52: reporting window = yesterday through +10 days, so a card survives
// the morning after a print (when the result is what you want to see).
const WINDOW_BACK_DAYS = 1;
const WINDOW_FORWARD_DAYS = 10;
const WALLET_LIMIT = 60;
const MIN_SHARES = 0.5;
const KPI_QUARTERS = 8;
const SHARP_ARCHETYPES: ReadonlySet<string> = new Set(["early_sharp", "longshot"]);

function windowBounds(): { start: string; end: string } {
  const day = (offset: number) => {
    const d = new Date();
    d.setUTCDate(d.getUTCDate() + offset);
    return d.toISOString().slice(0, 10);
  };
  return { start: day(-WINDOW_BACK_DAYS), end: day(WINDOW_FORWARD_DAYS) };
}

// The CBOE KPI snapshot's headline series for a ticker: prefer diluted EPS,
// else the first KPI it has (banks/holdings vary).
function kpiTrend(ticker: string): { label: string | null; series: { end: string; value: number }[] } {
  const company = (kpiSnapshot.companies as Record<string, { kpis: Record<string, { label: string; series: { end: string; value: number }[] }> }>)[ticker];
  if (!company) return { label: null, series: [] };
  const kpi = company.kpis["eps_diluted"] ?? Object.values(company.kpis)[0];
  if (!kpi) return { label: null, series: [] };
  return { label: kpi.label, series: kpi.series.slice(-KPI_QUARTERS).map((p) => ({ end: p.end, value: p.value })) };
}

type CalendarSeed = {
  ticker: string;
  question: string;
  conditionId: string;
  reportDate: string;
  eps: string | null;
  impliedProbYes: number | null;
  volume: number;
  consensus?: { yes: number; no: number; wallets: PredictionConsensusWallet[] };
};

async function liveSeeds(start: string, end: string): Promise<CalendarSeed[]> {
  const markets = (await getPolymarketOpenMarkets()).filter(
    (m) => m.ticker && m.report_date && m.report_date >= start && m.report_date <= end
  );
  if (markets.length === 0) return [];

  // Sharp consensus, same rules as the Prediction Markets tab: early sharps
  // + longshots only.
  const stats = await getPolymarketWalletStats(WALLET_LIMIT);
  const sharp = stats.filter((s) => SHARP_ARCHETYPES.has(s.archetype));
  const positions = sharp.length
    ? await getPolymarketOpenPositionsForWallets(sharp.map((s) => s.wallet))
    : [];
  const statsByWallet = new Map(sharp.map((s) => [s.wallet, s]));
  const consensusByMarket = new Map<string, PredictionConsensusWallet[]>();
  for (const position of positions) {
    const stat = statsByWallet.get(position.wallet);
    if (!stat) continue;
    const yes = position.net_yes;
    const no = position.net_no;
    if (yes <= MIN_SHARES && no <= MIN_SHARES) continue;
    const side = yes >= no ? "Yes" : "No";
    const shares = Math.round(Math.max(yes, no) * 10) / 10;
    const cohort = consensusByMarket.get(position.condition_id) ?? [];
    cohort.push({
      name: stat.name || `${stat.wallet.slice(0, 8)}…`,
      wallet: stat.wallet,
      archetype: stat.archetype as PredictionArchetype,
      side,
      shares,
    });
    consensusByMarket.set(position.condition_id, cohort);
  }

  return markets.map((m) => {
    const cohort = (consensusByMarket.get(m.condition_id) ?? []).sort((a, b) => b.shares - a.shares);
    return {
      ticker: m.ticker,
      question: m.question,
      conditionId: m.condition_id,
      reportDate: m.report_date!,
      eps: m.eps,
      impliedProbYes: m.implied_prob_yes,
      volume: m.volume,
      consensus: {
        yes: cohort.filter((w) => w.side === "Yes").length,
        no: cohort.filter((w) => w.side === "No").length,
        wallets: cohort,
      },
    };
  });
}

function snapshotSeeds(start: string, end: string): CalendarSeed[] {
  return predictionsSnapshot.calendar
    .filter((row) => row.ticker && row.reportDate && row.reportDate >= start && row.reportDate <= end)
    .map((row) => ({
      ticker: row.ticker,
      question: row.question,
      conditionId: row.conditionId,
      reportDate: row.reportDate!,
      eps: row.eps,
      impliedProbYes: row.impliedProbYes,
      volume: row.volume,
      consensus: {
        yes: row.consensus.yes,
        no: row.consensus.no,
        wallets: row.consensus.wallets.map((w) => ({
          name: w.name, wallet: w.wallet, archetype: w.archetype as PredictionArchetype,
          side: w.side, shares: w.shares,
        })),
      },
    }));
}

export async function GET() {
  const requestId = createRequestId();
  const { start, end } = windowBounds();

  let seeds: CalendarSeed[] = [];
  let isLive = true;
  let warning: string | undefined;
  try {
    seeds = await liveSeeds(start, end);
    if (seeds.length === 0) {
      // Distinguish "quiet week" from "tables empty pre-backfill": if the
      // snapshot has window entries but live doesn't, fall back.
      const fallback = snapshotSeeds(start, end);
      if (fallback.length > 0) {
        seeds = fallback;
        isLive = false;
        warning = "Live earnings markets unavailable - serving the committed snapshot.";
      }
    }
  } catch {
    seeds = snapshotSeeds(start, end);
    isLive = false;
    warning = "Live earnings markets unavailable (DB unreachable) - serving the committed snapshot.";
  }

  // Attention join (fail-soft): latest day + the day before for a delta.
  const mentions = new Map<string, number>();
  const mentionsPrev = new Map<string, number>();
  try {
    const date = await getLatestStockAttentionDate();
    if (date) {
      for (const row of await getDailyStockAttention(date, 500)) mentions.set(row.ticker, row.total_mention_count);
      const prev = new Date(date + "T00:00:00Z");
      prev.setUTCDate(prev.getUTCDate() - 1);
      for (const row of await getDailyStockAttention(prev.toISOString().slice(0, 10), 500)) {
        mentionsPrev.set(row.ticker, row.total_mention_count);
      }
    }
  } catch {
    // cards render without the attention column
  }

  const companies: EarningsWeekCompany[] = seeds
    .map((seed) => {
      const trend = kpiTrend(seed.ticker);
      return {
        ticker: seed.ticker,
        question: seed.question,
        reportDate: seed.reportDate,
        eps: seed.eps,
        impliedProbYes: seed.impliedProbYes,
        volume: seed.volume,
        consensus: seed.consensus ?? { yes: 0, no: 0, wallets: [] },
        kpiLabel: trend.label,
        kpiSeries: trend.series,
        mentions: mentions.has(seed.ticker) ? mentions.get(seed.ticker)! : null,
        mentionsPrev: mentionsPrev.has(seed.ticker) ? mentionsPrev.get(seed.ticker)! : null,
      };
    })
    .sort((a, b) => (a.reportDate === b.reportDate ? b.volume - a.volume : a.reportDate < b.reportDate ? -1 : 1));

  const data: MarketEarningsWeekData = {
    isLive,
    windowStart: start,
    windowEnd: end,
    companies,
    ...(warning ? { warning } : {}),
    generatedAt: new Date().toISOString(),
  };
  return ok(data, requestId);
}
