import { NextRequest, NextResponse } from "next/server";
import {
  getAttentionSweepConfig,
  saveAttentionSweepConfig,
  type AttentionSweepConfig,
  type AttentionSweepSubreddit,
} from "@/lib/server/neon";

export const dynamic = "force-dynamic";

// Mirror of neon_feeds.py's DEFAULT_ATTENTION_SWEEP_CONFIG - served when no
// row exists yet so the admin panel always has something concrete to edit.
const DEFAULT_CONFIG: AttentionSweepConfig = {
  subreddits: [
    { name: "wallstreetbets", tier: 1, weight: 1.0, active: true },
    { name: "stocks", tier: 1, weight: 1.0, active: true },
    { name: "investing", tier: 1, weight: 1.0, active: true },
    { name: "StockMarket", tier: 1, weight: 1.0, active: true },
    { name: "options", tier: 1, weight: 1.0, active: true },
    { name: "Daytrading", tier: 1, weight: 1.0, active: true },
    { name: "pennystocks", tier: 2, weight: 0.7, active: true },
    { name: "Shortsqueeze", tier: 2, weight: 0.7, active: true },
    { name: "SqueezePlays", tier: 2, weight: 0.7, active: true },
    { name: "smallstreetbets", tier: 2, weight: 0.7, active: true },
    { name: "ValueInvesting", tier: 2, weight: 0.9, active: true },
    { name: "dividends", tier: 2, weight: 0.9, active: true },
  ],
  bot_blocklist: ["automoderator", "visualmod", "wsbvotebot", "flairhelperbot"],
  symbol_overrides: { force_ambiguous: [], force_unambiguous: [] },
  author_weighting: { low_diversity_share: 0.8, low_diversity_max_tickers: 2, discount: 0.25, min_items: 5 },
};

function sanitizeSymbolList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(
    value
      .map((symbol) => String(symbol ?? "").trim().toUpperCase())
      .filter((symbol) => /^[A-Z.\-]{1,10}$/.test(symbol))
  )];
}

function sanitizeConfig(raw: unknown): AttentionSweepConfig | null {
  if (!raw || typeof raw !== "object") return null;
  const src = raw as Record<string, unknown>;

  const subreddits: AttentionSweepSubreddit[] = [];
  if (!Array.isArray(src.subreddits)) return null;
  for (const entry of src.subreddits) {
    if (!entry || typeof entry !== "object") continue;
    const item = entry as Record<string, unknown>;
    const name = String(item.name ?? "").trim().replace(/^r\//i, "");
    if (!/^[A-Za-z0-9_]{2,30}$/.test(name)) continue;
    const weight = Number(item.weight);
    subreddits.push({
      name,
      tier: Number(item.tier) === 2 ? 2 : 1,
      weight: Number.isFinite(weight) ? Math.min(2, Math.max(0, weight)) : 1.0,
      active: Boolean(item.active),
    });
  }
  if (subreddits.length === 0) return null;

  const blocklist = Array.isArray(src.bot_blocklist)
    ? [...new Set(src.bot_blocklist.map((name) => String(name ?? "").trim().toLowerCase()).filter(Boolean))]
    : [];

  const overridesRaw = (src.symbol_overrides ?? {}) as Record<string, unknown>;
  const weightingRaw = (src.author_weighting ?? {}) as Record<string, unknown>;
  const num = (value: unknown, fallback: number, min: number, max: number) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? Math.min(max, Math.max(min, parsed)) : fallback;
  };

  return {
    subreddits,
    bot_blocklist: blocklist,
    symbol_overrides: {
      force_ambiguous: sanitizeSymbolList(overridesRaw.force_ambiguous),
      force_unambiguous: sanitizeSymbolList(overridesRaw.force_unambiguous),
    },
    author_weighting: {
      low_diversity_share: num(weightingRaw.low_diversity_share, 0.8, 0.5, 1),
      low_diversity_max_tickers: Math.round(num(weightingRaw.low_diversity_max_tickers, 2, 1, 10)),
      discount: num(weightingRaw.discount, 0.25, 0, 1),
      min_items: Math.round(num(weightingRaw.min_items, 5, 1, 100)),
    },
  };
}

export async function GET(): Promise<NextResponse> {
  try {
    const config = await getAttentionSweepConfig();
    return NextResponse.json({ ok: true, data: { config: config ?? DEFAULT_CONFIG, saved: config != null } });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}

export async function PUT(req: NextRequest): Promise<NextResponse> {
  try {
    const body = (await req.json()) as { config?: unknown };
    const config = sanitizeConfig(body.config);
    if (!config) {
      return NextResponse.json({ ok: false, error: "config with at least one valid subreddit is required" }, { status: 400 });
    }
    await saveAttentionSweepConfig(config);
    return NextResponse.json({ ok: true, data: { config } });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}
