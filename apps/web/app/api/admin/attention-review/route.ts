import { NextRequest, NextResponse } from "next/server";
import {
  getAttentionReviewQueue,
  getAttentionSweepConfig,
  getRedditAttentionItems,
  resolveAttentionReviewItem,
  saveAttentionSweepConfig,
} from "@/lib/server/neon";

export const dynamic = "force-dynamic";

export async function GET(): Promise<NextResponse> {
  try {
    const queue = await getAttentionReviewQueue("pending");
    // Attach sample thread titles so the reviewer can judge without
    // leaving the panel - same items join the drawer uses.
    const allIds = [...new Set(queue.flatMap((row) => {
      try {
        const parsed = JSON.parse(row.sample_source_ids || "[]");
        return Array.isArray(parsed) ? parsed.map((id: unknown) => String(id)).slice(0, 3) : [];
      } catch {
        return [];
      }
    }))];
    const items = await getRedditAttentionItems(allIds);
    const itemsById = new Map(items.map((item) => [item.source_id, item]));
    const enriched = queue.map((row) => {
      let sampleIds: string[] = [];
      try {
        const parsed = JSON.parse(row.sample_source_ids || "[]");
        sampleIds = Array.isArray(parsed) ? parsed.map((id: unknown) => String(id)).slice(0, 3) : [];
      } catch { /* keep empty */ }
      return {
        ...row,
        samples: sampleIds
          .map((id) => itemsById.get(id))
          .filter((item): item is NonNullable<typeof item> => Boolean(item))
          .map((item) => ({ title: item.title, permalink: item.permalink, subreddit: item.subreddit })),
      };
    });
    return NextResponse.json({ ok: true, data: { queue: enriched } });
  } catch (err) {
    // Python-owned table may not exist yet - an empty queue with a warning
    // keeps the admin panel rendering.
    return NextResponse.json({ ok: true, data: { queue: [], warning: `Review queue unavailable: ${String(err)}` } });
  }
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const body = (await req.json()) as { id?: unknown; action?: unknown };
    const id = Number(body.id);
    const action = String(body.action ?? "");
    if (!Number.isInteger(id) || id <= 0 || !["legit", "false_positive"].includes(action)) {
      return NextResponse.json({ ok: false, error: "id and action ('legit' | 'false_positive') are required" }, { status: 400 });
    }

    const row = await resolveAttentionReviewItem(id, action as "legit" | "false_positive");
    if (!row) {
      return NextResponse.json({ ok: false, error: "review item not found" }, { status: 404 });
    }

    // A false positive also force-gates the symbol in the sweep config, so
    // the next sweep/rollup stops counting it bare (item 6 -> item 4 write-
    // back). Config write failures surface but don't undo the review mark.
    let configUpdated = false;
    if (action === "false_positive") {
      const config = await getAttentionSweepConfig();
      if (config) {
        const gated = new Set(config.symbol_overrides.force_ambiguous.map((symbol) => symbol.toUpperCase()));
        if (!gated.has(row.ticker.toUpperCase())) {
          config.symbol_overrides.force_ambiguous = [...gated, row.ticker.toUpperCase()].sort();
          await saveAttentionSweepConfig(config);
          configUpdated = true;
        }
      }
    }

    return NextResponse.json({ ok: true, data: { row, configUpdated } });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}
