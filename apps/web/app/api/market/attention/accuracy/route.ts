import { NextResponse } from "next/server";

import { getAttentionSourceStats } from "@/lib/server/neon";

export const dynamic = "force-dynamic";

// Per-subreddit and per-author forward-return hit rates (enhancement 2).
//
// Fail-soft: attention_source_stats is Python-owned and created by
// attention_outcomes.py, so it does not exist until that job has run. It also
// stays legitimately EMPTY for a while after that - a 20-day horizon needs 20
// trading days to resolve - so "no rows" is a normal state with its own
// message, not an error.
export async function GET(request: Request) {
  const url = new URL(request.url);
  const kindParam = url.searchParams.get("kind");
  const kind = kindParam === "subreddit" || kindParam === "author" ? kindParam : undefined;

  try {
    const rows = await getAttentionSourceStats(kind, 150);
    return NextResponse.json({
      ok: true,
      data: {
        rows,
        kind: kind ?? null,
        warning: rows.length
          ? undefined
          : "No scored outcomes yet. Hit rates appear once attention days have had time to resolve against forward returns.",
      },
    });
  } catch (error) {
    return NextResponse.json({
      ok: true,
      data: {
        rows: [],
        kind: kind ?? null,
        warning:
          error instanceof Error && /relation .*attention_source_stats.* does not exist/i.test(error.message)
            ? "Outcome scoring has not run yet."
            : "Attention accuracy is unavailable right now.",
      },
    });
  }
}
