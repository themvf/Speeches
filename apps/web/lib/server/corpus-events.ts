// Shared fail-soft loader turning recent corpus documents that name a ticker
// into per-ticker chips for the movers/attention boards. Same contract as
// filing-chips.ts: never throws - a missing table, a corpus that has not been
// indexed yet, or an unreachable DB just means no chips.
//
// The gating rules live in lib/corpus-event-display.ts so they can be tested
// without a database import; this file is only the fetch and the catch.

import type { CorpusEventChip } from "@/lib/server/types";
import { buildCorpusChips } from "@/lib/corpus-event-display";
import { getRecentCorpusEvents } from "@/lib/server/neon";

export async function loadCorpusEventChips(daysBack = 30): Promise<Map<string, CorpusEventChip[]>> {
  try {
    return buildCorpusChips(await getRecentCorpusEvents(daysBack));
  } catch {
    // fail-soft by design
    return new Map();
  }
}
