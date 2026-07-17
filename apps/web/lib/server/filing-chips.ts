// SEC-50: shared fail-soft loader turning recent filing_events rows into
// per-ticker catalyst chips for the movers/attention boards. Never throws -
// a missing table or unreachable DB just means no chips.

import type { FilingEventChip } from "@/lib/server/types";
import { getRecentFilingEvents } from "@/lib/server/neon";

const MAX_CHIPS_PER_TICKER = 2;

export async function loadFilingChips(hoursBack = 72): Promise<Map<string, FilingEventChip[]>> {
  const byTicker = new Map<string, FilingEventChip[]>();
  try {
    for (const row of await getRecentFilingEvents(hoursBack)) {
      const chips = byTicker.get(row.ticker) ?? [];
      if (chips.length >= MAX_CHIPS_PER_TICKER) continue; // rows arrive newest-first
      chips.push({
        form: row.form,
        filedAt: row.filed_at,
        label: row.summary || (row.form === "8-K" ? "8-K filed" : "Insider transaction"),
        url: row.url,
      });
      byTicker.set(row.ticker, chips);
    }
  } catch {
    // fail-soft by design
  }
  return byTicker;
}
