"use client";

import type { FilingEventChip } from "@/lib/server/types";

// SEC-50: "why is this moving?" catalyst chips. Color encodes the read:
// 8-K = cyan (event disclosed), insider buy = green, insider sell = red,
// other Form 4 = neutral. Links go to the EDGAR filing index page.

function timeAgo(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime();
  const hours = Math.floor(ms / 3_600_000);
  if (hours < 1) return "<1h";
  if (hours < 48) return `${hours}h`;
  return `${Math.floor(hours / 24)}d`;
}

function chipColor(chip: FilingEventChip): string {
  if (chip.form === "8-K") return "#4fd5ff";
  if (/bought/i.test(chip.label)) return "#41d39d";
  if (/sold/i.test(chip.label)) return "#f87171";
  return "var(--ink-faint)";
}

export function FilingChips({ filings }: { filings?: FilingEventChip[] }) {
  if (!filings || filings.length === 0) return null;
  return (
    <span className="inline-flex flex-wrap items-center gap-1 align-middle">
      {filings.map((chip) => {
        const color = chipColor(chip);
        return (
          <a
            key={chip.url + chip.filedAt}
            href={chip.url}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            title={`${chip.label} · filed ${new Date(chip.filedAt).toLocaleString()} · opens EDGAR`}
            className="rounded px-1.5 py-0.5 text-[9px] font-semibold leading-none hover:underline"
            style={{ color, backgroundColor: `color-mix(in srgb, ${color} 13%, transparent)` }}
          >
            {chip.form === "8-K" ? "8-K" : "Insider"} · {timeAgo(chip.filedAt)}
          </a>
        );
      })}
    </span>
  );
}
