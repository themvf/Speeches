"use client";

import type { CorpusEventChip } from "@/lib/server/types";

// Corpus events beside a ticker: what the document archive knows about this
// company, linked to the source.
//
// The chip text is the source kind plus the document's own title, verbatim in
// the tooltip. It deliberately never characterizes what the document says
// about the company - the moment we paraphrase, we are making the claim rather
// than linking to one, and these render on a public page next to a real name.

function daysAgo(publishedDate: string): string {
  const then = Date.parse(`${publishedDate}T00:00:00Z`);
  if (Number.isNaN(then)) return "";
  const days = Math.floor((Date.now() - then) / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "1d";
  return `${days}d`;
}

export function CorpusChips({ events }: { events?: CorpusEventChip[] }) {
  if (!events || events.length === 0) return null;
  return (
    <span className="inline-flex flex-wrap items-center gap-1 align-middle">
      {events.map((event) => {
        const age = daysAgo(event.publishedDate);
        return (
          <a
            key={event.documentId}
            href={event.url}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            title={`${event.title} — ${event.sourceLabel}${age ? ` · ${age} ago` : ""} · opens the source document`}
            className="rounded border border-[color:rgba(159,179,196,0.28)] px-1.5 py-0.5 text-[9px] font-semibold leading-none text-[color:var(--ink-faint)] hover:text-[color:var(--ink-soft)] hover:underline"
          >
            {event.sourceLabel}{age ? ` · ${age}` : ""}
          </a>
        );
      })}
    </span>
  );
}
