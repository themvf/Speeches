"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useSavedItems, type SavedItem } from "@/hooks/use-saved-items";

type SavedFilter = "all" | SavedItem["type"];

const FILTERS: Array<{ id: SavedFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "article", label: "Articles" },
  { id: "doc", label: "Documents" },
];

function formatSavedAt(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Saved recently";
  }
  return date.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function itemTypeLabel(type: SavedItem["type"]): string {
  return type === "doc" ? "Document" : "Article";
}

function matchesSearch(item: SavedItem, query: string): boolean {
  if (!query) return true;
  return [item.title, item.source, item.topic, item.url]
    .filter(Boolean)
    .join(" ")
    .toLowerCase()
    .includes(query);
}

function sortBySavedAt(items: SavedItem[]): SavedItem[] {
  return [...items].sort((a, b) => {
    const left = new Date(a.savedAt).getTime() || 0;
    const right = new Date(b.savedAt).getTime() || 0;
    return right - left;
  });
}

export function SavedItemsPage() {
  const { items, loaded, remove, clear } = useSavedItems();
  const [filter, setFilter] = useState<SavedFilter>("all");
  const [query, setQuery] = useState("");

  const normalizedQuery = query.trim().toLowerCase();
  const counts = useMemo(() => ({
    all: items.length,
    article: items.filter((item) => item.type === "article").length,
    doc: items.filter((item) => item.type === "doc").length,
  }), [items]);

  const visibleItems = useMemo(() => {
    const filtered = items.filter((item) => {
      const typeMatches = filter === "all" || item.type === filter;
      return typeMatches && matchesSearch(item, normalizedQuery);
    });
    return sortBySavedAt(filtered);
  }, [filter, items, normalizedQuery]);

  const clearSavedItems = () => {
    if (window.confirm("Remove all saved items?")) {
      clear();
    }
  };

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-6 md:px-8 md:py-10">
      <header className="panel hero-panel reveal p-6 md:p-8">
        <span className="kicker">Saved</span>
        <h1 className="mt-3 text-3xl font-bold leading-tight md:text-5xl">Saved Research</h1>
        <p className="mt-3 max-w-3xl text-base text-[color:var(--ink-soft)] md:text-lg">
          {loaded ? `${counts.all} saved item${counts.all === 1 ? "" : "s"}` : "Loading saved items"}
        </p>
      </header>

      <section className="grid gap-4 lg:grid-cols-[260px_minmax(0,1fr)]">
        <aside className="panel h-fit p-4">
          <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">Library</p>
          <div className="mt-3 grid gap-2">
            {FILTERS.map((item) => {
              const active = filter === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  aria-pressed={active}
                  onClick={() => setFilter(item.id)}
                  className={`flex items-center justify-between rounded-lg border px-3 py-2 text-left text-sm font-semibold transition ${
                    active
                      ? "border-[color:var(--accent)] bg-[color:rgba(79,213,255,0.14)] text-[color:var(--ink)]"
                      : "border-[color:var(--line)] bg-[color:rgba(9,22,36,0.5)] text-[color:var(--ink-soft)] hover:border-[color:var(--line-strong)] hover:text-[color:var(--ink)]"
                  }`}
                >
                  <span>{item.label}</span>
                  <span className="text-xs text-[color:var(--ink-faint)]">{counts[item.id]}</span>
                </button>
              );
            })}
          </div>

          {items.length > 0 ? (
            <button type="button" onClick={clearSavedItems} className="btn-muted mt-4 w-full px-3 py-2 text-sm">
              Clear Saved
            </button>
          ) : null}
        </aside>

        <section className="panel overflow-hidden">
          <div className="flex flex-col gap-3 border-b border-[color:var(--line)] p-4 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-xl font-semibold">Items</h2>
              <p className="mt-1 text-sm text-[color:var(--ink-faint)]">
                {visibleItems.length} shown
              </p>
            </div>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search saved items"
              className="form-control w-full px-3 py-2 text-sm md:w-72"
            />
          </div>

          {!loaded ? (
            <div className="p-5 text-sm text-[color:var(--ink-faint)]">Loading saved items...</div>
          ) : visibleItems.length === 0 ? (
            <div className="p-6">
              <p className="text-sm text-[color:var(--ink-soft)]">
                {items.length === 0 ? "No saved items yet." : "No saved items match the current filter."}
              </p>
              {items.length === 0 ? (
                <Link href="/intelbeta" className="btn-solid mt-4 inline-flex px-4 py-2 text-sm">
                  Open Intelligence Feed
                </Link>
              ) : null}
            </div>
          ) : (
            <div className="divide-y divide-[color:var(--line-soft)]">
              {visibleItems.map((item) => (
                <article key={item.id} className="grid gap-3 p-4 transition hover:bg-[color:rgba(79,213,255,0.05)] md:grid-cols-[minmax(0,1fr)_auto]">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="type-chip type-default">{itemTypeLabel(item.type)}</span>
                      <span className="text-xs text-[color:var(--ink-faint)]">{item.source || "Unknown source"}</span>
                      {item.topic ? <span className="tone-chip">{item.topic}</span> : null}
                    </div>
                    <h3 className="mt-2 text-lg font-semibold leading-snug text-[color:var(--ink)]">
                      {item.title || "Untitled"}
                    </h3>
                    <p className="mt-1 text-xs text-[color:var(--ink-faint)]">{formatSavedAt(item.savedAt)}</p>
                  </div>

                  <div className="flex items-center gap-2 md:justify-end">
                    {item.url ? (
                      <a href={item.url} target="_blank" rel="noreferrer" className="btn-solid px-3 py-2 text-sm">
                        Open Source
                      </a>
                    ) : null}
                    <button type="button" onClick={() => remove(item.id)} className="btn-muted px-3 py-2 text-sm">
                      Remove
                    </button>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </section>
    </div>
  );
}
