"use client";

import { useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import {
  buildXSignalInsights,
  suggestedXSignalSubjects,
  type XSignalPost,
  type XSignalWindow,
} from "@/lib/x-signal-insights";
import { decodeEntities } from "@/lib/intel-topic-matching";

const WINDOW_OPTIONS: { value: XSignalWindow; label: string }[] = [
  { value: "24h", label: "24 hours" },
  { value: "7d", label: "7 days" },
  { value: "all", label: "All saved" },
];
const buttonClass = "min-h-11 rounded-lg border border-[color:var(--line)] px-3 py-2 text-sm font-medium text-[color:var(--ink)] transition hover:bg-white/5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300";

function postDate(value: string | null): string {
  if (!value || !Number.isFinite(Date.parse(value))) return "Date unavailable";
  return new Intl.DateTimeFormat("en", { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" }).format(new Date(value));
}

function postAuthor(post: XSignalPost): string {
  return decodeEntities(post.author || post.feed_label || "Tracked account");
}

function sourceUrl(post: XSignalPost): string | null {
  try {
    const url = new URL(post.url);
    if (!["https:", "http:"].includes(url.protocol) || !/^(www\.|mobile\.)?(x|twitter)\.com$/i.test(url.hostname)) return null;
    url.protocol = "https:";
    return url.href;
  } catch {
    return null;
  }
}

function SourceLink({ post }: { post: XSignalPost }) {
  const url = sourceUrl(post);
  return url ? (
    <a href={url} target="_blank" rel="noopener noreferrer" className="inline-flex min-h-11 items-center gap-2 text-sm font-semibold text-cyan-300 underline-offset-4 hover:underline" aria-label={`Read post by ${postAuthor(post)} on X (opens in a new tab)`}>
      Read on X <span aria-hidden="true">↗</span>
    </a>
  ) : <span className="text-xs text-[color:var(--ink-faint)]">Source link unavailable</span>;
}

export function XSignalsWorkspace({ query, setQuery }: { query: string; setQuery: (value: string) => void }) {
  const deferredQuery = useDeferredValue(query.trim());
  const [window, setWindow] = useState<XSignalWindow>("7d");
  const [posts, setPosts] = useState<XSignalPost[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const controllerRef = useRef<AbortController | null>(null);

  const load = useCallback(async () => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/intel/feed?source=X&limit=500&includeDocuments=0", { signal: controller.signal, cache: "no-store" });
      if (!response.ok) throw new Error("The X feed could not be loaded. Try again in a moment.");
      const body = await response.json();
      if (!body.ok || !Array.isArray(body.data?.articles)) throw new Error("The X feed could not be loaded. Try again in a moment.");
      if (body.data.articlesAvailable === false) throw new Error("The saved X feed is not connected in this environment.");
      if (controller.signal.aborted) return;
      setPosts(body.data.articles);
      setUpdatedAt(body.data.generatedAt || new Date().toISOString());
      setNow(Date.now());
    } catch (cause) {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "The X feed could not be loaded.");
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    return () => controllerRef.current?.abort();
  }, [load]);

  const insights = useMemo(() => buildXSignalInsights(posts, { query: deferredQuery, window, now }), [posts, deferredQuery, window, now]);
  const allTime = useMemo(() => buildXSignalInsights(posts, { query: deferredQuery, window: "all", now }), [posts, deferredQuery, now]);
  const suggestions = useMemo(() => suggestedXSignalSubjects(posts), [posts]);
  const waiting = query.trim() !== deferredQuery;
  const hasLoaded = updatedAt !== null;

  return (
    <section className="min-w-0 space-y-4" aria-labelledby="x-signals-title">
      <div className="rounded-2xl border border-[color:var(--line)] bg-[color:var(--bg-elev)] p-4 md:p-5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest text-purple-300">Quick insights</p>
            <h2 id="x-signals-title" className="mt-1 text-2xl font-semibold">X signals</h2>
          </div>
          <button type="button" className={`${buttonClass} shrink-0 disabled:opacity-50`} onClick={() => void load()} disabled={loading}>
            {loading ? "Loading…" : "Refresh"}
          </button>
        </div>
        <label htmlFor="x-signal-query" className="mb-2 mt-4 block text-sm font-medium">What are you following?</label>
        <div className="flex gap-2">
          <input id="x-signal-query" type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Company, topic or @handle" autoComplete="off" className="min-h-12 min-w-0 flex-1 rounded-xl border border-[color:var(--line-strong)] bg-[color:var(--bg)] px-3 text-base text-[color:var(--ink)] placeholder:text-[color:var(--ink-faint)] focus:border-cyan-300 focus:outline-none focus:ring-1 focus:ring-cyan-300" />
          {query ? <button type="button" className={buttonClass} onClick={() => setQuery("")}>Clear</button> : null}
        </div>
        {!query ? (
          <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
            <span className="text-[color:var(--ink-faint)]">Try</span>
            {["Backpack", ...suggestions.filter((subject) => subject.toLowerCase() !== "backpack")].slice(0, 4).map((subject) => (
              <button key={subject} type="button" onClick={() => setQuery(subject)} className="min-h-11 max-w-full break-words rounded-lg px-2 py-2 text-cyan-300 hover:bg-cyan-300/10">{subject}</button>
            ))}
          </div>
        ) : null}
        <div className="mt-3 flex flex-wrap gap-2" role="group" aria-label="Post publication window">
          {WINDOW_OPTIONS.map((option) => (
            <button key={option.value} type="button" aria-pressed={window === option.value} onClick={() => { setWindow(option.value); setNow(Date.now()); }} className={`${buttonClass} ${window === option.value ? "border-purple-400/70 bg-purple-400/15 text-purple-200" : "text-[color:var(--ink-faint)]"}`}>{option.label}</button>
          ))}
        </div>
      </div>

      {error ? (
        <div role="alert" className="rounded-xl border border-amber-400/30 bg-amber-400/5 p-4 text-sm text-amber-200">
          <p>{error}{hasLoaded ? " Showing the last loaded posts below." : ""}</p>
          <button type="button" className={`${buttonClass} mt-3`} onClick={() => void load()} disabled={loading}>Try again</button>
        </div>
      ) : null}

      <div aria-live="polite" aria-busy={loading || waiting}>
        {loading && !hasLoaded ? (
          <div role="status" className="rounded-2xl border border-[color:var(--line)] p-5 text-sm text-[color:var(--ink-soft)]">Loading saved X posts…</div>
        ) : !hasLoaded ? null : insights.posts.length === 0 ? (
          <div className="rounded-2xl border border-[color:var(--line)] bg-[color:var(--bg-elev)] p-5">
            <h3 className="break-words text-lg font-semibold">{deferredQuery ? `No matching posts for “${deferredQuery}”` : "No posts in this window"}</h3>
            <p className="mt-2 text-sm leading-relaxed text-[color:var(--ink-soft)]">
              {allTime.posts.length > 0 ? `${allTime.posts.length} matching saved post${allTime.posts.length === 1 ? " is" : "s are"} outside this window or missing a publication date.` : "Try another name or handle. This view covers saved posts from tracked accounts; it does not search all of X."}
            </p>
            {window !== "all" ? <button type="button" className={`${buttonClass} mt-4`} onClick={() => setWindow("all")}>Search all saved posts</button> : null}
          </div>
        ) : (
          <div className={`space-y-4 ${waiting ? "opacity-60" : ""}`}>
            <div className="rounded-2xl border border-purple-400/25 bg-gradient-to-br from-purple-400/10 to-[color:var(--bg-elev)] p-4 md:p-5">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <h3 className="min-w-0 break-words text-xl font-semibold">{deferredQuery || "X"} at a glance</h3>
                <span className="rounded-full bg-purple-300/10 px-2.5 py-1 text-xs font-medium text-purple-200">{WINDOW_OPTIONS.find((option) => option.value === window)?.label}</span>
              </div>
              <p className="mt-2 text-sm text-[color:var(--ink-soft)]">{insights.posts.length} saved post{insights.posts.length === 1 ? "" : "s"} · {insights.accountCount} account{insights.accountCount === 1 ? "" : "s"}</p>
              <p className="mt-1 text-xs leading-relaxed text-[color:var(--ink-faint)]">Latest post: {postDate(insights.latestPublishedAt)}</p>
              <p className="mt-4 text-xs font-semibold uppercase tracking-widest text-purple-200">The quick read</p>
              <ol className="mt-2 divide-y divide-purple-300/15">
                {insights.takeaways.map((takeaway, index) => (
                  <li key={takeaway.post.id} className="py-4 first:pt-2 last:pb-0">
                    <div className="flex items-center gap-2 text-xs text-[color:var(--ink-faint)]">
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-purple-300/15 font-semibold text-purple-200">{index + 1}</span>
                      <span>{takeaway.kind === "analysis" ? "Saved analysis" : "Post excerpt"}</span>
                    </div>
                    <p className="mt-2 break-words text-base font-medium leading-relaxed">{takeaway.text}</p>
                    {takeaway.whyItMatters ? <p className="mt-2 break-words text-sm leading-relaxed text-[color:var(--ink-soft)]"><span className="font-semibold text-purple-200">Why it matters: </span>{takeaway.whyItMatters}</p> : null}
                    {takeaway.watchNext ? <p className="mt-2 break-words text-sm leading-relaxed text-[color:var(--ink-soft)]"><span className="font-semibold text-purple-200">Watch next: </span>{takeaway.watchNext}</p> : null}
                    <div className="mt-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
                      <div className="min-w-0 text-xs leading-relaxed text-[color:var(--ink-faint)]">
                        <p className="break-words">{postAuthor(takeaway.post)}</p>
                        <p>{postDate(takeaway.post.published_at)}</p>
                      </div>
                      <SourceLink post={takeaway.post} />
                    </div>
                  </li>
                ))}
              </ol>
            </div>
            <details key={`${deferredQuery}:${window}`} className="rounded-2xl border border-[color:var(--line)] bg-[color:var(--bg-elev)]">
              <summary className="min-h-12 cursor-pointer px-4 py-4 text-sm font-semibold">All supporting posts ({insights.posts.length})</summary>
              <div className="divide-y divide-[color:var(--line)] px-4">
                {insights.posts.map((post) => (
                  <article key={post.id} className="py-4">
                    <p className="break-words text-sm font-semibold">{postAuthor(post)}</p>
                    <p className="mt-1 text-xs text-[color:var(--ink-faint)]">{postDate(post.published_at)}</p>
                    <p className="mt-2 whitespace-pre-line break-words text-sm leading-relaxed text-[color:var(--ink-soft)]">{decodeEntities(post.description || post.title).replace(/<[^>]*>/g, "")}</p>
                    <SourceLink post={post} />
                  </article>
                ))}
              </div>
            </details>
          </div>
        )}
      </div>
      <p className="px-1 text-xs leading-relaxed text-[color:var(--ink-faint)]">
        Based on up to 500 saved X posts from tracked accounts that match the feed’s topics. Takeaways reflect individual posts, not a verified consensus.
        {insights.undatedCount > 0 ? ` ${insights.undatedCount} matching post${insights.undatedCount === 1 ? " has" : "s have"} no publication date.` : ""}
        {updatedAt ? ` Last loaded: ${postDate(updatedAt)}.` : ""}
      </p>
    </section>
  );
}
