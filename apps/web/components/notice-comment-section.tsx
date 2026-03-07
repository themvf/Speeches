"use client";

import { useDeferredValue, useEffect, useMemo, useState } from "react";

interface ApiEnvelope<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

interface NoticeCommentItem {
  document_id: string;
  title: string;
  commenter_name: string;
  commenter_org: string;
  speaker: string;
  url: string;
  comment_url: string;
  pdf_url: string;
  published_at: string;
  summary: string;
  tags: string[];
  keywords: string[];
  enrichment_status: string;
  review_decision: string;
  comment_position: {
    label: string;
    confidence: number;
    rationale: string;
  };
}

interface NoticeGroupItem {
  notice_key: string;
  notice_document_id: string;
  notice_number: string;
  title: string;
  summary: string;
  organization: string;
  url: string;
  pdf_url: string;
  published_at: string;
  effective_date: string;
  comment_deadline: string;
  tags: string[];
  keywords: string[];
  enrichment_status: string;
  review_decision: string;
  comment_count: number;
  latest_comment_at: string;
  comments: NoticeCommentItem[];
}

interface NoticeCommentsResponse {
  groups: NoticeGroupItem[];
  totals: {
    notices: number;
    comments: number;
    enriched_comments: number;
    pending_review_comments: number;
  };
}

function fmt(value: number): string {
  return new Intl.NumberFormat("en-US").format(value || 0);
}

function fmtDateOnly(value: string): string {
  if (!value) return "-";
  const d = new Date(value);
  return Number.isNaN(d.getTime())
    ? value
    : d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function statusClass(value: string): string {
  const s = String(value || "").toLowerCase();
  if (["enriched", "reviewed", "success"].includes(s)) return "status-chip status-success";
  if (["fallback_enriched", "queued", "running"].includes(s)) return "status-chip status-warn";
  if (["failed", "rejected"].includes(s)) return "status-chip status-failure";
  return "status-chip status-neutral";
}

function positionClass(value: string): string {
  const label = String(value || "").toLowerCase();
  if (label === "supportive") return "status-chip status-success";
  if (label === "opposed") return "status-chip status-failure";
  if (label === "mixed") return "status-chip status-warn";
  return "status-chip status-neutral";
}

function formatPositionLabel(value: string): string {
  const normalized = String(value || "").trim();
  if (!normalized) {
    return "Unclear";
  }
  return normalized
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url, {
    cache: "no-store",
    headers: { "Content-Type": "application/json" }
  });
  const payload = (await res.json()) as ApiEnvelope<T>;
  if (!res.ok || !payload?.ok || !payload.data) {
    throw new Error(payload?.error || `Request failed (${res.status})`);
  }
  return payload.data;
}

function detailChips(items: string[], emptyLabel: string) {
  if (items.length === 0) {
    return <span className="text-xs text-[color:var(--ink-faint)]">{emptyLabel}</span>;
  }
  return (
    <div className="flex flex-wrap gap-2">
      {items.map((item) => (
        <span key={item} className="tone-chip">
          {item}
        </span>
      ))}
    </div>
  );
}

export function NoticeCommentSection() {
  const [data, setData] = useState<NoticeCommentsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const deferredSearch = useDeferredValue(search);

  useEffect(() => {
    let canceled = false;

    const load = async () => {
      setLoading(true);
      setError("");
      try {
        const payload = await fetchJson<NoticeCommentsResponse>("/api/notices-comments");
        if (!canceled) {
          setData(payload);
        }
      } catch (err) {
        if (!canceled) {
          setError(err instanceof Error ? err.message : "Failed to load FINRA notices/comments.");
        }
      } finally {
        if (!canceled) {
          setLoading(false);
        }
      }
    };

    void load();
    return () => {
      canceled = true;
    };
  }, []);

  const filteredGroups = useMemo(() => {
    if (!data) {
      return [];
    }

    const token = deferredSearch.trim().toLowerCase();
    if (!token) {
      return data.groups;
    }

    return data.groups.filter((group) => {
      const haystack = [
        group.notice_number,
        group.title,
        group.summary,
        group.published_at,
        group.effective_date,
        group.comment_deadline,
        ...group.tags,
        ...group.keywords,
        ...group.comments.flatMap((comment) => [
          comment.title,
          comment.commenter_name,
          comment.commenter_org,
          comment.speaker,
          comment.summary,
          comment.comment_position.label,
          comment.comment_position.rationale,
          ...comment.tags,
          ...comment.keywords
        ])
      ]
        .join("\n")
        .toLowerCase();

      return haystack.includes(token);
    });
  }, [data, deferredSearch]);

  const filteredCommentCount = useMemo(
    () => filteredGroups.reduce((sum, group) => sum + group.comments.length, 0),
    [filteredGroups]
  );

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-6 md:px-8 md:py-10">
      <header className="panel hero-panel reveal p-6 md:p-8">
        <span className="kicker">FINRA Workflow</span>
        <h1 className="mt-3 text-3xl font-bold leading-tight md:text-5xl">Notices &amp; Comments</h1>
        <p className="mt-3 max-w-3xl text-base text-[color:var(--ink-soft)] md:text-lg">
          Review each FINRA Regulatory Notice alongside its summary, linked comment letters, and the
          extracted tags and keywords for every comment file.
        </p>
      </header>

      <section className="grid gap-4 md:grid-cols-[1.3fr_0.7fr]">
        <article className="panel reveal reveal-delay-1 p-5">
          <label className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
            Search Notices Or Comments
          </label>
          <input
            className="form-control mt-3 w-full px-3 py-2 text-sm"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Notice number, notice title, commenter, tags, keywords..."
          />
          <p className="mt-3 text-sm text-[color:var(--ink-faint)]">
            This view is optimized for notice-centric review. Comment position is estimated from enrichment for FINRA
            comment letters, with a legacy stance fallback until older records are re-enriched.
          </p>
        </article>

        <section className="grid gap-3 sm:grid-cols-2">
          <article className="panel p-4">
            <p className="text-xs uppercase tracking-[0.1em]">Notices</p>
            <p className="mt-1 text-2xl font-semibold">
              {loading ? "..." : `${fmt(filteredGroups.length)} / ${fmt(data?.totals.notices || 0)}`}
            </p>
          </article>
          <article className="panel p-4">
            <p className="text-xs uppercase tracking-[0.1em]">Comments</p>
            <p className="mt-1 text-2xl font-semibold">
              {loading ? "..." : `${fmt(filteredCommentCount)} / ${fmt(data?.totals.comments || 0)}`}
            </p>
          </article>
          <article className="panel p-4">
            <p className="text-xs uppercase tracking-[0.1em]">Enriched Comments</p>
            <p className="mt-1 text-2xl font-semibold">{loading ? "..." : fmt(data?.totals.enriched_comments || 0)}</p>
          </article>
          <article className="panel p-4">
            <p className="text-xs uppercase tracking-[0.1em]">Pending Review</p>
            <p className="mt-1 text-2xl font-semibold">
              {loading ? "..." : fmt(data?.totals.pending_review_comments || 0)}
            </p>
          </article>
        </section>
      </section>

      {error ? <p className="callout callout-error">{error}</p> : null}

      {loading ? (
        <section className="panel p-5">
          <p className="text-sm">Loading FINRA notices and comment letters...</p>
        </section>
      ) : filteredGroups.length === 0 ? (
        <section className="panel p-5">
          <p className="text-sm">No FINRA notice/comment groups matched the current search.</p>
        </section>
      ) : (
        <section className="grid gap-4">
          {filteredGroups.map((group) => (
            <article key={group.notice_key} className="panel reveal reveal-delay-2 overflow-hidden p-5">
              <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div className="space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    {group.notice_number ? (
                      <span className="type-chip type-regulatory">{group.notice_number}</span>
                    ) : null}
                    <span className="tone-chip">{group.comment_count} comments</span>
                    <span className={statusClass(group.enrichment_status)}>{group.enrichment_status}</span>
                  </div>
                  <h2 className="text-2xl font-semibold leading-tight">{group.title || "FINRA Regulatory Notice"}</h2>
                  <p className="max-w-4xl text-sm text-[color:var(--ink-soft)]">
                    {group.summary || "No notice summary is available yet. Enrich the notice to generate one."}
                  </p>
                </div>

                <div className="grid gap-2 text-xs text-[color:var(--ink-faint)] sm:grid-cols-2">
                  <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.84)] px-3 py-2">
                    <p className="uppercase tracking-[0.08em]">Published</p>
                    <p className="mt-1 text-sm text-[color:var(--ink)]">{fmtDateOnly(group.published_at)}</p>
                  </div>
                  <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.84)] px-3 py-2">
                    <p className="uppercase tracking-[0.08em]">Effective</p>
                    <p className="mt-1 text-sm text-[color:var(--ink)]">{group.effective_date || "-"}</p>
                  </div>
                  <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.84)] px-3 py-2">
                    <p className="uppercase tracking-[0.08em]">Comment Deadline</p>
                    <p className="mt-1 text-sm text-[color:var(--ink)]">{group.comment_deadline || "-"}</p>
                  </div>
                  <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.84)] px-3 py-2">
                    <p className="uppercase tracking-[0.08em]">Latest Comment</p>
                    <p className="mt-1 text-sm text-[color:var(--ink)]">{fmtDateOnly(group.latest_comment_at)}</p>
                  </div>
                </div>
              </div>

              <div className="mt-4 flex flex-wrap gap-3 text-sm">
                {group.url ? (
                  <a href={group.url} target="_blank" rel="noreferrer" className="link-inline">
                    Open notice
                  </a>
                ) : null}
                {group.pdf_url ? (
                  <a href={group.pdf_url} target="_blank" rel="noreferrer" className="link-inline">
                    Notice PDF
                  </a>
                ) : null}
              </div>

              {(group.tags.length > 0 || group.keywords.length > 0) && (
                <div className="mt-4 grid gap-3 md:grid-cols-2">
                  <div>
                    <p className="mb-2 text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
                      Notice Tags
                    </p>
                    {detailChips(group.tags, "No notice tags")}
                  </div>
                  <div>
                    <p className="mb-2 text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
                      Notice Keywords
                    </p>
                    {detailChips(group.keywords, "No notice keywords")}
                  </div>
                </div>
              )}

              <div className="my-5 soft-divider" />

              {group.comments.length === 0 ? (
                <p className="text-sm text-[color:var(--ink-faint)]">No comment letters are linked to this notice yet.</p>
              ) : (
                <div className="grid gap-3 lg:grid-cols-2">
                  {group.comments.map((comment) => (
                    <article
                      key={comment.document_id}
                      className="rounded-2xl border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.9)] p-4"
                    >
                      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                        <div>
                          <h3 className="text-lg font-semibold leading-snug">{comment.title || "Comment Letter"}</h3>
                          <p className="mt-1 text-sm text-[color:var(--ink-soft)]">
                            {comment.commenter_name || comment.speaker || "Commenter"}
                            {comment.commenter_org ? ` | ${comment.commenter_org}` : ""}
                          </p>
                        </div>
                        <div className="flex flex-wrap items-center gap-2">
                          <span className={statusClass(comment.enrichment_status)}>{comment.enrichment_status}</span>
                          <span className="tone-chip">{fmtDateOnly(comment.published_at)}</span>
                        </div>
                      </div>

                      <p className="mt-3 text-sm text-[color:var(--ink-soft)]">
                        {comment.summary || "No comment summary is available yet. Enrich this comment to generate one."}
                      </p>

                      <div className="mt-3 flex flex-wrap gap-3 text-xs text-[color:var(--ink-faint)]">
                        <span>Review: {comment.review_decision || "pending"}</span>
                        <span className={positionClass(comment.comment_position.label)}>
                          Position: {formatPositionLabel(comment.comment_position.label)}
                        </span>
                        {comment.comment_position.confidence > 0 ? (
                          <span>Confidence: {Math.round(comment.comment_position.confidence * 100)}%</span>
                        ) : null}
                      </div>
                      {comment.comment_position.rationale ? (
                        <p className="mt-2 text-xs text-[color:var(--ink-faint)]">
                          {comment.comment_position.rationale}
                        </p>
                      ) : null}

                      <div className="mt-4 grid gap-3 md:grid-cols-2">
                        <div>
                          <p className="mb-2 text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
                            Tags
                          </p>
                          {detailChips(comment.tags, "No tags yet")}
                        </div>
                        <div>
                          <p className="mb-2 text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
                            Keywords
                          </p>
                          {detailChips(comment.keywords, "No keywords yet")}
                        </div>
                      </div>

                      <div className="mt-4 flex flex-wrap gap-3 text-sm">
                        {comment.comment_url ? (
                          <a href={comment.comment_url} target="_blank" rel="noreferrer" className="link-inline">
                            Open comment
                          </a>
                        ) : null}
                        {comment.pdf_url ? (
                          <a href={comment.pdf_url} target="_blank" rel="noreferrer" className="link-inline">
                            Comment PDF
                          </a>
                        ) : null}
                        {!comment.pdf_url && comment.url ? (
                          <a href={comment.url} target="_blank" rel="noreferrer" className="link-inline">
                            Source page
                          </a>
                        ) : null}
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </article>
          ))}
        </section>
      )}
    </div>
  );
}
