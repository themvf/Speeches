"use client";

import { useDeferredValue, useEffect, useMemo, useState } from "react";

interface ApiEnvelope<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

interface NoticeCommentItem {
  document_id: string;
  source_kind: string;
  source_family: string;
  title: string;
  commenter_name: string;
  commenter_org: string;
  speaker: string;
  url: string;
  comment_url: string;
  pdf_url: string;
  resolved_content_url: string;
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
  source_kind: string;
  source_family: string;
  source_family_label: string;
  group_type_label: string;
  group_identifier_label: string;
  group_identifier: string;
  notice_document_id: string;
  notice_number: string;
  docket_id: string;
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

function familyChipClass(value: string): string {
  return value === "regulations_gov" ? "type-chip type-news" : "type-chip type-regulatory";
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

function commentDisplayName(comment: NoticeCommentItem): string {
  return comment.commenter_name || comment.commenter_org || comment.speaker || "Commenter";
}

function commentOrgSuffix(comment: NoticeCommentItem): string {
  const primary = commentDisplayName(comment);
  const org = String(comment.commenter_org || "").trim();
  if (!org || org === primary) {
    return "";
  }
  return org;
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

function groupPrimaryLinkLabel(group: NoticeGroupItem): string {
  return group.source_family === "regulations_gov" ? "Open rule/docket" : "Open notice";
}

function groupPdfLabel(group: NoticeGroupItem): string {
  return group.source_family === "regulations_gov" ? "Rule PDF" : "Notice PDF";
}

export function NoticeCommentSection() {
  const [data, setData] = useState<NoticeCommentsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [familyFilter, setFamilyFilter] = useState("all");
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
          setError(err instanceof Error ? err.message : "Failed to load notices and comments.");
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

  const familyOptions = useMemo(() => {
    if (!data) {
      return [];
    }

    const map = new Map<string, { label: string; groupCount: number; commentCount: number }>();
    for (const group of data.groups) {
      const current = map.get(group.source_family) || {
        label: group.source_family_label,
        groupCount: 0,
        commentCount: 0
      };
      current.groupCount += 1;
      current.commentCount += group.comments.length;
      map.set(group.source_family, current);
    }

    return [...map.entries()].map(([value, counts]) => ({
      value,
      label: counts.label,
      groupCount: counts.groupCount,
      commentCount: counts.commentCount
    }));
  }, [data]);

  const filteredGroups = useMemo(() => {
    if (!data) {
      return [];
    }

    const token = deferredSearch.trim().toLowerCase();

    return data.groups.filter((group) => {
      if (familyFilter !== "all" && group.source_family !== familyFilter) {
        return false;
      }

      if (!token) {
        return true;
      }

      const haystack = [
        group.source_family_label,
        group.group_type_label,
        group.group_identifier_label,
        group.group_identifier,
        group.notice_number,
        group.docket_id,
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
  }, [data, deferredSearch, familyFilter]);

  const filteredCommentCount = useMemo(
    () => filteredGroups.reduce((sum, group) => sum + group.comments.length, 0),
    [filteredGroups]
  );

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-6 md:px-8 md:py-10">
      <header className="panel hero-panel reveal p-6 md:p-8">
        <span className="kicker">Notice Review</span>
        <h1 className="mt-3 text-3xl font-bold leading-tight md:text-5xl">Rulemakings &amp; Comments</h1>
        <p className="mt-3 max-w-3xl text-base text-[color:var(--ink-soft)] md:text-lg">
          Review notice and rulemaking records alongside linked comments, summaries, and extracted tags and keywords
          across FINRA and Regulations.gov workflows.
        </p>
      </header>

      <section className="grid gap-4 md:grid-cols-[1.3fr_0.7fr]">
        <article className="panel reveal reveal-delay-1 p-5">
          <label className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
            Search Rulemakings Or Comments
          </label>
          <input
            className="form-control mt-3 w-full px-3 py-2 text-sm"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Docket, notice number, title, commenter, tags, keywords..."
          />

          <div className="mt-4 flex flex-wrap gap-2">
            <button
              className={`rounded-full border px-3 py-1 text-xs font-semibold tracking-[0.08em] ${
                familyFilter === "all"
                  ? "border-[color:var(--accent)] bg-[color:rgba(79,213,255,0.14)] text-[color:var(--ink)]"
                  : "border-[color:var(--line)] text-[color:var(--ink-faint)]"
              }`}
              onClick={() => setFamilyFilter("all")}
            >
              All Sources
            </button>
            {familyOptions.map((option) => (
              <button
                key={option.value}
                className={`rounded-full border px-3 py-1 text-xs font-semibold tracking-[0.08em] ${
                  familyFilter === option.value
                    ? "border-[color:var(--accent)] bg-[color:rgba(79,213,255,0.14)] text-[color:var(--ink)]"
                    : "border-[color:var(--line)] text-[color:var(--ink-faint)]"
                }`}
                onClick={() => setFamilyFilter(option.value)}
              >
                {option.label} {option.groupCount}/{option.commentCount}
              </button>
            ))}
          </div>

          <p className="mt-3 text-sm text-[color:var(--ink-faint)]">
            Comment position is estimated from enrichment when available. The page groups FINRA by notice and
            Regulations.gov by docket/rule link.
          </p>
        </article>

        <section className="grid gap-3 sm:grid-cols-2">
          <article className="panel p-4">
            <p className="text-xs uppercase tracking-[0.1em]">Groups</p>
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
          <p className="text-sm">Loading notice and comment groups...</p>
        </section>
      ) : filteredGroups.length === 0 ? (
        <section className="panel p-5">
          <p className="text-sm">No notice or rulemaking groups matched the current filters.</p>
        </section>
      ) : (
        <section className="grid gap-4">
          {filteredGroups.map((group) => (
            <article key={group.notice_key} className="panel reveal reveal-delay-2 overflow-hidden p-5">
              <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div className="space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className={familyChipClass(group.source_family)}>{group.source_family_label}</span>
                    <span className="tone-chip">{group.group_type_label}</span>
                    {group.group_identifier ? (
                      <span className="tone-chip">
                        {group.group_identifier_label}: {group.group_identifier}
                      </span>
                    ) : null}
                    <span className="tone-chip">{group.comment_count} comments</span>
                    <span className={statusClass(group.enrichment_status)}>{group.enrichment_status}</span>
                  </div>
                  <h2 className="text-2xl font-semibold leading-tight">{group.title || "Notice or Rulemaking"}</h2>
                  <p className="max-w-4xl text-sm text-[color:var(--ink-soft)]">
                    {group.summary || "No summary is available yet. Enrich the record to generate one."}
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
                    {groupPrimaryLinkLabel(group)}
                  </a>
                ) : null}
                {group.pdf_url ? (
                  <a href={group.pdf_url} target="_blank" rel="noreferrer" className="link-inline">
                    {groupPdfLabel(group)}
                  </a>
                ) : null}
              </div>

              {(group.tags.length > 0 || group.keywords.length > 0) && (
                <div className="mt-4 grid gap-3 md:grid-cols-2">
                  <div>
                    <p className="mb-2 text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
                      Group Tags
                    </p>
                    {detailChips(group.tags, "No group tags")}
                  </div>
                  <div>
                    <p className="mb-2 text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
                      Group Keywords
                    </p>
                    {detailChips(group.keywords, "No group keywords")}
                  </div>
                </div>
              )}

              <div className="my-5 soft-divider" />

              {group.comments.length === 0 ? (
                <p className="text-sm text-[color:var(--ink-faint)]">No linked comments are associated with this group yet.</p>
              ) : (
                <div className="grid gap-3 lg:grid-cols-2">
                  {group.comments.map((comment) => (
                    <article
                      key={comment.document_id}
                      className="rounded-2xl border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.9)] p-4"
                    >
                      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                        <div>
                          <div className="flex flex-wrap items-center gap-2">
                            <span className={familyChipClass(comment.source_family)}>{group.source_family_label}</span>
                            <span className={statusClass(comment.enrichment_status)}>{comment.enrichment_status}</span>
                          </div>
                          <h3 className="mt-2 text-lg font-semibold leading-snug">{comment.title || "Comment"}</h3>
                          <p className="mt-1 text-sm text-[color:var(--ink-soft)]">
                            {commentDisplayName(comment)}
                            {commentOrgSuffix(comment) ? ` | ${commentOrgSuffix(comment)}` : ""}
                          </p>
                        </div>
                        <div className="flex flex-wrap items-center gap-2">
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
                        <p className="mt-2 text-xs text-[color:var(--ink-faint)]">{comment.comment_position.rationale}</p>
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
                        {!comment.pdf_url && comment.resolved_content_url && comment.resolved_content_url !== comment.comment_url ? (
                          <a href={comment.resolved_content_url} target="_blank" rel="noreferrer" className="link-inline">
                            Resolved file
                          </a>
                        ) : null}
                        {!comment.comment_url && comment.url ? (
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
