"use client";

import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";

import type { DocumentsFacets, TimelineBucket, TimelineResponseData } from "@/lib/server/types";

type TimelineGrain = TimelineResponseData["grain"];

interface ApiEnvelope<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

const EMPTY_FACETS: DocumentsFacets = {
  sources: [],
  organizations: [],
  topics: [],
  key_topics: [],
  keywords: [],
  statuses: []
};

const SOURCE_KIND_LABELS: Record<string, string> = {
  sec_speech: "SEC Speeches & Statements",
  sec_tm_faq: "SEC Trading & Markets FAQ",
  sec_enforcement_litigation: "SEC Litigation Releases",
  finra_regulatory_notice: "FINRA Regulatory Notices",
  finra_comment_letter: "FINRA Comment Letters",
  finra_key_topic: "FINRA Key Topics",
  doj_usao_press_release: "DOJ USAO Press Releases",
  federal_reserve_speech_testimony: "Federal Reserve Speeches/Testimony",
  cftc_press_release: "CFTC Press Releases",
  cftc_public_statement_remark: "CFTC Public Statements & Remarks",
  jdsupra_article: "JD Supra",
  investmentnews_article: "InvestmentNews",
  citywire_article: "Citywire",
  congress_crs_product: "Congress CRS Products",
  newsapi_article: "News",
  uploaded: "Uploaded"
};

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

function displaySourceKind(value: string): string {
  const raw = String(value || "").trim();
  if (!raw) {
    return "Unknown";
  }
  return SOURCE_KIND_LABELS[raw] || raw;
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

function buildTimelineUrl(params: {
  q: string;
  org: string;
  source: string;
  topic: string;
  keyword: string;
  dateFrom: string;
  dateTo: string;
  grain: TimelineGrain;
}): string {
  const next = new URLSearchParams();
  if (params.q.trim()) next.set("q", params.q.trim());
  if (params.org.trim()) next.set("org", params.org.trim());
  if (params.source.trim()) next.set("source_kind", params.source.trim());
  if (params.topic.trim()) next.set("topic", params.topic.trim());
  if (params.keyword.trim()) next.set("keyword", params.keyword.trim());
  if (params.dateFrom.trim()) next.set("date_from", params.dateFrom.trim());
  if (params.dateTo.trim()) next.set("date_to", params.dateTo.trim());
  if (params.grain !== "month") next.set("grain", params.grain);
  const query = next.toString();
  return query ? `/timeline?${query}` : "/timeline";
}

function buildCorpusHref(
  bucket: TimelineBucket,
  params: {
    q: string;
    org: string;
    source: string;
    topic: string;
    keyword: string;
  }
): string {
  const next = new URLSearchParams();
  if (params.q.trim()) next.set("q", params.q.trim());
  if (params.org.trim()) next.set("org", params.org.trim());
  if (params.source.trim()) next.set("source_kind", params.source.trim());
  if (params.topic.trim()) next.set("topic", params.topic.trim());
  if (params.keyword.trim()) next.set("keyword", params.keyword.trim());
  next.set("date_from", bucket.start);
  next.set("date_to", bucket.end);
  return `/?${next.toString()}`;
}

function shouldShowBucketLabel(index: number, bucketCount: number): boolean {
  if (bucketCount <= 8) {
    return true;
  }
  const step = Math.max(1, Math.ceil(bucketCount / 8));
  return index % step === 0 || index === bucketCount - 1;
}

export function TimelineView() {
  const searchParams = useSearchParams();

  const [q, setQ] = useState(searchParams.get("q") || "");
  const [org, setOrg] = useState(searchParams.get("org") || "");
  const [source, setSource] = useState(searchParams.get("source_kind") || searchParams.get("source") || "");
  const [topic, setTopic] = useState(searchParams.get("topic") || "");
  const [keyword, setKeyword] = useState(searchParams.get("keyword") || "");
  const [dateFrom, setDateFrom] = useState(searchParams.get("date_from") || "");
  const [dateTo, setDateTo] = useState(searchParams.get("date_to") || "");
  const [grain, setGrain] = useState<TimelineGrain>(
    searchParams.get("grain") === "year" || searchParams.get("grain") === "quarter"
      ? (searchParams.get("grain") as TimelineGrain)
      : "month"
  );

  const [data, setData] = useState<TimelineResponseData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const deferredQ = useDeferredValue(q);
  const deferredKeyword = useDeferredValue(keyword);

  useEffect(() => {
    const href = buildTimelineUrl({ q, org, source, topic, keyword, dateFrom, dateTo, grain });
    window.history.replaceState(null, "", href);
  }, [dateFrom, dateTo, grain, keyword, org, q, source, topic]);

  useEffect(() => {
    let canceled = false;

    const load = async () => {
      setLoading(true);
      setError("");

      try {
        const params = new URLSearchParams();
        if (deferredQ.trim()) params.set("q", deferredQ.trim());
        if (org.trim()) params.set("org", org.trim());
        if (source.trim()) params.set("source_kind", source.trim());
        if (topic.trim()) params.set("topic", topic.trim());
        if (deferredKeyword.trim()) params.set("keyword", deferredKeyword.trim());
        if (dateFrom.trim()) params.set("date_from", dateFrom.trim());
        if (dateTo.trim()) params.set("date_to", dateTo.trim());
        if (grain !== "month") params.set("grain", grain);

        const payload = await fetchJson<TimelineResponseData>(`/api/timeline?${params.toString()}`);
        if (!canceled) {
          setData(payload);
        }
      } catch (err) {
        if (!canceled) {
          setData(null);
          setError(err instanceof Error ? err.message : "Failed to load timeline.");
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
  }, [dateFrom, dateTo, deferredKeyword, deferredQ, grain, org, source, topic]);

  const facets = data?.facets || EMPTY_FACETS;
  const sourceOptions = useMemo(
    () => [...facets.sources].sort((a, b) => displaySourceKind(a).localeCompare(displaySourceKind(b))),
    [facets.sources]
  );
  const topicOptions = useMemo(
    () => (facets.key_topics.length > 0 ? facets.key_topics : facets.topics.slice(0, 20)),
    [facets.key_topics, facets.topics]
  );
  const buckets = data?.buckets || [];
  const peakBucketKey = data?.totals.peak_bucket_key || "";
  const peakBucket = useMemo(
    () => buckets.find((bucket) => bucket.key === peakBucketKey) || null,
    [buckets, peakBucketKey]
  );
  const maxBucketCount = useMemo(
    () => buckets.reduce((best, bucket) => (bucket.count > best ? bucket.count : best), 0),
    [buckets]
  );

  const resetFilters = () => {
    setQ("");
    setOrg("");
    setSource("");
    setTopic("");
    setKeyword("");
    setDateFrom("");
    setDateTo("");
    setGrain("month");
  };

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-6 md:px-8 md:py-10">
      <header className="panel hero-panel reveal p-6 md:p-8">
        <span className="kicker">Timeline Explorer</span>
        <h1 className="mt-3 text-3xl font-bold leading-tight md:text-5xl">Corpus Timeline</h1>
        <p className="mt-3 max-w-3xl text-base text-[color:var(--ink-soft)] md:text-lg">
          Track document volume over time around a topic, structured keyword, or broader full-text term, then drill
          directly into the matching corpus window.
        </p>
      </header>

      <section className="grid gap-4 xl:grid-cols-[1.45fr_0.55fr]">
        <article className="panel reveal reveal-delay-1 p-5">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
                Timeline Filters
              </p>
              <p className="mt-1 text-sm text-[color:var(--ink-soft)]">
                Use <strong>Search Text</strong> for broad full-text matching and <strong>Keyword</strong> for stored
                enrichment keywords.
              </p>
            </div>
            <button type="button" className="btn-muted px-3 py-2 text-sm" onClick={resetFilters}>
              Reset
            </button>
          </div>

          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            <input
              className="form-control px-3 py-2 text-sm"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search text"
            />
            <select className="form-control px-3 py-2 text-sm" value={org} onChange={(e) => setOrg(e.target.value)}>
              <option value="">All orgs</option>
              {facets.organizations.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
            <select className="form-control px-3 py-2 text-sm" value={source} onChange={(e) => setSource(e.target.value)}>
              <option value="">All sources</option>
              {sourceOptions.map((value) => (
                <option key={value} value={value}>
                  {displaySourceKind(value)}
                </option>
              ))}
            </select>
            <select className="form-control px-3 py-2 text-sm" value={topic} onChange={(e) => setTopic(e.target.value)}>
              <option value="">All topics</option>
              {topicOptions.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
            <div>
              <input
                className="form-control w-full px-3 py-2 text-sm"
                value={keyword}
                onChange={(e) => setKeyword(e.target.value)}
                placeholder="Structured keyword"
                list="timeline-keywords"
              />
              <datalist id="timeline-keywords">
                {facets.keywords.slice(0, 150).map((value) => (
                  <option key={value} value={value} />
                ))}
              </datalist>
            </div>
            <select
              className="form-control px-3 py-2 text-sm"
              value={grain}
              onChange={(e) => setGrain(e.target.value === "year" || e.target.value === "quarter" ? (e.target.value as TimelineGrain) : "month")}
            >
              <option value="month">Monthly</option>
              <option value="quarter">Quarterly</option>
              <option value="year">Yearly</option>
            </select>
            <label className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
              From
              <input
                type="date"
                className="form-control mt-2 w-full px-3 py-2 text-sm"
                value={dateFrom}
                onChange={(e) => setDateFrom(e.target.value)}
              />
            </label>
            <label className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
              To
              <input
                type="date"
                className="form-control mt-2 w-full px-3 py-2 text-sm"
                value={dateTo}
                onChange={(e) => setDateTo(e.target.value)}
              />
            </label>
            <div className="rounded-2xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.84)] px-4 py-3 text-sm text-[color:var(--ink-soft)]">
              Click any bucket below to open the matching time window in the main corpus explorer.
            </div>
          </div>
        </article>

        <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
          <article className="panel p-4">
            <p className="text-xs uppercase tracking-[0.1em]">Matching Docs</p>
            <p className="mt-1 text-2xl font-semibold">
              {loading ? "..." : fmt(data?.totals.matching_documents || 0)}
            </p>
          </article>
          <article className="panel p-4">
            <p className="text-xs uppercase tracking-[0.1em]">Dated Docs</p>
            <p className="mt-1 text-2xl font-semibold">
              {loading ? "..." : fmt(data?.totals.dated_documents || 0)}
            </p>
          </article>
          <article className="panel p-4">
            <p className="text-xs uppercase tracking-[0.1em]">Date Range</p>
            <p className="mt-1 text-sm font-semibold">
              {loading
                ? "..."
                : data?.totals.start_date
                  ? `${fmtDateOnly(data.totals.start_date)} to ${fmtDateOnly(data.totals.end_date)}`
                  : "No dated documents"}
            </p>
          </article>
          <article className="panel p-4">
            <p className="text-xs uppercase tracking-[0.1em]">Peak Bucket</p>
            <p className="mt-1 text-sm font-semibold">
              {loading
                ? "..."
                : peakBucket
                  ? `${peakBucket.label} (${fmt(peakBucket.count)})`
                  : "No bucket peak"}
            </p>
          </article>
        </section>
      </section>

      {error ? <p className="callout callout-error">{error}</p> : null}
      {!loading && (data?.totals.undated_documents || 0) > 0 ? (
        <p className="callout callout-info">
          {fmt(data?.totals.undated_documents || 0)} matching document(s) are excluded from the chart because they do
          not have a usable published date.
        </p>
      ) : null}

      <section className="grid gap-4 xl:grid-cols-[1.35fr_0.65fr]">
        <article className="panel reveal reveal-delay-2 p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-xl font-semibold">Document Volume Over Time</h2>
              <p className="mt-1 text-sm text-[color:var(--ink-soft)]">
                {loading
                  ? "Loading timeline..."
                  : `${fmt(data?.totals.bucket_count || 0)} bucket(s) at ${grain} grain.`}
              </p>
            </div>
            <a
              href={buildTimelineUrl({ q, org, source, topic, keyword, dateFrom, dateTo, grain })}
              className="link-inline text-sm"
            >
              Copyable URL State
            </a>
          </div>

          {loading ? (
            <div className="mt-4 rounded-2xl border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.84)] px-4 py-16 text-sm">
              Loading timeline buckets...
            </div>
          ) : buckets.length === 0 ? (
            <div className="mt-4 rounded-2xl border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.84)] px-4 py-16 text-sm">
              No matching dated documents for the current filters.
            </div>
          ) : (
            <div className="mt-4 overflow-x-auto rounded-2xl border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.9)] p-4">
              <div className="flex min-w-[760px] items-end gap-3 pb-2 pt-8">
                {buckets.map((bucket, index) => {
                  const height =
                    maxBucketCount > 0 ? Math.max(14, Math.round((bucket.count / maxBucketCount) * 220)) : 14;
                  const dominantSource = bucket.source_counts[0];
                  const isPeak = bucket.key === peakBucketKey;

                  return (
                    <a
                      key={bucket.key}
                      href={buildCorpusHref(bucket, { q, org, source, topic, keyword })}
                      className="group flex min-w-[72px] flex-1 flex-col items-center text-center no-underline"
                      title={`${bucket.label}: ${bucket.count} document(s)`}
                    >
                      <span
                        className={`mb-2 text-xs font-semibold ${
                          isPeak ? "text-[color:var(--amber)]" : "text-[color:var(--ink-faint)]"
                        }`}
                      >
                        {fmt(bucket.count)}
                      </span>
                      <div className="flex h-[236px] items-end">
                        <div
                          className={`w-10 rounded-t-2xl border transition duration-150 group-hover:-translate-y-1 group-hover:border-[color:var(--accent)] ${
                            isPeak
                              ? "border-[color:rgba(240,155,61,0.58)] bg-[linear-gradient(180deg,rgba(240,155,61,0.92),rgba(158,94,34,0.94))]"
                              : "border-[color:rgba(79,213,255,0.34)] bg-[linear-gradient(180deg,rgba(79,213,255,0.92),rgba(26,74,112,0.96))]"
                          }`}
                          style={{ height: `${height}px` }}
                        />
                      </div>
                      <span className="mt-3 text-[11px] uppercase tracking-[0.12em] text-[color:var(--ink-faint)]">
                        {shouldShowBucketLabel(index, buckets.length) ? bucket.label : "\u00a0"}
                      </span>
                      <span className="mt-1 text-[11px] text-[color:var(--ink-faint)]">
                        {dominantSource ? displaySourceKind(dominantSource.source_kind) : "No source"}
                      </span>
                    </a>
                  );
                })}
              </div>
            </div>
          )}
        </article>

        <article className="panel p-5">
          <h2 className="text-xl font-semibold">Timeline Readout</h2>
          {loading ? (
            <p className="mt-3 text-sm">Loading summary...</p>
          ) : buckets.length === 0 ? (
            <p className="mt-3 text-sm">Apply a topic, keyword, or date range to generate a time series.</p>
          ) : (
            <div className="mt-3 space-y-3 text-sm">
              <div className="rounded-2xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.84)] px-4 py-3">
                <p className="text-xs uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Peak Period</p>
                <p className="mt-1 text-lg font-semibold">{peakBucket?.label || "-"}</p>
                <p className="mt-1 text-[color:var(--ink-soft)]">{fmt(data?.totals.peak_bucket_count || 0)} documents</p>
              </div>
              <div className="rounded-2xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.84)] px-4 py-3">
                <p className="text-xs uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Latest Bucket</p>
                <p className="mt-1 text-lg font-semibold">{buckets[buckets.length - 1]?.label || "-"}</p>
                <p className="mt-1 text-[color:var(--ink-soft)]">
                  {fmt(buckets[buckets.length - 1]?.count || 0)} documents
                </p>
              </div>
              <div className="rounded-2xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.84)] px-4 py-3">
                <p className="text-xs uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Filter State</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {topic ? <span className="tone-chip">Topic: {topic}</span> : null}
                  {keyword ? <span className="tone-chip">Keyword: {keyword}</span> : null}
                  {q ? <span className="tone-chip">Search: {q}</span> : null}
                  {org ? <span className="tone-chip">Org: {org}</span> : null}
                  {source ? <span className="tone-chip">Source: {displaySourceKind(source)}</span> : null}
                  {!topic && !keyword && !q && !org && !source ? (
                    <span className="text-[color:var(--ink-faint)]">All corpus documents</span>
                  ) : null}
                </div>
              </div>
            </div>
          )}
        </article>
      </section>

      <section className="panel reveal reveal-delay-3 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold">Bucket Breakdown</h2>
            <p className="mt-1 text-sm text-[color:var(--ink-soft)]">
              Each row can be opened in the main corpus explorer for document-level review.
            </p>
          </div>
        </div>

        <div className="feed-table-wrap mt-4">
          <table className="feed-table">
            <thead>
              <tr>
                <th>Bucket</th>
                <th>Window</th>
                <th>Documents</th>
                <th>Dominant Source</th>
                <th>Drill Down</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={5} className="text-sm">
                    Loading buckets...
                  </td>
                </tr>
              ) : buckets.length === 0 ? (
                <tr>
                  <td colSpan={5} className="text-sm">
                    No buckets matched the current filters.
                  </td>
                </tr>
              ) : (
                buckets.map((bucket) => {
                  const dominantSource = bucket.source_counts[0];
                  return (
                    <tr key={bucket.key}>
                      <td className="text-xs font-semibold">{bucket.label}</td>
                      <td className="text-xs">
                        {fmtDateOnly(bucket.start)} to {fmtDateOnly(bucket.end)}
                      </td>
                      <td className="text-xs">{fmt(bucket.count)}</td>
                      <td className="text-xs">
                        {dominantSource ? `${displaySourceKind(dominantSource.source_kind)} (${fmt(dominantSource.count)})` : "-"}
                      </td>
                      <td className="text-xs">
                        <a href={buildCorpusHref(bucket, { q, org, source, topic, keyword })} className="link-inline">
                          Open matching docs
                        </a>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
