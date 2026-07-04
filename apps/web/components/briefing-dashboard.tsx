"use client";

import { useMemo, useState } from "react";
import type { DocumentsFacets } from "@/lib/server/types";

type BriefingStyle = "executive" | "compliance" | "analyst" | "digest";

type BriefingSource = {
  document_id: string;
  title: string;
  organization: string;
  source_kind: string;
  doc_type: string;
  published_at: string;
  url: string;
  summary: string;
  topics: string[];
  keywords: string[];
};

type BriefingReport = {
  id: string;
  generated_at: string;
  title: string;
  comparison_window: { date_from: string; date_to: string };
  metrics: {
    current_document_count: number;
    previous_document_count: number;
    delta: number;
    agency_count: number;
    topic_count: number;
    source_kind_count: number;
  };
  executive_summary: string[];
  changed_topics: Array<{ label: string; count: number; previous_count: number; delta: number }>;
  agency_activity: Array<{ label: string; count: number }>;
  source_type_activity: Array<{ label: string; count: number }>;
  topic_sections: Array<{
    label: string;
    document_count: number;
    previous_count: number;
    delta: number;
    risk_level: "low" | "medium" | "high";
    why_it_matters: string;
    sources: BriefingSource[];
  }>;
  source_appendix: BriefingSource[];
  empty: boolean;
};

type Preset = {
  name: string;
  agencies: string[];
  topics: string[];
  sourceKinds: string[];
  style: BriefingStyle;
};

const PRESETS: Preset[] = [
  {
    name: "All Regulatory Activity",
    agencies: [],
    topics: [],
    sourceKinds: [],
    style: "executive"
  },
  {
    name: "SEC Enforcement",
    agencies: ["SEC"],
    topics: ["Enforcement"],
    sourceKinds: ["sec_enforcement_litigation"],
    style: "compliance"
  },
  {
    name: "Crypto & Digital Assets",
    agencies: [],
    topics: ["Crypto & Digital Assets", "Digital Assets", "Crypto"],
    sourceKinds: [],
    style: "analyst"
  },
  {
    name: "AI Governance",
    agencies: [],
    topics: ["Artificial Intelligence", "AI", "Technology"],
    sourceKinds: [],
    style: "compliance"
  },
  {
    name: "Rulemakings & Comments",
    agencies: [],
    topics: [],
    sourceKinds: ["sec_rule_release", "sec_rule_comment", "finra_regulatory_notice", "finra_comment_letter"],
    style: "digest"
  }
];

function isoDaysAgo(days: number): string {
  const date = new Date();
  date.setDate(date.getDate() - days);
  return date.toISOString().slice(0, 10);
}

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

function toggleValue(values: string[], value: string): string[] {
  return values.includes(value) ? values.filter((item) => item !== value) : [...values, value];
}

function sourceKindLabel(value: string): string {
  const labels: Record<string, string> = {
    sec_speech: "SEC Speeches & Statements",
    sec_tm_faq: "SEC Trading & Markets FAQ",
    sec_enforcement_litigation: "SEC Enforcement Litigation",
    sec_press_release_rss: "SEC Press Releases",
    sec_administrative_proceeding: "SEC Administrative Proceedings",
    sec_trading_suspension: "SEC Trading Suspensions",
    sec_federal_register: "SEC Federal Register",
    sec_pcaob_rulemaking: "SEC PCAOB Rulemaking",
    finra_regulatory_notice: "FINRA Regulatory Notices",
    finra_comment_letter: "FINRA Comment Letters",
    finra_awc: "FINRA AWC Disciplinary Actions",
    doj_usao_press_release: "DOJ USAO Press Releases",
    federal_reserve_speech_testimony: "Federal Reserve Speeches/Testimony",
    cisa_cybersecurity_advisory: "CISA Cybersecurity Advisories",
    cftc_press_release: "CFTC Press Releases",
    cftc_public_statement_remark: "CFTC Public Statements & Remarks",
    pcaob_update: "PCAOB Updates",
    msrb_press_release: "MSRB Press Releases",
    treasury_featured_story: "Treasury Featured Stories",
    treasury_press_release: "Treasury Press Releases",
    treasury_statement_remark: "Treasury Statements & Remarks",
    sifma_news_item: "SIFMA",
    ici_news_item: "ICI",
    isda_news_item: "ISDA",
    mfa_news_item: "Managed Funds Association",
    fia_news_item: "FIA",
    aba_news_item: "American Bankers Association",
    bpi_news_item: "Bank Policy Institute",
    icba_news_item: "ICBA",
    lsta_news_item: "LSTA",
    bloomberg_public_article: "Bloomberg",
    bloomberg_apify_article: "Bloomberg",
    substack_public_article: "Substack",
    jdsupra_article: "JD Supra",
    investmentnews_article: "InvestmentNews",
    citywire_article: "Citywire",
    therecord_media_article: "The Record",
    krebs_on_security_article: "Krebs on Security",
    the_hacker_news_article: "The Hacker News",
    welivesecurity_article: "WeLiveSecurity",
    sophos_security_operations_article: "Sophos Security Operations",
    flashpoint_blog_article: "Flashpoint",
    recorded_future_article: "Recorded Future",
    intel471_blog_article: "Intel 471",
    securityweek_article: "SecurityWeek",
    dark_reading_article: "Dark Reading",
    wired_article: "WIRED",
    tripwire_article: "Tripwire",
    akamai_blog_article: "Akamai Blog",
    ritholtz_article: "The Big Picture",
    ft_portfolios_market_commentary: "First Trust Market Commentary",
    liberty_street_economics_article: "Liberty Street Economics",
    wealth_of_common_sense_article: "A Wealth of Common Sense",
    congress_crs_product: "Congress CRS Products",
    senate_committee_site: "Senate Committee Sites",
    wsj_dow_jones: "WSJ / Dow Jones",
    reddit_post: "Reddit",
    hedge_fund_letter: "Hedge Fund Letters",
    newsapi_article: "News",
    uploaded: "Uploaded"
  };
  return labels[value] || value.replace(/[_-]+/g, " ").replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function MultiSelectGroup({
  label,
  values,
  selected,
  onChange,
  formatLabel = (value) => value,
  maxVisible = 24
}: {
  label: string;
  values: string[];
  selected: string[];
  onChange: (next: string[]) => void;
  formatLabel?: (value: string) => string;
  maxVisible?: number;
}) {
  const visible = values.slice(0, maxVisible);
  const allVisibleSelected = visible.length > 0 && visible.every((value) => selected.includes(value));

  return (
    <section className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(8,18,30,0.5)] p-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-[color:var(--ink)]">{label}</h3>
          <p className="text-xs text-[color:var(--ink-faint)]">{selected.length ? `${selected.length} selected` : "All included"}</p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => onChange(allVisibleSelected ? [] : visible)}
            className="rounded-lg border border-[color:var(--line)] px-2.5 py-1.5 text-xs font-semibold text-[color:var(--ink-soft)] hover:border-[color:var(--line-strong)] hover:text-[color:var(--ink)]"
          >
            {allVisibleSelected ? "Clear" : "Select all"}
          </button>
          <button
            type="button"
            onClick={() => onChange([])}
            className="rounded-lg border border-transparent px-2.5 py-1.5 text-xs font-semibold text-[color:var(--ink-faint)] hover:text-[color:var(--ink)]"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="mt-3 flex max-h-52 flex-wrap gap-2 overflow-auto pr-1">
        {visible.length ? visible.map((value) => {
          const active = selected.includes(value);
          return (
            <button
              key={value}
              type="button"
              onClick={() => onChange(toggleValue(selected, value))}
              className={
                active
                  ? "rounded-full border border-[color:rgba(79,213,255,0.6)] bg-[color:rgba(79,213,255,0.16)] px-3 py-1.5 text-xs font-semibold text-[color:var(--ink)]"
                  : "rounded-full border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.72)] px-3 py-1.5 text-xs font-medium text-[color:var(--ink-faint)] hover:text-[color:var(--ink)]"
              }
            >
              {formatLabel(value)}
            </button>
          );
        }) : (
          <p className="text-sm text-[color:var(--ink-faint)]">No options available from the current data source.</p>
        )}
      </div>
    </section>
  );
}

function MetricCard({ label, value, detail }: { label: string; value: string | number; detail: string }) {
  return (
    <div className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.72)] p-3">
      <p className="text-xs text-[color:var(--ink-faint)]">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-[color:var(--ink)]">{value}</p>
      <p className="mt-1 text-xs text-[color:var(--ink-faint)]">{detail}</p>
    </div>
  );
}

function SourceLink({ source }: { source: BriefingSource }) {
  const title = source.title || "Untitled source";
  const content = (
    <>
      <span className="font-medium text-[color:var(--ink)]">{title}</span>
      <span className="block text-xs text-[color:var(--ink-faint)]">
        {source.organization} | {sourceKindLabel(source.source_kind)} | {source.published_at || "undated"}
      </span>
      {source.summary ? <span className="mt-1 block text-xs leading-5 text-[color:var(--ink-soft)]">{source.summary}</span> : null}
    </>
  );

  if (!source.url) {
    return <div className="rounded-lg border border-[color:var(--line-soft)] p-3">{content}</div>;
  }

  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="block rounded-lg border border-[color:var(--line-soft)] p-3 hover:border-[color:var(--line-strong)]"
    >
      {content}
    </a>
  );
}

function RiskBadge({ level }: { level: "low" | "medium" | "high" }) {
  const className =
    level === "high"
      ? "border-[color:rgba(255,107,127,0.45)] bg-[color:rgba(255,107,127,0.12)] text-[color:var(--danger)]"
      : level === "medium"
        ? "border-[color:rgba(242,171,67,0.45)] bg-[color:rgba(242,171,67,0.12)] text-[color:var(--warn)]"
        : "border-[color:rgba(65,211,157,0.35)] bg-[color:rgba(65,211,157,0.1)] text-[color:var(--ok)]";
  return <span className={`rounded-full border px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] ${className}`}>{level}</span>;
}

function ReportView({ report }: { report: BriefingReport }) {
  return (
    <div className="space-y-4">
      <section className="panel p-5">
        <div className="flex flex-col justify-between gap-4 md:flex-row md:items-start">
          <div>
            <p className="kicker">Generated Briefing</p>
            <h2 className="mt-4 text-2xl font-semibold text-[color:var(--ink)]">{report.title}</h2>
            <p className="mt-1 text-sm text-[color:var(--ink-faint)]">
              Compared against {report.comparison_window.date_from} to {report.comparison_window.date_to}.
            </p>
          </div>
          <p className="text-xs text-[color:var(--ink-faint)]">{new Date(report.generated_at).toLocaleString()}</p>
        </div>

        <div className="mt-5 grid gap-3 md:grid-cols-4">
          <MetricCard
            label="Documents"
            value={report.metrics.current_document_count}
            detail={`${report.metrics.delta >= 0 ? "+" : ""}${report.metrics.delta} vs prior period`}
          />
          <MetricCard label="Agencies" value={report.metrics.agency_count} detail="matched selected filters" />
          <MetricCard label="Themes" value={report.metrics.topic_count} detail="detected across sources" />
          <MetricCard label="Source Types" value={report.metrics.source_kind_count} detail="represented in corpus" />
        </div>

        {report.empty ? (
          <div className="mt-5 rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.72)] p-4 text-sm text-[color:var(--ink-soft)]">
            No documents matched this briefing setup. Expand the dates or clear one of the filters.
          </div>
        ) : (
          <ul className="mt-5 space-y-2">
            {report.executive_summary.map((item) => (
              <li key={item} className="flex gap-2 text-sm leading-6 text-[color:var(--ink-soft)]">
                <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-[color:var(--accent)]" aria-hidden="true" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <div className="grid gap-4 lg:grid-cols-[1.4fr_0.8fr]">
        <section className="panel p-5">
          <h2 className="text-lg font-semibold text-[color:var(--ink)]">What Changed</h2>
          <div className="mt-4 space-y-3">
            {report.changed_topics.length ? report.changed_topics.map((topic) => (
              <div key={topic.label} className="rounded-lg border border-[color:var(--line-soft)] p-3">
                <div className="flex items-center justify-between gap-3">
                  <p className="font-semibold text-[color:var(--ink)]">{topic.label}</p>
                  <span className="text-sm text-[color:var(--ink-faint)]">
                    {topic.count} now | {topic.previous_count} prior | {topic.delta >= 0 ? "+" : ""}{topic.delta}
                  </span>
                </div>
              </div>
            )) : (
              <p className="text-sm text-[color:var(--ink-faint)]">No topic changes were detected for this selection.</p>
            )}
          </div>
        </section>

        <section className="panel p-5">
          <h2 className="text-lg font-semibold text-[color:var(--ink)]">Activity Mix</h2>
          <div className="mt-4 space-y-4">
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-[color:var(--ink-faint)]">Agencies</h3>
              <div className="mt-2 space-y-2">
                {report.agency_activity.map((entry) => (
                  <div key={entry.label} className="flex justify-between gap-3 text-sm">
                    <span className="text-[color:var(--ink-soft)]">{entry.label}</span>
                    <span className="font-semibold text-[color:var(--ink)]">{entry.count}</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-[color:var(--ink-faint)]">Source Types</h3>
              <div className="mt-2 space-y-2">
                {report.source_type_activity.map((entry) => (
                  <div key={entry.label} className="flex justify-between gap-3 text-sm">
                    <span className="text-[color:var(--ink-soft)]">{entry.label}</span>
                    <span className="font-semibold text-[color:var(--ink)]">{entry.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      </div>

      <section className="panel p-5">
        <h2 className="text-lg font-semibold text-[color:var(--ink)]">Topic Sections</h2>
        <div className="mt-4 space-y-4">
          {report.topic_sections.length ? report.topic_sections.map((section) => (
            <article key={section.label} className="rounded-lg border border-[color:var(--line-soft)] p-4">
              <div className="flex flex-col justify-between gap-3 md:flex-row md:items-start">
                <div>
                  <h3 className="text-base font-semibold text-[color:var(--ink)]">{section.label}</h3>
                  <p className="mt-1 text-sm text-[color:var(--ink-soft)]">{section.why_it_matters}</p>
                </div>
                <div className="flex items-center gap-2">
                  <RiskBadge level={section.risk_level} />
                  <span className="text-xs text-[color:var(--ink-faint)]">
                    {section.document_count} docs | {section.delta >= 0 ? "+" : ""}{section.delta}
                  </span>
                </div>
              </div>
              <div className="mt-4 grid gap-3 lg:grid-cols-2">
                {section.sources.map((source) => <SourceLink key={source.document_id} source={source} />)}
              </div>
            </article>
          )) : (
            <p className="text-sm text-[color:var(--ink-faint)]">No topic sections were generated.</p>
          )}
        </div>
      </section>

      <section className="panel p-5">
        <h2 className="text-lg font-semibold text-[color:var(--ink)]">Source Appendix</h2>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {report.source_appendix.map((source) => <SourceLink key={source.document_id} source={source} />)}
        </div>
      </section>
    </div>
  );
}

export function BriefingDashboard({ facets }: { facets: DocumentsFacets }) {
  const [dateFrom, setDateFrom] = useState(isoDaysAgo(7));
  const [dateTo, setDateTo] = useState(todayIso());
  const [agencies, setAgencies] = useState<string[]>([]);
  const [topics, setTopics] = useState<string[]>([]);
  const [sourceKinds, setSourceKinds] = useState<string[]>([]);
  const [entitiesText, setEntitiesText] = useState("");
  const [style, setStyle] = useState<BriefingStyle>("executive");
  const [report, setReport] = useState<BriefingReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const topicOptions = useMemo(() => {
    const keyTopics = facets.key_topics || [];
    const rest = (facets.topics || []).filter((topic) => !keyTopics.includes(topic));
    return [...keyTopics, ...rest];
  }, [facets.key_topics, facets.topics]);

  const applyPreset = (preset: Preset) => {
    setAgencies(preset.agencies.filter((agency) => facets.organizations.includes(agency)));
    setTopics(preset.topics.filter((topic) => topicOptions.some((option) => option.toLowerCase().includes(topic.toLowerCase()) || topic.toLowerCase().includes(option.toLowerCase()))));
    setSourceKinds(preset.sourceKinds.filter((sourceKind) => facets.sources.includes(sourceKind)));
    setStyle(preset.style);
  };

  const generate = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/briefings/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          date_from: dateFrom,
          date_to: dateTo,
          agencies,
          topics,
          source_kinds: sourceKinds,
          entities: entitiesText.split(",").map((item) => item.trim()).filter(Boolean),
          style
        })
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok || !json.ok) {
        setError(json.error || `Briefing generation failed (${res.status})`);
        return;
      }
      setReport(json.data as BriefingReport);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Network error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-5">
      <section className="panel hero-panel overflow-hidden p-5 md:p-6">
        <div className="relative z-10 max-w-4xl">
          <p className="kicker">Briefings</p>
          <h1 className="mt-4 text-3xl font-semibold text-[color:var(--ink)] md:text-4xl">Custom regulatory briefing report</h1>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-[color:var(--ink-soft)] md:text-base">
            Choose the dates, agencies, topics, source types, and watchlist terms. The report compares the selected period
            against the prior comparable period and turns the corpus into a concise briefing with source evidence.
          </p>
        </div>
      </section>

      <div className="grid gap-5 lg:grid-cols-[0.9fr_1.4fr]">
        <aside className="panel h-fit p-4">
          <div className="flex flex-col gap-3">
            <div>
              <h2 className="text-lg font-semibold text-[color:var(--ink)]">Briefing Setup</h2>
              <p className="mt-1 text-sm text-[color:var(--ink-faint)]">Leave a filter empty to include all values.</p>
            </div>

            <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
              <label className="block text-sm font-medium text-[color:var(--ink-soft)]">
                From
                <input className="form-control mt-1 w-full px-3 py-2" type="date" value={dateFrom} onChange={(event) => setDateFrom(event.target.value)} />
              </label>
              <label className="block text-sm font-medium text-[color:var(--ink-soft)]">
                To
                <input className="form-control mt-1 w-full px-3 py-2" type="date" value={dateTo} onChange={(event) => setDateTo(event.target.value)} />
              </label>
            </div>

            <label className="block text-sm font-medium text-[color:var(--ink-soft)]">
              Output style
              <select className="form-control mt-1 w-full px-3 py-2" value={style} onChange={(event) => setStyle(event.target.value as BriefingStyle)}>
                <option value="executive">Executive brief</option>
                <option value="compliance">Compliance memo</option>
                <option value="analyst">Analyst note</option>
                <option value="digest">Bullet digest</option>
              </select>
            </label>

            <section className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(8,18,30,0.5)] p-3">
              <h3 className="text-sm font-semibold text-[color:var(--ink)]">Presets</h3>
              <div className="mt-3 flex flex-wrap gap-2">
                {PRESETS.map((preset) => (
                  <button
                    key={preset.name}
                    type="button"
                    onClick={() => applyPreset(preset)}
                    className="rounded-full border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.72)] px-3 py-1.5 text-xs font-semibold text-[color:var(--ink-soft)] hover:border-[color:var(--line-strong)] hover:text-[color:var(--ink)]"
                  >
                    {preset.name}
                  </button>
                ))}
              </div>
            </section>

            <MultiSelectGroup label="Agencies" values={facets.organizations || []} selected={agencies} onChange={setAgencies} />
            <MultiSelectGroup label="Topics" values={topicOptions} selected={topics} onChange={setTopics} />
            <MultiSelectGroup label="Source Types" values={facets.sources || []} selected={sourceKinds} onChange={setSourceKinds} formatLabel={sourceKindLabel} />

            <label className="block text-sm font-medium text-[color:var(--ink-soft)]">
              Entity or ticker watchlist
              <textarea
                className="form-control mt-1 min-h-20 w-full px-3 py-2"
                value={entitiesText}
                onChange={(event) => setEntitiesText(event.target.value)}
                placeholder="Coinbase, private credit, cybersecurity"
              />
              <span className="mt-1 block text-xs text-[color:var(--ink-faint)]">Comma-separated terms matched against document text and metadata.</span>
            </label>

            <button
              type="button"
              onClick={generate}
              disabled={loading}
              className="min-h-11 rounded-lg border border-[color:rgba(79,213,255,0.55)] bg-[color:rgba(79,213,255,0.14)] px-4 py-2 text-sm font-semibold text-[color:var(--ink)] hover:bg-[color:rgba(79,213,255,0.2)] disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? "Generating..." : "Generate briefing"}
            </button>

            {error ? <p className="rounded-lg border border-[color:rgba(255,107,127,0.35)] bg-[color:rgba(255,107,127,0.1)] p-3 text-sm text-[color:var(--danger)]">{error}</p> : null}
          </div>
        </aside>

        <div>
          {report ? (
            <ReportView report={report} />
          ) : (
            <section className="panel p-6">
              <h2 className="text-xl font-semibold text-[color:var(--ink)]">Ready to generate</h2>
              <p className="mt-2 text-sm leading-6 text-[color:var(--ink-soft)]">
                Start with a preset or build a custom selection. The first version is evidence-first and deterministic:
                it finds matching corpus items, compares them with the prior period, and organizes the output into a
                briefing shell we can later upgrade with LLM narrative generation, saved presets, email delivery, and PDF export.
              </p>
              <div className="mt-5 grid gap-3 md:grid-cols-3">
                <MetricCard label="Available Agencies" value={(facets.organizations || []).length} detail="from corpus facets" />
                <MetricCard label="Available Topics" value={(facets.topics || []).length} detail="from enrichment tags" />
                <MetricCard label="Source Types" value={(facets.sources || []).length} detail="document classes" />
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
