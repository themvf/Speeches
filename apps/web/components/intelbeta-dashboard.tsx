"use client";

import { Fragment, useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import type { StoredRssArticle, StoredRssTopicRule } from "@/lib/server/neon";
import type { DocumentListItem } from "@/lib/server/types";
import { BookmarkButton } from "@/components/bookmark-button";
import { useSavedItems } from "@/hooks/use-saved-items";
import {
  decodeEntities,
  getMatchingTopics,
  normalizeMatchText,
  normalizeTopicRules,
  type TopicRuleView,
} from "@/lib/intel-topic-matching";

type TopicFilter = string | "ALL";
type SourceFilter =
  | "ALL"
  | "SEC_SPEECHES"
  | "SEC_ENFORCEMENT"
  | "SEC_ALL"
  | "FINRA"
  | "DOJ"
  | "FED"
  | "CFTC"
  | "TREASURY"
  | "MARKET_SOURCES"
  | "CONGRESS"
  | "PONZI_INVESTOR_FRAUD"
  | "CYBER"
  | "WSJ"
  | "BLOOMBERG"
  | "SUBSTACK"
  | "NEWSAPI"
  | "SIFMA"
  | "TRADE_ASSOCIATIONS"
  | "LEGAL_SOURCES"
  | "TRADE_MEDIA"
  | "REDDIT";

const FEED_RENDER_BATCH_SIZE = 20;
const LIVE_FEED_REFRESH_LIMIT = 500;
const LIVE_FEED_POLL_INTERVAL_MS = 60 * 60_000;

type FeedMeta = {
  label: string;
  code: string;
  color: string;
};

type FeedItem = StoredRssArticle & {
  item_type?: "article" | "document";
  document_id?: string;
  organization?: string;
  source_kind?: string;
  doc_type?: string;
  enrichment_model?: string;
  matched_finra_firms?: string[];
  topics?: string[];
  keywords?: string[];
  analysis?: unknown;
};

interface ApiEnvelope<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

interface DocumentDetailData {
  metadata: {
    document_id: string;
    published_at: string;
  };
  content: {
    full_text: string;
    paragraphs: string[];
    sentences: string[];
  };
  enrichment: {
    status: string;
    model: string;
    summary: string;
    tags: string[];
    keywords: string[];
    entities: string[];
    evidence_spans: Array<Record<string, unknown>>;
    stance: Record<string, unknown>;
    comment_position: Record<string, unknown>;
    confidence: number;
  };
  review: {
    decision: string;
    notes: string;
    reviewed_at: string;
  };
  sentiment: {
    score: number;
    label: string;
    rationale: string;
    status: string;
  } | null;
}

interface FeedItemAnalysis {
  thesis: string;
  why_it_matters: string[];
  risk_signals: string[];
  follow_up_questions: string[];
  keywords: string[];
  individuals: string[];
  entities: string[];
  model: string;
  generated_at: string;
  fallback: boolean;
}

function toFeedItemAnalysis(value: unknown): FeedItemAnalysis | undefined {
  if (!value || typeof value !== "object") return undefined;
  const src = value as Partial<FeedItemAnalysis>;
  const thesis = String(src.thesis || "").trim();
  if (!thesis) return undefined;
  return {
    thesis,
    why_it_matters: Array.isArray(src.why_it_matters) ? src.why_it_matters.map(String).filter(Boolean) : [],
    risk_signals: Array.isArray(src.risk_signals) ? src.risk_signals.map(String).filter(Boolean) : [],
    follow_up_questions: Array.isArray(src.follow_up_questions) ? src.follow_up_questions.map(String).filter(Boolean) : [],
    keywords: Array.isArray(src.keywords) ? src.keywords.map(String).filter(Boolean) : [],
    individuals: Array.isArray(src.individuals) ? src.individuals.map(String).filter(Boolean) : [],
    entities: Array.isArray(src.entities) ? src.entities.map(String).filter(Boolean) : [],
    model: String(src.model || ""),
    generated_at: String(src.generated_at || ""),
    fallback: Boolean(src.fallback),
  };
}

const FEED_META: Record<string, FeedMeta> = {
  sec_press_releases: { label: "SEC Press Releases", code: "SEC", color: "#7cc4ff" },
  sec_speeches_statements: { label: "SEC Speeches and Statements", code: "SEC", color: "#7cc4ff" },
  document_sec_speech: { label: "SEC Speeches", code: "SEC", color: "#7cc4ff" },
  document_sec_enforcement_litigation: { label: "SEC Enforcement", code: "SEC", color: "#ff8aa0" },
  document_sec_administrative_proceeding: { label: "SEC Administrative Proceedings", code: "SEC", color: "#ff8aa0" },
  document_sec_trading_suspension: { label: "SEC Trading Suspensions", code: "SEC", color: "#ff8aa0" },
  document_sec_press_release_rss: { label: "SEC Press Releases", code: "SEC", color: "#7cc4ff" },
  document_sec_federal_register: { label: "SEC Federal Register", code: "SEC", color: "#7cc4ff" },
  document_sec_pcaob_rulemaking: { label: "SEC PCAOB Rulemaking", code: "SEC", color: "#7cc4ff" },
  sec_litigation_releases: { label: "SEC Litigation Releases", code: "SEC", color: "#ff8aa0" },
  sec_administrative_proceedings: { label: "SEC Administrative Proceedings", code: "SEC", color: "#ff8aa0" },
  sec_trading_suspensions: { label: "SEC Trading Suspensions", code: "SEC", color: "#ff8aa0" },
  finra_notices: { label: "FINRA Regulatory Notices", code: "FINRA", color: "#77d7a8" },
  finra_rule_filings: { label: "FINRA Rule Filings", code: "FINRA", color: "#77d7a8" },
  finra_dispute_resolution_rule_filings: { label: "FINRA Dispute Resolution Rule Filings", code: "FINRA", color: "#77d7a8" },
  finra_news: { label: "FINRA News Releases and Speeches", code: "FINRA", color: "#77d7a8" },
  finra_upc_advisories: { label: "FINRA UPC Advisories", code: "FINRA", color: "#77d7a8" },
  wsj_us_business: { label: "WSJ Business", code: "WSJB", color: "#63a8ff" },
  wsj_markets: { label: "WSJ Markets", code: "WSJM", color: "#ffc857" },
  wsj_opinion: { label: "WSJ Opinion", code: "WSJO", color: "#b88fff" },
  mw_top_stories: { label: "MarketWatch", code: "MW", color: "#4dd39f" },
  rss_nytimes_com_services_xml_rss_nyt_business_xml: { label: "NYT Business", code: "NYTB", color: "#ffe066" },
  rss_nytimes_com_services_xml_rss_nyt_technology_xml: { label: "NYT Tech", code: "NYTT", color: "#74c0fc" },
  rss_nytimes_com_services_xml_rss_nyt_politics_xml: { label: "NYT Politics", code: "NYTP", color: "#ff8787" },
  coindesk: { label: "CoinDesk", code: "CDSK", color: "#f0b90b" },
  cointelegraph: { label: "Cointelegraph", code: "CTEL", color: "#f3c969" },
  decrypt: { label: "Decrypt", code: "DECR", color: "#8ce99a" },
  the_block: { label: "The Block", code: "BLCK", color: "#63e6be" },
  cisa_cybersecurity_advisories: { label: "CISA", code: "CISA", color: "#4dabf7" },
  bleepingcomputer: { label: "BleepingComputer", code: "BLC", color: "#74c0fc" },
  krebs_on_security: { label: "Krebs on Security", code: "KRBS", color: "#ffa8a8" },
  the_hacker_news: { label: "The Hacker News", code: "THN", color: "#ff8787" },
  welivesecurity: { label: "WeLiveSecurity", code: "ESET", color: "#20c997" },
  sophos_security_operations: { label: "Sophos Security Operations", code: "SOPH", color: "#38d9a9" },
  flashpoint_blog: { label: "Flashpoint", code: "FLPT", color: "#b197fc" },
  recorded_future: { label: "Recorded Future", code: "RF", color: "#91a7ff" },
  intel471_blog: { label: "Intel 471", code: "I471", color: "#f783ac" },
  prnewswire_all: { label: "PR Newswire", code: "PRN", color: "#66d9e8" },
  prnewswire_consumer_technology: { label: "PR Newswire Consumer Technology", code: "PRN", color: "#66d9e8" },
  prnewswire_financial_services: { label: "PR Newswire Financial Services", code: "PRN", color: "#66d9e8" },
  prnewswire_policy_public_interest: { label: "PR Newswire Policy & Public Interest", code: "PRN", color: "#66d9e8" },
  google_news_ponzi_investor_fraud: { label: "Google News: Ponzi & Investor Fraud", code: "GNEWS", color: "#8ce99a" },
  google_news_finra_member_firms: { label: "Google News: FINRA Member Firms", code: "GNEWS", color: "#d0bfff" },
  google_news_senate_banking_committee: { label: "Google News: Senate Banking Committee", code: "SEN", color: "#b88fff" },
  google_news_senate_finance_committee: { label: "Google News: Senate Finance Committee", code: "SEN", color: "#b88fff" },
  google_news_senate_agriculture_committee: { label: "Google News: Senate Agriculture Committee", code: "SEN", color: "#b88fff" },
  google_news_senate_judiciary_committee: { label: "Google News: Senate Judiciary Committee", code: "SEN", color: "#b88fff" },
  google_news_senate_hsgac: { label: "Google News: Senate Homeland Security Committee", code: "SEN", color: "#b88fff" },
  google_news_senate_commerce_committee: { label: "Google News: Senate Commerce Committee", code: "SEN", color: "#b88fff" },
  cftc_general_press_releases: { label: "CFTC General Press Releases", code: "CFTC", color: "#ffd43b" },
  cftc_enforcement_press_releases: { label: "CFTC Enforcement Press Releases", code: "CFTC", color: "#ff8aa0" },
  cftc_speeches_testimony: { label: "CFTC Speeches and Testimony", code: "CFTC", color: "#ffd43b" },
  fed_all_press_releases: { label: "Federal Reserve", code: "FED", color: "#91a7ff" },
  occ_news_releases: { label: "OCC News Releases", code: "OCC", color: "#a5d8ff" },
  ftc_consumer_protection_press_releases: { label: "FTC Consumer Protection", code: "FTC", color: "#ffd43b" },
  search_cnbc_com_rs_search_combinedcms_view_xml: { label: "CNBC", code: "CNBC", color: "#4dabf7" },
  rss_nytimes_com_services_xml_rss_nyt_dealbook_xml: { label: "NYT DealBook", code: "NYT", color: "#ffe066" },
  rss_nytimes_com_services_xml_rss_nyt_economy_xml: { label: "NYT Economy", code: "NYT", color: "#ffe066" },
  ft_news_feed: { label: "Financial Times", code: "FT", color: "#ffc857" },
  ft_markets: { label: "Financial Times Markets", code: "FT", color: "#ffc857" },
  ft_financials: { label: "Financial Times Financials", code: "FT", color: "#ffc857" },
  economist_finance_economics: { label: "The Economist Finance & Economics", code: "ECO", color: "#74c0fc" },
  economist_business: { label: "The Economist Business", code: "ECO", color: "#74c0fc" },
  economist_united_states: { label: "The Economist United States", code: "ECO", color: "#74c0fc" },
  investmentnews: { label: "InvestmentNews", code: "INV", color: "#66d9e8" },
  american_banker: { label: "American Banker", code: "AB", color: "#66d9e8" },
  harvard_corp_gov_forum: { label: "Harvard Corporate Governance Forum", code: "HLS", color: "#d0bfff" },
  cls_blue_sky_blog: { label: "CLS Blue Sky Blog", code: "CLS", color: "#d0bfff" },
  the_corporate_counsel_net: { label: "The Corporate Counsel", code: "TCC", color: "#d0bfff" },
  www_centralbanking_com_feeds_rss_category_central_banks_fina: { label: "Central Banking", code: "CB", color: "#91a7ff" },
  ballard_spahr_consumer_finance_monitor: { label: "Ballard Spahr Consumer Finance Monitor", code: "BS", color: "#a5d8ff" },
  cooley_governance_beat: { label: "Cooley Governance Beat", code: "COO", color: "#a5d8ff" },
  covington_global_policy_watch: { label: "Covington Global Policy Watch", code: "COV", color: "#a5d8ff" },
  covington_inside_privacy: { label: "Covington Inside Privacy", code: "COV", color: "#a5d8ff" },
  kelley_drye_ad_law_access: { label: "Kelley Drye Ad Law Access", code: "KD", color: "#a5d8ff" },
  latham_global_financial_regulatory_blog: { label: "Latham Global Financial Regulatory Blog", code: "LW", color: "#a5d8ff" },
  bradley_financial_services_perspectives: { label: "Bradley Financial Services Perspectives", code: "BRD", color: "#a5d8ff" },
  bradley_eye_on_enforcement: { label: "Bradley Eye on Enforcement", code: "BRD", color: "#a5d8ff" },
  gibson_dunn_securities_regulation_monitor: { label: "Gibson Dunn Securities Regulation Monitor", code: "GD", color: "#a5d8ff" },
  the_record: { label: "The Record", code: "REC", color: "#ff922b" },
  wired_security: { label: "WIRED Security", code: "WRD", color: "#f783ac" },
  ritholtz_big_picture: { label: "The Big Picture", code: "RIT", color: "#ffd43b" },
  ft_portfolios_market_commentary: { label: "First Trust Market Commentary", code: "FT", color: "#b197fc" },
  wealth_of_common_sense: { label: "A Wealth of Common Sense", code: "AWC", color: "#ffc078" },
  dark_reading: { label: "Dark Reading", code: "DARK", color: "#b197fc" },
  securityweek: { label: "SecurityWeek", code: "SECW", color: "#91a7ff" },
  microsoft_security_blog: { label: "Microsoft Security Blog", code: "MSFT", color: "#69db7c" },
  document_bloomberg_apify_article: { label: "Bloomberg", code: "BBG", color: "#ffb703" },
  document_bloomberg_public_article: { label: "Bloomberg", code: "BBG", color: "#ffb703" },
  document_substack_public_article: { label: "Substack", code: "SUB", color: "#f26b38" },
  document_newsapi_article: { label: "NewsAPI", code: "NEWS", color: "#69db7c" },
  document_sifma_news_item: { label: "SIFMA", code: "SIFMA", color: "#a5d8ff" },
  document_ici_news_item: { label: "ICI", code: "ICI", color: "#a5d8ff" },
  document_isda_news_item: { label: "ISDA", code: "ISDA", color: "#a5d8ff" },
  document_mfa_news_item: { label: "Managed Funds Association", code: "MFA", color: "#a5d8ff" },
  document_fia_news_item: { label: "FIA", code: "FIA", color: "#a5d8ff" },
  document_aba_news_item: { label: "American Bankers Association", code: "ABA", color: "#a5d8ff" },
  document_bpi_news_item: { label: "Bank Policy Institute", code: "BPI", color: "#a5d8ff" },
  document_icba_news_item: { label: "ICBA", code: "ICBA", color: "#a5d8ff" },
  document_lsta_news_item: { label: "LSTA", code: "LSTA", color: "#a5d8ff" },
  document_federal_reserve_speech_testimony: { label: "Federal Reserve", code: "FED", color: "#91a7ff" },
  document_cisa_cybersecurity_advisory: { label: "CISA", code: "CISA", color: "#4dabf7" },
  document_treasury_featured_story: { label: "Treasury", code: "TRE", color: "#ffd43b" },
  document_treasury_press_release: { label: "Treasury", code: "TRE", color: "#ffd43b" },
  document_treasury_statement_remark: { label: "Treasury", code: "TRE", color: "#ffd43b" },
  document_cftc_press_release: { label: "CFTC", code: "CFTC", color: "#ffd43b" },
  document_cftc_public_statement_remark: { label: "CFTC", code: "CFTC", color: "#ffd43b" },
  document_pcaob_update: { label: "PCAOB", code: "PCAOB", color: "#a5d8ff" },
  document_msrb_press_release: { label: "MSRB", code: "MSRB", color: "#a5d8ff" },
  document_congress_crs_product: { label: "Congress CRS", code: "CRS", color: "#b88fff" },
  document_senate_committee_site: { label: "Senate Committee Sites", code: "SEN", color: "#b88fff" },
  document_sec_tm_faq: { label: "SEC TM FAQ", code: "SEC", color: "#7cc4ff" },
  document_finra_regulatory_notice: { label: "FINRA Regulatory Notices", code: "FINRA", color: "#77d7a8" },
  document_finra_awc: { label: "FINRA AWC", code: "FINRA", color: "#77d7a8" },
  document_jdsupra_article: { label: "JD Supra", code: "JDS", color: "#ffa94d" },
  document_investmentnews_article: { label: "InvestmentNews", code: "INV", color: "#66d9e8" },
  document_citywire_article: { label: "Citywire", code: "CITY", color: "#d0bfff" },
  document_therecord_media_article: { label: "The Record", code: "REC", color: "#ff922b" },
  document_krebs_on_security_article: { label: "Krebs on Security", code: "KRBS", color: "#ffa8a8" },
  document_the_hacker_news_article: { label: "The Hacker News", code: "THN", color: "#ff8787" },
  document_welivesecurity_article: { label: "WeLiveSecurity", code: "ESET", color: "#20c997" },
  document_sophos_security_operations_article: { label: "Sophos Security Operations", code: "SOPH", color: "#38d9a9" },
  document_flashpoint_blog_article: { label: "Flashpoint", code: "FLPT", color: "#b197fc" },
  document_recorded_future_article: { label: "Recorded Future", code: "RF", color: "#91a7ff" },
  document_intel471_blog_article: { label: "Intel 471", code: "I471", color: "#f783ac" },
  document_securityweek_article: { label: "SecurityWeek", code: "SECW", color: "#91a7ff" },
  document_dark_reading_article: { label: "Dark Reading", code: "DARK", color: "#b197fc" },
  document_wired_article: { label: "WIRED", code: "WRD", color: "#f783ac" },
  document_tripwire_article: { label: "Tripwire", code: "TRP", color: "#69db7c" },
  document_akamai_blog_article: { label: "Akamai Blog", code: "AKM", color: "#74c0fc" },
  document_ritholtz_article: { label: "The Big Picture", code: "RIT", color: "#ffd43b" },
  document_ft_portfolios_market_commentary: { label: "First Trust", code: "FT", color: "#b197fc" },
  document_liberty_street_economics_article: { label: "Liberty Street Economics", code: "LSE", color: "#91a7ff" },
  document_wealth_of_common_sense_article: { label: "A Wealth of Common Sense", code: "AWC", color: "#ffc078" },
  document_wsj_dow_jones: { label: "WSJ / Dow Jones", code: "WSJ", color: "#63a8ff" },
  document_reddit_post: { label: "Reddit", code: "RDDT", color: "#ff922b" },
};

const SOURCE_FILTERS: Array<{ key: Exclude<SourceFilter, "ALL">; label: string }> = [
  { key: "SEC_SPEECHES", label: "SEC Speeches" },
  { key: "SEC_ENFORCEMENT", label: "SEC Enforcement" },
  { key: "SEC_ALL", label: "SEC All" },
  { key: "FINRA", label: "FINRA" },
  { key: "DOJ", label: "DOJ" },
  { key: "FED", label: "Federal Reserve" },
  { key: "CFTC", label: "CFTC" },
  { key: "TREASURY", label: "Treasury" },
  { key: "MARKET_SOURCES", label: "Market Sources" },
  { key: "CONGRESS", label: "Congress" },
  { key: "PONZI_INVESTOR_FRAUD", label: "Ponzi & Investor Fraud" },
  { key: "CYBER", label: "Cyber" },
  { key: "WSJ", label: "WSJ" },
  { key: "BLOOMBERG", label: "Bloomberg" },
  { key: "SUBSTACK", label: "Substack" },
  { key: "NEWSAPI", label: "NewsAPI" },
  { key: "SIFMA", label: "SIFMA" },
  { key: "TRADE_ASSOCIATIONS", label: "Trade Associations" },
  { key: "LEGAL_SOURCES", label: "Legal Sources" },
  { key: "TRADE_MEDIA", label: "Trade Media" },
  { key: "REDDIT", label: "Reddit" },
];

const LEGAL_SOURCE_FEED_KEYS = new Set([
  "ballard_spahr_consumer_finance_monitor",
  "bradley_eye_on_enforcement",
  "bradley_financial_services_perspectives",
  "cleary_enforcement_watch",
  "cls_blue_sky_blog",
  "cooley_cyber_data_privacy",
  "cooley_governance_beat",
  "cooley_pubco",
  "covington_global_policy_watch",
  "covington_inside_government_contracts",
  "covington_inside_privacy",
  "gibson_dunn_sec_sentinel",
  "gibson_dunn_securities_regulation_monitor",
  "harvard_corp_gov_forum",
  "kelley_drye_ad_law_access",
  "latham_global_financial_regulatory_blog",
  "latham_london",
  "norton_rose_fulbright_data_protection_report",
  "squire_patton_boggs_privacy_world",
  "the_corporate_counsel_net",
]);

const TRADE_MEDIA_FEED_KEYS = new Set([
  "ritholtz_big_picture",
  "ft_portfolios_market_commentary",
  "wealth_of_common_sense",
]);

const CONGRESS_SOURCE_FEED_KEYS = new Set([
  "google_news_senate_banking_committee",
  "google_news_senate_finance_committee",
  "google_news_senate_agriculture_committee",
  "google_news_senate_judiciary_committee",
  "google_news_senate_hsgac",
  "google_news_senate_commerce_committee",
]);

const CYBER_SOURCE_FEED_KEYS = new Set([
  "cisa_cybersecurity_advisories",
  "bleepingcomputer",
  "krebs_on_security",
  "the_hacker_news",
  "welivesecurity",
  "sophos_security_operations",
  "flashpoint_blog",
  "recorded_future",
  "intel471_blog",
  "the_record",
  "wired_security",
  "tripwire_state_of_security",
  "akamai_blog",
  "dark_reading",
  "securityweek",
  "microsoft_security_blog",
]);

const CYBER_SOURCE_KINDS = new Set([
  "cisa_cybersecurity_advisory",
  "therecord_media_article",
  "krebs_on_security_article",
  "the_hacker_news_article",
  "welivesecurity_article",
  "sophos_security_operations_article",
  "flashpoint_blog_article",
  "recorded_future_article",
  "intel471_blog_article",
  "securityweek_article",
  "dark_reading_article",
  "wired_article",
  "tripwire_article",
  "akamai_blog_article",
]);

const SOURCE_LABEL_ACRONYMS = new Set([
  "ai",
  "api",
  "cfpb",
  "cftc",
  "cisa",
  "cls",
  "cnbc",
  "finra",
  "ft",
  "ftc",
  "ipo",
  "nyc",
  "nyt",
  "occ",
  "rss",
  "sec",
  "wsj",
]);

function cleanSourceLabel(value: string | null | undefined): string {
  return decodeEntities(value || "").replace(/\s+/g, " ").trim();
}

function labelFromFeedKey(feedKey: string): string {
  const key = String(feedKey || "").trim();
  if (!key) return "";
  return key
    .replace(/^rss_/, "")
    .replace(/^www_/, "")
    .split(/[_-]+/g)
    .map((part) => part.trim().toLowerCase())
    .filter((part) => part && !["com", "org", "net", "xml", "rss", "feed", "feeds", "services"].includes(part))
    .map((part) => SOURCE_LABEL_ACRONYMS.has(part) ? part.toUpperCase() : `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

function sourceCodeFromLabel(label: string, fallback: string): string {
  const words = label.match(/[a-z0-9]+/gi) || [];
  const acronym = words
    .filter((word) => !["the", "and", "of", "for"].includes(word.toLowerCase()))
    .map((word) => word.charAt(0))
    .join("")
    .toUpperCase()
    .slice(0, 5);
  return acronym || fallback.slice(0, 4).toUpperCase() || "SRC";
}

function getFeedMeta(feedKey: string, feedLabel?: string | null): FeedMeta {
  const cleanFeedLabel = cleanSourceLabel(feedLabel);
  const known = FEED_META[feedKey];
  if (known) {
    return cleanFeedLabel && cleanFeedLabel !== feedKey ? { ...known, label: cleanFeedLabel } : known;
  }
  if (feedKey.startsWith("document_")) {
    const sourceKind = feedKey.replace(/^document_/, "");
    const normalized = sourceKind.replace(/[_-]+/g, " ").trim();
    const code = normalized
      .split(" ")
      .map((word) => word.charAt(0))
      .join("")
      .toUpperCase()
      .slice(0, 4) || "DOC";
    return {
      label: normalized.replace(/\b\w/g, (ch) => ch.toUpperCase()) || "Document",
      code,
      color: "#4fd5ff",
    };
  }
  const label = cleanFeedLabel || labelFromFeedKey(feedKey) || feedKey || "Unknown";
  return {
    label,
    code: sourceCodeFromLabel(label, feedKey),
    color: "#8fa7c8",
  };
}

function isDeepSeekFeedAnalysis(analysis: FeedItemAnalysis | undefined): boolean {
  return String(analysis?.model || "").trim().toLowerCase().startsWith("deepseek");
}

function shouldRegenerateFeedAnalysis(analysis: FeedItemAnalysis | undefined): boolean {
  return Boolean(analysis) && (Boolean(analysis?.fallback) || !isDeepSeekFeedAnalysis(analysis));
}

function feedAnalysisModelLabel(analysis: FeedItemAnalysis | undefined): string {
  const model = String(analysis?.model || "").trim();
  if (!model) return "not recorded";
  return analysis?.fallback ? `${model} fallback` : model;
}

function feedAnalysisModelTitle(analysis: FeedItemAnalysis | undefined): string {
  const model = String(analysis?.model || "").trim();
  if (!model) return "Model was not recorded for this analysis.";
  const fallback = analysis?.fallback ? " fallback" : "";
  return `Model used: ${model}${fallback}`;
}

function hostedModelLabel(model: string, fallback = false): string {
  const normalized = String(model || "").trim();
  if (!normalized) return "not recorded";
  return fallback ? `${normalized} fallback` : normalized;
}

function hostedModelTitle(model: string, fallback = false): string {
  const normalized = String(model || "").trim();
  if (!normalized) return "Model was not recorded for this enrichment.";
  return `Model used: ${normalized}${fallback ? " fallback" : ""}`;
}

function feedSourceLabel(article: FeedItem, source: FeedMeta): string {
  const organization = cleanSourceLabel(article.organization);
  const author = cleanSourceLabel(article.author);
  const feedLabel = cleanSourceLabel(article.feed_label);
  if (organization.toLowerCase() === "news" && author) {
    return author;
  }
  return organization || feedLabel || source.label || labelFromFeedKey(article.feed_key) || "Unknown";
}

function isNewsApiArticle(article: FeedItem): boolean {
  return article.item_type === "document" && String(article.source_kind || "").toLowerCase() === "newsapi_article";
}

function SourceProvenanceChip({ article }: { article: FeedItem }) {
  if (!isNewsApiArticle(article)) {
    return null;
  }
  return (
    <span
      title="Discovered through NewsAPI; source name is the original publisher."
      style={{
        display: "inline-flex",
        alignItems: "center",
        width: "fit-content",
        border: "1px solid rgba(105, 219, 124, 0.34)",
        borderRadius: 4,
        padding: "2px 5px",
        color: "#9df0ad",
        background: "rgba(105, 219, 124, 0.1)",
        fontSize: 9,
        fontWeight: 800,
        lineHeight: 1.15,
        letterSpacing: "0.09em",
        textTransform: "uppercase",
      }}
    >
      NewsAPI
    </span>
  );
}

function articleSourceText(article: FeedItem): string {
  const source = getFeedMeta(article.feed_key, article.feed_label);
  return [
    article.feed_key,
    article.feed_label ?? "",
    source.label,
    article.organization ?? "",
    article.author ?? "",
    article.source_kind ?? "",
    article.doc_type ?? "",
    article.url ?? "",
  ].join(" ").toLowerCase();
}

function matchesLegalSource(article: FeedItem): boolean {
  const feedKey = String(article.feed_key || "").toLowerCase();
  const sourceKind = String(article.source_kind || "").toLowerCase();
  const text = articleSourceText(article);
  return (
    LEGAL_SOURCE_FEED_KEYS.has(feedKey) ||
    sourceKind === "jdsupra_article" ||
    text.includes("jdsupra.com") ||
    text.includes("jdsupra") ||
    text.includes("law360") ||
    text.includes("thecorporatecounsel.net") ||
    text.includes("clsbluesky.law.columbia.edu") ||
    text.includes("harvard corporate governance") ||
    text.includes("gibsondunn") ||
    text.includes("clearyenforcementwatch") ||
    text.includes("cooley") ||
    text.includes("latham") ||
    text.includes("globalfinregblog") ||
    text.includes("covington") ||
    text.includes("insideprivacy.com") ||
    text.includes("globalpolicywatch.com") ||
    text.includes("ballardspahr") ||
    text.includes("consumerfinancemonitor.com") ||
    text.includes("kelleydrye") ||
    text.includes("dataprotectionreport.com") ||
    text.includes("privacyworld.blog") ||
    text.includes("financialservicesperspectives.com") ||
    text.includes("eyeonenforcement.com")
  );
}

function matchesSecAll(article: FeedItem): boolean {
  const feedKey = String(article.feed_key || "").toLowerCase();
  const sourceKind = String(article.source_kind || "").toLowerCase();
  const text = articleSourceText(article);
  return (
    feedKey.startsWith("sec_") ||
    feedKey.startsWith("document_sec_") ||
    sourceKind.startsWith("sec_") ||
    text.includes("sec.gov") ||
    text.includes("securities and exchange commission")
  );
}

function matchesCyberSource(article: FeedItem): boolean {
  const feedKey = String(article.feed_key || "").toLowerCase();
  const sourceKind = String(article.source_kind || "").toLowerCase();
  const text = articleSourceText(article);
  return (
    CYBER_SOURCE_FEED_KEYS.has(feedKey) ||
    CYBER_SOURCE_KINDS.has(sourceKind) ||
    text.includes("cisa.gov") ||
    text.includes("krebsonsecurity.com") ||
    text.includes("thehackernews.com") ||
    text.includes("welivesecurity.com") ||
    text.includes("sophos.com") ||
    text.includes("flashpoint.io") ||
    text.includes("recordedfuture.com") ||
    text.includes("intel471.com") ||
    text.includes("therecord.media") ||
    text.includes("wired.com") ||
    text.includes("tripwire.com") ||
    text.includes("akamai.com") ||
    text.includes("darkreading.com") ||
    text.includes("securityweek.com") ||
    text.includes("microsoft.com/security")
  );
}

function fallbackDocumentTopicMatches(article: FeedItem, rules: TopicRuleView[]): TopicRuleView[] {
  if (article.item_type !== "document") return [];
  const text = normalizeMatchText([
    article.title,
    article.description ?? "",
    article.doc_type ?? "",
    article.topics?.join(" ") ?? "",
    article.keywords?.join(" ") ?? "",
    article.url ?? "",
  ].join(" "));
  const topicKeys = new Set<string>();

  if (/\b(crude|crude oil|oil|brent|wti|natural gas|lng|energy prices|persian gulf|hormuz)\b/.test(text)) {
    topicKeys.add("COMMODITIES_ENERGY_MARKETS");
  }
  if (/\b(persian gulf|hormuz|iran|tariff|sanctions|shipping lanes|trade war|export controls)\b/.test(text)) {
    topicKeys.add("GEOPOLITICAL_TRADE_RISK");
  }
  if (/\b(ipo|initial public offering|public offering|share sale|listing)\b/.test(text)) {
    topicKeys.add("CAPITAL_FORMATION");
  }
  if (/\b(central bank|interest rate|hawkish fed|federal reserve|ecb|bank capital)\b/.test(text)) {
    topicKeys.add("BANKING_PAYMENTS");
    topicKeys.add("ECONOMIC_GROWTH");
  }

  if (topicKeys.size === 0) return [];
  const fallbackLabels: Record<string, { label: string; sort_order: number }> = {
    COMMODITIES_ENERGY_MARKETS: { label: "Commodities & Energy Markets", sort_order: 160 },
    GEOPOLITICAL_TRADE_RISK: { label: "Geopolitical & Trade Risk", sort_order: 170 },
    CAPITAL_FORMATION: { label: "Capital Formation", sort_order: 20 },
    BANKING_PAYMENTS: { label: "Banking & Payments", sort_order: 90 },
    ECONOMIC_GROWTH: { label: "Economic Growth", sort_order: 120 },
  };

  return [...topicKeys]
    .map((key) => {
      const configured = rules.find((rule) => rule.topic_key === key);
      if (configured) return configured;
      const fallback = fallbackLabels[key];
      if (!fallback) return null;
      return {
        topic_key: key,
        label: fallback.label,
        keywords: [],
        keywordMatchers: [],
        sort_order: fallback.sort_order,
      };
    })
    .filter((topic): topic is TopicRuleView => Boolean(topic))
    .sort((a, b) => a.sort_order - b.sort_order || a.label.localeCompare(b.label));
}

function matchesSourceFilter(article: FeedItem, sourceFilter: SourceFilter): boolean {
  if (sourceFilter === "ALL") return true;
  const feedKey = String(article.feed_key || "").toLowerCase();
  const sourceKind = String(article.source_kind || "").toLowerCase();
  const url = String(article.url || "").toLowerCase();
  const text = articleSourceText(article);
  if (sourceFilter === "SEC_SPEECHES") {
    return (
      feedKey === "sec_speeches_statements" ||
      sourceKind === "sec_speech" ||
      url.includes("/newsroom/speeches-statements/") ||
      url.includes("/news/speeches-statements/")
    );
  }
  if (sourceFilter === "SEC_ENFORCEMENT") {
    return (
      feedKey === "sec_litigation_releases" ||
      feedKey === "document_sec_enforcement_litigation" ||
      feedKey === "document_sec_administrative_proceeding" ||
      feedKey === "document_sec_trading_suspension" ||
      feedKey === "sec_administrative_proceedings" ||
      feedKey === "sec_trading_suspensions" ||
      sourceKind === "sec_enforcement_litigation" ||
      sourceKind === "sec_administrative_proceeding" ||
      sourceKind === "sec_trading_suspension" ||
      url.includes("/enforcement-litigation/")
    );
  }
  if (sourceFilter === "SEC_ALL") {
    return matchesSecAll(article);
  }
  if (sourceFilter === "FINRA") {
    return feedKey.startsWith("finra_") || sourceKind.startsWith("finra_") || text.includes("finra") || text.includes("financial industry regulatory authority");
  }
  if (sourceFilter === "DOJ") {
    return (
      sourceKind.startsWith("doj_") ||
      text.includes("doj") ||
      text.includes("justice.gov") ||
      text.includes("department of justice") ||
      text.includes("justice department")
    );
  }
  if (sourceFilter === "FED") {
    return sourceKind.startsWith("federal_reserve_") || text.includes("federal reserve") || text.includes("federalreserve.gov");
  }
  if (sourceFilter === "CFTC") {
    return sourceKind.startsWith("cftc_") || text.includes("cftc.gov") || text.includes("commodity futures trading commission");
  }
  if (sourceFilter === "TREASURY") {
    return sourceKind.startsWith("treasury_") || text.includes("treasury.gov") || text.includes("treasury");
  }
  if (sourceFilter === "MARKET_SOURCES") {
    if (matchesSecAll(article) || matchesCyberSource(article)) {
      return false;
    }
    return (
      sourceKind === "pcaob_update" ||
      sourceKind === "msrb_press_release" ||
      text.includes("pcaobus.org") ||
      text.includes("msrb.org") ||
      text.includes("federalregister.gov")
    );
  }
  if (sourceFilter === "CONGRESS") {
    return (
      CONGRESS_SOURCE_FEED_KEYS.has(feedKey) ||
      sourceKind === "congress_crs_product" ||
      sourceKind === "senate_committee_site" ||
      text.includes("congress.gov") ||
      text.includes("banking.senate.gov") ||
      text.includes("finance.senate.gov") ||
      text.includes("agriculture.senate.gov") ||
      text.includes("judiciary.senate.gov") ||
      text.includes("hsgac.senate.gov") ||
      text.includes("commerce.senate.gov") ||
      text.includes("senate banking committee") ||
      text.includes("senate finance committee") ||
      text.includes("senate agriculture committee") ||
      text.includes("senate judiciary committee") ||
      text.includes("senate homeland security") ||
      text.includes("senate commerce committee") ||
      text.includes("crs product")
    );
  }
  if (sourceFilter === "PONZI_INVESTOR_FRAUD") {
    return feedKey === "google_news_ponzi_investor_fraud";
  }
  if (sourceFilter === "CYBER") {
    return matchesCyberSource(article);
  }
  if (sourceFilter === "WSJ") {
    return feedKey.startsWith("wsj_") || sourceKind === "wsj_rss_article" || sourceKind === "wsj_dow_jones" || text.includes("wall street journal") || text.includes("wsj.com") || text.includes("dowjones");
  }
  if (sourceFilter === "BLOOMBERG") {
    return sourceKind === "bloomberg_apify_article" || sourceKind === "bloomberg_public_article" || text.includes("bloomberg.com") || text.includes("bloomberg");
  }
  if (sourceFilter === "SUBSTACK") {
    return sourceKind === "substack_public_article" || text.includes("substack.com");
  }
  if (sourceFilter === "NEWSAPI") {
    return sourceKind === "newsapi_article";
  }
  if (sourceFilter === "SIFMA") {
    return sourceKind === "sifma_news_item" || text.includes("sifma");
  }
  if (sourceFilter === "TRADE_ASSOCIATIONS") {
    return (
      sourceKind === "ici_news_item" ||
      sourceKind === "isda_news_item" ||
      sourceKind === "mfa_news_item" ||
      sourceKind === "fia_news_item" ||
      sourceKind === "aba_news_item" ||
      sourceKind === "bpi_news_item" ||
      sourceKind === "icba_news_item" ||
      sourceKind === "lsta_news_item" ||
      text.includes("ici.org") ||
      text.includes("isda.org") ||
      text.includes("mfaalts.org") ||
      text.includes("fia.org") ||
      text.includes("aba.com") ||
      text.includes("bpi.com") ||
      text.includes("icba.org") ||
      text.includes("lsta.org")
    );
  }
  if (sourceFilter === "LEGAL_SOURCES") {
    return matchesLegalSource(article);
  }
  if (sourceFilter === "TRADE_MEDIA") {
    if (matchesLegalSource(article) || matchesCyberSource(article)) {
      return false;
    }
    if (TRADE_MEDIA_FEED_KEYS.has(feedKey)) {
      return true;
    }
    return [
      "investmentnews_article",
      "citywire_article",
      "ritholtz_article",
      "ft_portfolios_market_commentary",
      "liberty_street_economics_article",
      "wealth_of_common_sense_article",
    ].includes(sourceKind);
  }
  if (sourceFilter === "REDDIT") {
    return sourceKind === "reddit_post" || text.includes("reddit.com");
  }
  return false;
}

function isBloombergArticle(article: FeedItem): boolean {
  const sourceKind = String(article.source_kind || "").toLowerCase();
  return article.item_type === "document" && (sourceKind === "bloomberg_apify_article" || sourceKind === "bloomberg_public_article");
}

function savedArticleId(article: FeedItem): string {
  if (article.item_type === "document" && article.document_id) {
    return `document:${article.document_id}`;
  }
  return `article:${article.id}`;
}

const TONE_STYLE: Record<string, { color: string; bg: string; label: string; short: string; glyph: string }> = {
  positive: {
    color: "#41d39d",
    bg: "rgba(65, 211, 157, 0.12)",
    label: "Bullish",
    short: "POS",
    glyph: "▲",
  },
  negative: {
    color: "#ff6b7f",
    bg: "rgba(255, 107, 127, 0.12)",
    label: "Bearish",
    short: "NEG",
    glyph: "▼",
  },
  neutral: {
    color: "#8b95a1",
    bg: "rgba(139, 149, 161, 0.10)",
    label: "Neutral",
    short: "NEU",
    glyph: "◆",
  },
};

function matchesTopic(article: FeedItem, rule: TopicRuleView | null, topicMatchesByArticleId: Map<number, TopicRuleView[]>): boolean {
  if (!rule) return true;
  return (topicMatchesByArticleId.get(article.id) ?? []).some((item) => item.topic_key === rule.topic_key);
}

function matchesSearch(article: FeedItem, searchTerm: string): boolean {
  if (!searchTerm) return true;
  const haystack = [
    article.title,
    article.description ?? "",
    article.author ?? "",
    article.organization ?? "",
    article.source_kind ?? "",
    article.doc_type ?? "",
    ...(article.topics ?? []),
    ...(article.keywords ?? []),
  ].join(" ").toLowerCase();
  return haystack.includes(searchTerm);
}

function formatRelativeTime(dateStr: string | null): string {
  if (!dateStr) return "";
  const parsed = new Date(dateStr);
  const parsedMs = parsed.getTime();
  if (!Number.isFinite(parsedMs)) return "";
  const ms = Date.now() - parsedMs;
  if (ms < 0) return parsed.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  const mins = Math.floor(ms / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return new Date(dateStr).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function feedItemDate(article: Pick<FeedItem, "published_at" | "fetched_at">): string {
  return article.published_at || article.fetched_at || "";
}

function feedItemDateMs(article: Pick<FeedItem, "published_at" | "fetched_at">): number {
  const ms = new Date(feedItemDate(article)).getTime();
  if (!Number.isFinite(ms)) return 0;
  const maxFutureSkewMs = 6 * 60 * 60 * 1000;
  return ms > Date.now() + maxFutureSkewMs ? 0 : ms;
}

function normalizeDedupeText(value: string | null | undefined): string {
  return decodeEntities(String(value || ""))
    .toLowerCase()
    .replace(/['"\u2018\u2019\u201c\u201d]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function canonicalFeedItemUrl(value: string | null | undefined): string {
  const raw = String(value || "").trim();
  if (!raw) return "";
  try {
    const url = new URL(raw);
    url.hash = "";
    for (const key of [...url.searchParams.keys()]) {
      const lower = key.toLowerCase();
      if (
        lower.startsWith("utm_") ||
        ["fbclid", "gclid", "mc_cid", "mc_eid", "cmpid", "smid", "ref", "source"].includes(lower)
      ) {
        url.searchParams.delete(key);
      }
    }
    url.hostname = url.hostname.toLowerCase().replace(/^www\./, "");
    url.pathname = url.pathname.replace(/\/+$/, "");
    return `${url.hostname}${url.pathname}${url.searchParams.toString() ? `?${url.searchParams.toString()}` : ""}`.toLowerCase();
  } catch {
    return normalizeDedupeText(raw);
  }
}

function feedItemDedupeKeys(article: FeedItem): string[] {
  const urlKey = canonicalFeedItemUrl(article.url);
  const titleKey = normalizeDedupeText(article.title);
  const descriptionKey = normalizeDedupeText(article.description).slice(0, 120);
  const sourceKey = normalizeDedupeText(article.organization || article.feed_label || article.author || article.feed_key);
  return [
    urlKey ? `url:${urlKey}` : "",
    titleKey && sourceKey ? `title-source:${titleKey}:${sourceKey}` : "",
    titleKey && descriptionKey ? `title-desc:${titleKey}:${descriptionKey}` : "",
  ].filter(Boolean);
}

function dedupeFeedItems(items: FeedItem[]): FeedItem[] {
  const seen = new Set<string>();
  const out: FeedItem[] = [];
  for (const item of items) {
    const keys = feedItemDedupeKeys(item);
    if (keys.some((key) => seen.has(key))) {
      continue;
    }
    for (const key of keys) {
      seen.add(key);
    }
    out.push(item);
  }
  return out;
}

function formatClock(date: Date): string {
  return date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
}

function formatUpdated(dateStr: string | null): string {
  if (!dateStr) return "";
  return new Date(dateStr).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
}

function ellipsize(text: string, max = 120): string {
  const value = decodeEntities(text || "");
  return value.length > max ? `${value.slice(0, max - 1).trimEnd()}…` : value;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  const payload = (await res.json()) as ApiEnvelope<T>;
  if (!res.ok || !payload.ok || payload.data === undefined) {
    throw new Error(payload.error || `Request failed with ${res.status}`);
  }
  return payload.data;
}

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const media = window.matchMedia(query);
    const update = () => setMatches(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, [query]);

  return matches;
}

function statusClass(value: string): string {
  const s = String(value || "").toLowerCase();
  if (["enriched", "reviewed", "success"].includes(s)) return "status-chip status-success";
  if (["fallback_enriched", "queued", "running"].includes(s)) return "status-chip status-warn";
  if (["failed", "rejected"].includes(s)) return "status-chip status-failure";
  return "status-chip status-neutral";
}

function analysisChipClass(value: string): string {
  const label = String(value || "").toLowerCase();
  if (["supportive", "supports", "aligned", "favorable", "positive"].includes(label)) return "status-chip status-success";
  if (["opposed", "opposes", "critical", "negative", "adverse"].includes(label)) return "status-chip status-failure";
  if (["mixed", "qualified", "partially_supportive"].includes(label)) return "status-chip status-warn";
  return "status-chip status-neutral";
}

function formatAnalysisLabel(value: string): string {
  const normalized = String(value || "").trim();
  if (!normalized) return "";
  return normalized
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function readStringField(value: unknown, key: string): string {
  if (!value || typeof value !== "object") return "";
  const out = (value as Record<string, unknown>)[key];
  return typeof out === "string" ? out.trim() : "";
}

function readNumberField(value: unknown, key: string): number {
  if (!value || typeof value !== "object") return 0;
  const out = Number.parseFloat(String((value as Record<string, unknown>)[key] ?? "0"));
  return Number.isFinite(out) ? out : 0;
}

function pickPrimaryAnalysis(detail: DocumentDetailData | null | undefined): {
  kind: "position" | "stance" | "summary" | "";
  label: string;
  tone: string;
  rationale: string;
  confidence: number;
} {
  if (!detail) return { kind: "", label: "", tone: "", rationale: "", confidence: 0 };

  const positionLabel = readStringField(detail.enrichment.comment_position, "label").toLowerCase();
  const positionRationale = readStringField(detail.enrichment.comment_position, "rationale");
  const positionConfidence = readNumberField(detail.enrichment.comment_position, "confidence");
  if (positionLabel && positionLabel !== "not_applicable" && positionLabel !== "unclear") {
    return {
      kind: "position",
      label: positionLabel,
      tone: positionLabel,
      rationale: positionRationale,
      confidence: Math.max(0, Math.min(1, positionConfidence)),
    };
  }

  const stanceLabel = readStringField(detail.enrichment.stance, "label").toLowerCase();
  const stanceTarget = readStringField(detail.enrichment.stance, "target");
  if (stanceLabel && stanceLabel !== "unclear" && stanceLabel !== "not_applicable") {
    return {
      kind: "stance",
      label: stanceTarget ? `${stanceLabel} (${stanceTarget})` : stanceLabel,
      tone: stanceLabel,
      rationale: "",
      confidence: Math.max(0, Math.min(1, Number(detail.enrichment.confidence || 0))),
    };
  }

  if (detail.enrichment.summary) {
    return { kind: "summary", label: "summary_ready", tone: "summary_ready", rationale: "", confidence: 0 };
  }

  return { kind: "", label: "", tone: "", rationale: "", confidence: 0 };
}

function renderAnalysisChips(items: string[], emptyLabel: string) {
  if (!items.length) {
    return <span style={{ color: "#6f819d", fontSize: 12 }}>{emptyLabel}</span>;
  }
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
      {items.slice(0, 8).map((item) => (
        <span key={item} className="tone-chip">
          {item}
        </span>
      ))}
    </div>
  );
}

function FullArticleModal({
  article,
  detail,
  loading,
  error,
  onClose,
  retry,
  compact,
}: {
  article: FeedItem;
  detail: DocumentDetailData | undefined;
  loading: boolean;
  error: string;
  onClose: () => void;
  retry: () => void;
  compact: boolean;
}) {
  const paragraphs = detail?.content.paragraphs?.length
    ? detail.content.paragraphs
    : detail?.content.full_text
      ? detail.content.full_text.split(/\n{2,}/).map((item) => item.trim()).filter(Boolean)
      : [];
  const sourceLabel = feedSourceLabel(article, getFeedMeta(article.feed_key, article.feed_label));

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Full Bloomberg article"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 80,
        display: "flex",
        alignItems: compact ? "stretch" : "center",
        justifyContent: "center",
        padding: compact ? 0 : 24,
        background: "rgba(2, 8, 14, 0.74)",
        backdropFilter: "blur(6px)",
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: "min(960px, 100%)",
          maxHeight: compact ? "100vh" : "86vh",
          display: "grid",
          gridTemplateRows: "auto minmax(0, 1fr)",
          border: "1px solid rgba(79,213,255,0.24)",
          borderRadius: compact ? 0 : 8,
          background: "linear-gradient(180deg, rgba(8,18,30,0.98), rgba(5,13,23,0.98))",
          boxShadow: "0 28px 90px rgba(0,0,0,0.54)",
          overflow: "hidden",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
            justifyContent: "space-between",
            gap: 14,
            padding: compact ? "14px 14px 12px" : "18px 20px 14px",
            borderBottom: "1px solid rgba(112,142,187,0.16)",
          }}
        >
          <div style={{ minWidth: 0 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
              <span className="tone-chip">{sourceLabel}</span>
              <SourceProvenanceChip article={article} />
              <span className="tone-chip">{formatRelativeTime(feedItemDate(article))}</span>
              {article.author ? <span className="tone-chip">By {decodeEntities(article.author)}</span> : null}
            </div>
            <h2 style={{ margin: 0, color: "#f4f7fc", fontSize: compact ? 17 : 20, lineHeight: 1.35 }}>
              {decodeEntities(article.title || "Bloomberg article")}
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close article"
            style={{
              minWidth: 34,
              minHeight: 34,
              borderRadius: 999,
              border: "1px solid rgba(112,142,187,0.24)",
              background: "rgba(14,24,39,0.72)",
              color: "#dbe7f5",
              cursor: "pointer",
              fontSize: 18,
              lineHeight: 1,
            }}
          >
            x
          </button>
        </div>
        <div style={{ overflow: "auto", padding: compact ? "14px" : "18px 20px 22px" }}>
          {loading ? (
            <p style={{ color: "#9fb0c7", fontSize: 13 }}>Loading article text...</p>
          ) : error ? (
            <div style={{ display: "grid", gap: 10 }}>
              <p style={{ color: "#ff8aa0", fontSize: 13 }}>{error}</p>
              <button type="button" className="link-inline text-xs" onClick={retry}>
                Retry
              </button>
            </div>
          ) : paragraphs.length ? (
            <article style={{ display: "grid", gap: 13, color: "#d8e4f4", fontSize: compact ? 14 : 15, lineHeight: 1.72 }}>
              {paragraphs.map((paragraph, idx) => (
                <p key={`${idx}_${paragraph.slice(0, 20)}`} style={{ margin: 0 }}>
                  {decodeEntities(paragraph)}
                </p>
              ))}
            </article>
          ) : (
            <p style={{ color: "#9fb0c7", fontSize: 13 }}>No stored article text is available yet.</p>
          )}
          {article.url ? (
            <div style={{ marginTop: 18, paddingTop: 14, borderTop: "1px solid rgba(112,142,187,0.14)" }}>
              <a href={article.url} target="_blank" rel="noopener noreferrer" className="link-inline text-xs">
                Open original Bloomberg page
              </a>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function articleListSignature(articles: FeedItem[]): string {
  const first = articles[0];
  const last = articles[articles.length - 1];
  const firstDate = first ? feedItemDate(first) : "";
  const lastDate = last ? feedItemDate(last) : "";
  return `${articles.length}:${first?.id ?? ""}:${firstDate}:${last?.id ?? ""}:${lastDate}`;
}

function documentListSignature(documents: DocumentListItem[]): string {
  const first = documents[0];
  const last = documents[documents.length - 1];
  return [
    documents.length,
    first?.document_id ?? "",
    first?.published_at || first?.date || "",
    last?.document_id ?? "",
    last?.published_at || last?.date || "",
  ].join(":");
}

function analysisMapFromFeedItems(items: FeedItem[]): Record<string, FeedItemAnalysis> {
  const out: Record<string, FeedItemAnalysis> = {};
  for (const item of items) {
    const analysis = toFeedItemAnalysis(item.analysis);
    if (analysis && !shouldRegenerateFeedAnalysis(analysis)) {
      out[savedArticleId(item)] = analysis;
    }
  }
  return out;
}

function stableNegativeId(value: string): number {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = ((hash << 5) - hash + value.charCodeAt(i)) | 0;
  }
  return -Math.max(1, Math.abs(hash));
}

function documentDescription(document: DocumentListItem): string {
  const summary = decodeEntities(document.enrichment_summary || "").trim();
  if (summary) {
    return summary;
  }
  return [
    document.doc_type,
    document.speaker ? `By ${document.speaker}` : "",
    (document.topics || []).slice(0, 5).join(", "),
    (document.keywords || []).slice(0, 5).join(", "),
  ].filter(Boolean).join(" | ");
}

function documentToFeedItem(document: DocumentListItem): FeedItem {
  const publishedAt = document.published_at || document.date || new Date(0).toISOString();
  return {
    id: stableNegativeId(document.document_id),
    guid: document.document_id,
    feed_key: `document_${document.source_kind || "document"}`,
    title: document.title || "Untitled document",
    url: document.url,
    description: documentDescription(document),
    author: document.speaker || document.organization || "Document",
    published_at: publishedAt,
    tone_label: document.sentiment_label || null,
    fetched_at: publishedAt,
    item_type: "document",
    document_id: document.document_id,
    organization: document.organization,
    source_kind: document.source_kind,
    doc_type: document.doc_type,
    enrichment_model: document.enrichment_model || "",
    topics: document.topics || [],
    keywords: document.keywords || [],
  };
}

function topicRulesSignature(rules: StoredRssTopicRule[]): string {
  return rules
    .map((rule) => `${rule.id}:${rule.topic_key}:${rule.active}:${rule.sort_order}:${rule.updated_at}:${rule.keywords}`)
    .join("|");
}

function TopicPill({ label }: { label: string }) {
  return (
    <span
      style={{
        border: "1px solid rgba(93, 123, 171, 0.32)",
        borderRadius: 4,
        padding: "2px 6px",
        fontSize: 10,
        lineHeight: 1.2,
        letterSpacing: "0.08em",
        color: "#8fa7c8",
        textTransform: "uppercase",
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </span>
  );
}

function finraFirmMatches(article: FeedItem): string[] {
  return Array.isArray(article.matched_finra_firms)
    ? article.matched_finra_firms.map(String).filter(Boolean)
    : [];
}

function FinraFirmMatchPills({ firms }: { firms: string[] }) {
  if (firms.length === 0) return null;

  return (
    <>
      {firms.slice(0, 2).map((firm) => (
        <span
          key={firm}
          title={`FINRA member firm match: ${firm}`}
          style={{
            display: "inline-flex",
            alignItems: "center",
            border: "1px solid rgba(184, 143, 255, 0.58)",
            borderRadius: 4,
            padding: "2px 6px",
            maxWidth: 220,
            overflow: "hidden",
            textOverflow: "ellipsis",
            fontSize: 10,
            lineHeight: 1.2,
            letterSpacing: "0.08em",
            color: "#d4c4ff",
            background: "rgba(137, 87, 229, 0.16)",
            textTransform: "uppercase",
            whiteSpace: "nowrap",
          }}
        >
          FINRA Firm: {firm}
        </span>
      ))}
      {firms.length > 2 ? (
        <span
          title={firms.slice(2).join(", ")}
          style={{
            display: "inline-flex",
            alignItems: "center",
            border: "1px solid rgba(184, 143, 255, 0.42)",
            borderRadius: 4,
            padding: "2px 6px",
            fontSize: 10,
            lineHeight: 1.2,
            letterSpacing: "0.08em",
            color: "#bea8f5",
            background: "rgba(137, 87, 229, 0.12)",
            textTransform: "uppercase",
            whiteSpace: "nowrap",
          }}
        >
          +{firms.length - 2}
        </span>
      ) : null}
    </>
  );
}

function ToneChip({ label }: { label: string | null }) {
  const tone = label && TONE_STYLE[label] ? label : "neutral";
  const style = TONE_STYLE[tone];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 5,
        color: style.color,
        fontSize: 11,
        letterSpacing: "0.10em",
        textTransform: "uppercase",
        whiteSpace: "nowrap",
      }}
    >
      <span>{style.glyph}</span>
      <span>{style.short}</span>
    </span>
  );
}

function TopicButton({
  label,
  active,
  onClick,
  count,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  count: number;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        width: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 8,
        textAlign: "left",
        border: "none",
        borderLeft: active ? "2px solid #63a8ff" : "2px solid transparent",
        background: active ? "rgba(67, 112, 186, 0.18)" : "transparent",
        color: active ? "#e6eef9" : "#9ba9bc",
        borderRadius: 4,
        padding: "7px 10px 7px 8px",
        cursor: "pointer",
        fontSize: 13,
        transition: "background 120ms ease, color 120ms ease, border-color 120ms ease",
      }}
      onMouseEnter={(e) => {
        if (!active) {
          (e.currentTarget as HTMLElement).style.background = "rgba(67, 112, 186, 0.08)";
          (e.currentTarget as HTMLElement).style.color = "#dbe6f4";
        }
      }}
      onMouseLeave={(e) => {
        if (!active) {
          (e.currentTarget as HTMLElement).style.background = "transparent";
          (e.currentTarget as HTMLElement).style.color = "#9ba9bc";
        }
      }}
    >
      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{label}</span>
      <span style={{ color: active ? "#8fb2ea" : "#64728a", fontSize: 12 }}>{count}</span>
    </button>
  );
}

function FeedRow({
  article,
  matchedTopics,
  active,
  onSelect,
  saved,
  onToggleSave,
  analysisOpen,
  analysisLabel,
  onToggleAnalysis,
  onOpenFullArticle,
  compact = false,
}: {
  article: FeedItem;
  matchedTopics: TopicRuleView[];
  active: boolean;
  onSelect: () => void;
  saved: boolean;
  onToggleSave: () => void;
  analysisOpen: boolean;
  analysisLabel: string;
  onToggleAnalysis: () => void;
  onOpenFullArticle?: () => void;
  compact?: boolean;
}) {
  const source = getFeedMeta(article.feed_key, article.feed_label);
  const sourceLabel = feedSourceLabel(article, source);
  const visibleTopics = matchedTopics.slice(0, 3);
  const matchedFirms = finraFirmMatches(article);
  const hasFirmMatch = matchedFirms.length > 0;
  const rowBackground = active
    ? "rgba(67, 112, 186, 0.08)"
    : hasFirmMatch
      ? "rgba(137, 87, 229, 0.07)"
      : "transparent";
  const rowAccent = hasFirmMatch ? "inset 3px 0 0 rgba(184, 143, 255, 0.72)" : undefined;
  const description = ellipsize(article.description ?? "", article.item_type === "document" ? 120 : 82);
  const showFullArticle = isBloombergArticle(article) && !!onOpenFullArticle;
  const analysisButtonStyle = {
    border: analysisOpen ? "1px solid rgba(79,213,255,0.55)" : "1px solid rgba(90,118,162,0.28)",
    background: analysisOpen ? "rgba(79,213,255,0.12)" : "rgba(14,24,39,0.58)",
    color: analysisOpen ? "#e8f7ff" : "#9fb0c7",
    borderRadius: 999,
    padding: compact ? "8px 12px" : "4px 9px",
    fontSize: compact ? 11 : 10,
    fontWeight: 700,
    letterSpacing: "0.06em",
    textTransform: "uppercase" as const,
    cursor: "pointer",
  };

  if (compact) {
    return (
      <div
        style={{
          display: "grid",
          gap: 10,
          padding: "12px 0",
          borderTop: "1px solid rgba(112, 142, 187, 0.12)",
          background: rowBackground,
          boxShadow: rowAccent,
          cursor: "pointer",
          minWidth: 0,
        }}
        onClick={onSelect}
      >
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", minWidth: 0 }}>
            <span style={{ color: "#7f8faa", fontSize: 12, whiteSpace: "nowrap" }}>{formatRelativeTime(feedItemDate(article))}</span>
            <span style={{ color: source.color, fontSize: 12, fontWeight: 700 }}>{sourceLabel}</span>
            <SourceProvenanceChip article={article} />
            <ToneChip label={article.tone_label} />
          </div>
          <div onClick={(e) => e.stopPropagation()}>
            <BookmarkButton saved={saved} onToggle={onToggleSave} />
          </div>
        </div>

        <div style={{ minWidth: 0 }}>
          <a
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              color: "#edf3fb",
              fontSize: 15,
              fontWeight: 650,
              textDecoration: "none",
              lineHeight: 1.35,
            }}
          >
            {decodeEntities(article.title)}
          </a>
          {description ? (
            <div style={{ color: "#7f8faa", fontSize: 12, marginTop: 5, lineHeight: 1.45 }}>{description}</div>
          ) : null}
          {article.item_type === "document" ? (
            <div style={{ color: "#4fd5ff", fontSize: 10, marginTop: 6, letterSpacing: "0.12em", textTransform: "uppercase" }}>
              Primary document
            </div>
          ) : null}
        </div>

        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {visibleTopics.map((topic) => (
            <TopicPill key={`${article.id}_${topic.topic_key}`} label={topic.label} />
          ))}
          <FinraFirmMatchPills firms={matchedFirms} />
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onToggleAnalysis();
            }}
            style={{ ...analysisButtonStyle, justifySelf: "start" }}
          >
            {analysisLabel}
          </button>
          {showFullArticle ? (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onOpenFullArticle?.();
              }}
              style={{ ...analysisButtonStyle, justifySelf: "start" }}
            >
              Read Article
            </button>
          ) : null}
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "80px 150px 66px minmax(0, 1fr) 220px 24px",
        gap: 14,
        alignItems: "start",
        padding: "10px 0",
        borderTop: "1px solid rgba(112, 142, 187, 0.12)",
        background: rowBackground,
        boxShadow: rowAccent,
        cursor: "pointer",
      }}
      onClick={onSelect}
    >
      <div style={{ color: "#7f8faa", fontSize: 12, whiteSpace: "nowrap" }}>{formatRelativeTime(feedItemDate(article))}</div>
      <div style={{ display: "grid", gap: 4, minWidth: 0 }} title={sourceLabel}>
        <span style={{ color: source.color, fontSize: 12, fontWeight: 700, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {sourceLabel}
        </span>
        <SourceProvenanceChip article={article} />
      </div>
      <ToneChip label={article.tone_label} />
      <div style={{ minWidth: 0 }}>
        <a
          href={article.url}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            color: "#edf3fb",
            fontSize: 15,
            fontWeight: 600,
            textDecoration: "none",
            lineHeight: 1.35,
          }}
        >
          {decodeEntities(article.title)}
        </a>
        {description ? (
          <div style={{ color: "#7f8faa", fontSize: 12, marginTop: 3, lineHeight: 1.45 }}>{description}</div>
        ) : null}
        {article.item_type === "document" ? (
          <div style={{ color: "#4fd5ff", fontSize: 10, marginTop: 5, letterSpacing: "0.12em", textTransform: "uppercase" }}>
            Primary document
          </div>
        ) : null}
        <div style={{ marginTop: 8, display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onToggleAnalysis();
            }}
            style={analysisButtonStyle}
          >
            {analysisLabel}
          </button>
          {showFullArticle ? (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onOpenFullArticle?.();
              }}
              style={analysisButtonStyle}
            >
              Read Article
            </button>
          ) : null}
        </div>
      </div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "flex-end" }}>
        {visibleTopics.map((topic) => (
          <TopicPill key={`${article.id}_${topic.topic_key}`} label={topic.label} />
        ))}
        <FinraFirmMatchPills firms={matchedFirms} />
      </div>
      <BookmarkButton saved={saved} onToggle={onToggleSave} />
    </div>
  );
}

function FeaturedCard({
  article,
  matchedTopics,
  saved,
  onToggleSave,
  analysisOpen,
  analysisLabel,
  onToggleAnalysis,
  onOpenFullArticle,
  compact = false,
}: {
  article: FeedItem;
  matchedTopics: TopicRuleView[];
  saved: boolean;
  onToggleSave: () => void;
  analysisOpen: boolean;
  analysisLabel: string;
  onToggleAnalysis: () => void;
  onOpenFullArticle?: () => void;
  compact?: boolean;
}) {
  const source = getFeedMeta(article.feed_key, article.feed_label);
  const sourceLabel = feedSourceLabel(article, source);
  const tone = article.tone_label && TONE_STYLE[article.tone_label] ? article.tone_label : "neutral";
  const matchedFirms = finraFirmMatches(article);
  const hasFirmMatch = matchedFirms.length > 0;
  const firmMatchBackground = hasFirmMatch ? "rgba(137, 87, 229, 0.06)" : "transparent";
  const firmMatchAccent = hasFirmMatch ? "inset 3px 0 0 rgba(184, 143, 255, 0.72)" : undefined;
  const showFullArticle = isBloombergArticle(article) && !!onOpenFullArticle;
  const analysisButtonStyle = {
    border: analysisOpen ? "1px solid rgba(79,213,255,0.55)" : "1px solid rgba(90,118,162,0.28)",
    background: analysisOpen ? "rgba(79,213,255,0.12)" : "rgba(14,24,39,0.58)",
    color: analysisOpen ? "#e8f7ff" : "#9fb0c7",
    borderRadius: 999,
    padding: compact ? "8px 12px" : "5px 10px",
    fontSize: compact ? 11 : 11,
    fontWeight: 700,
    letterSpacing: "0.06em",
    textTransform: "uppercase" as const,
    cursor: "pointer",
  };

  if (compact) {
    return (
      <div
        style={{
          borderTop: "1px solid rgba(112, 142, 187, 0.16)",
          borderBottom: "1px solid rgba(112, 142, 187, 0.16)",
          padding: "14px 0 16px",
          marginBottom: 4,
          display: "grid",
          gap: 11,
          minWidth: 0,
          background: firmMatchBackground,
          boxShadow: firmMatchAccent,
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", minWidth: 0 }}>
            <span style={{ color: "#8fa7c8", fontSize: 12, whiteSpace: "nowrap" }}>{formatRelativeTime(feedItemDate(article))}</span>
            <span style={{ color: source.color, fontSize: 12, fontWeight: 700 }}>{sourceLabel}</span>
            <SourceProvenanceChip article={article} />
            <ToneChip label={tone} />
          </div>
          <BookmarkButton saved={saved} onToggle={onToggleSave} size={16} />
        </div>

        <div style={{ minWidth: 0 }}>
          <a
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              color: "#f4f7fc",
              fontWeight: 700,
              fontSize: 16,
              lineHeight: 1.38,
              textDecoration: "none",
            }}
          >
            {decodeEntities(article.title)}
          </a>
          {article.description ? (
            <div style={{ marginTop: 9, color: "#b8c6d8", fontSize: 13, lineHeight: 1.55 }}>
              {ellipsize(article.description, 180)}
            </div>
          ) : null}
        </div>

        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {matchedTopics.slice(0, 4).map((topic) => (
            <TopicPill key={`${article.id}_${topic.topic_key}`} label={topic.label} />
          ))}
          <FinraFirmMatchPills firms={matchedFirms} />
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", color: "#8da0bc", fontSize: 11 }}>
          <span>{decodeEntities(article.author || (article.item_type === "document" ? "Document" : "News Desk"))}</span>
          <span aria-hidden="true">/</span>
          <span>{sourceLabel}</span>
          <span aria-hidden="true">/</span>
          <span style={{ color: TONE_STYLE[tone].color, fontWeight: 700 }}>{TONE_STYLE[tone].label}</span>
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onToggleAnalysis();
            }}
            style={{ ...analysisButtonStyle, justifySelf: "start" }}
          >
            {analysisLabel}
          </button>
          {showFullArticle ? (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onOpenFullArticle?.();
              }}
              style={{ ...analysisButtonStyle, justifySelf: "start" }}
            >
              Read Article
            </button>
          ) : null}
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        borderTop: "1px solid rgba(112, 142, 187, 0.16)",
        borderBottom: "1px solid rgba(112, 142, 187, 0.16)",
        padding: "14px 0 18px",
        marginBottom: 4,
        background: firmMatchBackground,
        boxShadow: firmMatchAccent,
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "80px 150px 66px minmax(0, 1fr) 240px 24px",
          gap: 14,
          alignItems: "start",
        }}
      >
        <div style={{ color: "#8fa7c8", fontSize: 12 }}>{formatRelativeTime(feedItemDate(article))}</div>
        <div style={{ display: "grid", gap: 4, minWidth: 0 }} title={sourceLabel}>
          <span style={{ color: source.color, fontSize: 12, fontWeight: 700, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {sourceLabel}
          </span>
          <SourceProvenanceChip article={article} />
        </div>
        <ToneChip label={tone} />
        <div style={{ minWidth: 0 }}>
          <a
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              color: "#f4f7fc",
              fontWeight: 700,
              fontSize: 16,
              lineHeight: 1.4,
              textDecoration: "none",
            }}
          >
            {decodeEntities(article.title)}
          </a>
          {article.description ? (
            <div style={{ marginTop: 14 }}>
              <div
                style={{
                  color: "#6e7e98",
                  fontSize: 10,
                  letterSpacing: "0.18em",
                  textTransform: "uppercase",
                  marginBottom: 8,
                }}
              >
                Why It Matters
              </div>
              <div
                style={{
                  color: "#dce7f7",
                  fontSize: 16,
                  lineHeight: 1.65,
                  fontStyle: "italic",
                  fontFamily: '"Iowan Old Style", "Palatino Linotype", serif',
                }}
              >
                {decodeEntities(article.description)}
              </div>
            </div>
          ) : null}
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "74px minmax(0, 1fr)",
            gap: "6px 10px",
            fontSize: 11,
            color: "#8da0bc",
            alignSelf: "stretch",
          }}
        >
          <div style={{ letterSpacing: "0.12em", textTransform: "uppercase", color: "#5e708a" }}>Author</div>
          <div style={{ color: "#d7e1ef" }}>{decodeEntities(article.author || (article.item_type === "document" ? "Document" : "News Desk"))}</div>
          <div style={{ letterSpacing: "0.12em", textTransform: "uppercase", color: "#5e708a" }}>Source</div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap", color: "#d7e1ef" }}>
            <span>{sourceLabel}</span>
            <SourceProvenanceChip article={article} />
          </div>
          <div style={{ letterSpacing: "0.12em", textTransform: "uppercase", color: "#5e708a" }}>Impact</div>
          <div style={{ color: TONE_STYLE[tone].color, fontWeight: 700 }}>{TONE_STYLE[tone].label.toUpperCase()}</div>
          <div style={{ letterSpacing: "0.12em", textTransform: "uppercase", color: "#5e708a" }}>Topics</div>
          <div style={{ color: "#d7e1ef" }}>
            {matchedTopics.length > 0 ? matchedTopics.map((topic) => topic.label).join(", ") : "Unmapped"}
          </div>
          {matchedFirms.length > 0 ? (
            <>
              <div style={{ letterSpacing: "0.12em", textTransform: "uppercase", color: "#5e708a" }}>FINRA Firms</div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                <FinraFirmMatchPills firms={matchedFirms} />
              </div>
            </>
          ) : null}
        </div>
        <BookmarkButton saved={saved} onToggle={onToggleSave} size={16} />
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12, paddingLeft: 214 }}>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onToggleAnalysis();
          }}
          style={analysisButtonStyle}
        >
          {analysisLabel}
        </button>
        {showFullArticle ? (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onOpenFullArticle?.();
            }}
            style={analysisButtonStyle}
          >
            Read Article
          </button>
        ) : null}
      </div>
    </div>
  );
}

function FeedAnalysisPanel({
  article,
  matchedTopics,
  analysis,
  analysisLoading,
  analysisError,
  retryAnalysis,
  detail,
  loading,
  error,
  retry,
  compact = false,
}: {
  article: FeedItem;
  matchedTopics: TopicRuleView[];
  analysis: FeedItemAnalysis | undefined;
  analysisLoading: boolean;
  analysisError: string;
  retryAnalysis: () => void;
  detail: DocumentDetailData | undefined;
  loading: boolean;
  error: string;
  retry: () => void;
  compact?: boolean;
}) {
  const source = getFeedMeta(article.feed_key, article.feed_label);
  const sourceLabel = feedSourceLabel(article, source);
  const primaryAnalysis = pickPrimaryAnalysis(detail);
  const tone = article.tone_label && TONE_STYLE[article.tone_label] ? article.tone_label : "neutral";
  const decodedDescription = decodeEntities(article.description || "");
  const topicLabels = matchedTopics.map((topic) => topic.label);
  const matchedFirms = finraFirmMatches(article);
  const articleSummary = decodedDescription || "No feed summary is available for this article yet.";
  const analysisModel = feedAnalysisModelLabel(analysis);
  const analysisModelTitle = feedAnalysisModelTitle(analysis);
  const documentModel = detail?.enrichment.model || article.enrichment_model || "";
  const documentModelLabel = hostedModelLabel(documentModel, detail?.enrichment.status === "fallback_enriched");
  const documentModelTitle = hostedModelTitle(documentModel, detail?.enrichment.status === "fallback_enriched");
  const mainGridColumns = compact ? "1fr" : "minmax(0,1.45fr) minmax(220px,0.55fr)";
  const twoColumnGrid = compact ? "1fr" : "repeat(2, minmax(0, 1fr))";
  const threeColumnGrid = compact ? "1fr" : "repeat(3, minmax(0, 1fr))";
  const analysisBlock = analysisLoading ? (
    <p style={{ color: "#9fb0c7", fontSize: 13 }}>Generating analysis...</p>
  ) : analysisError ? (
    <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
      <p style={{ color: "#ff8aa0", fontSize: 13 }}>{analysisError}</p>
      <button type="button" className="link-inline text-xs" onClick={retryAnalysis}>
        Retry
      </button>
    </div>
  ) : analysis ? (
    <div style={{ display: "grid", gap: 14 }}>
      <div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          <span className={analysisChipClass(tone)}>Tone: {TONE_STYLE[tone].label}</span>
          <span className="tone-chip">Source: {sourceLabel}</span>
          <SourceProvenanceChip article={article} />
          <FinraFirmMatchPills firms={matchedFirms} />
          <span className="tone-chip" title={analysisModelTitle}>Model: {analysisModel}</span>
        </div>
        <p style={{ marginTop: 10, color: "#dbe7f5", fontSize: 14, fontWeight: 700, lineHeight: 1.55 }}>
          {analysis.thesis}
        </p>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: twoColumnGrid, gap: 14 }}>
        <div>
          <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
            Why It Matters
          </p>
          <ul style={{ margin: 0, paddingLeft: 16, color: "#b7c7dc", fontSize: 12, lineHeight: 1.6 }}>
            {analysis.why_it_matters.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>
        <div>
          <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
            Risk Signals
          </p>
          <ul style={{ margin: 0, paddingLeft: 16, color: "#b7c7dc", fontSize: 12, lineHeight: 1.6 }}>
            {analysis.risk_signals.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: threeColumnGrid, gap: 14 }}>
        <div>
          <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
            Keywords
          </p>
          {renderAnalysisChips(analysis.keywords, "No keywords extracted")}
        </div>
        <div>
          <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
            Individuals
          </p>
          {renderAnalysisChips(analysis.individuals, "No individuals identified")}
        </div>
        <div>
          <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
            Entities
          </p>
          {renderAnalysisChips(analysis.entities, "No entities identified")}
        </div>
      </div>
      <div>
        <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
          Follow-Up
        </p>
        <ul style={{ margin: 0, paddingLeft: 16, color: "#8ea0ba", fontSize: 12, lineHeight: 1.6 }}>
          {analysis.follow_up_questions.map((item) => <li key={item}>{item}</li>)}
        </ul>
      </div>
    </div>
  ) : (
    <p style={{ color: "#7f8faa", fontSize: 13 }}>Open analysis is preparing this item.</p>
  );

  if (article.item_type === "document") {
    if (loading) {
      return <p style={{ color: "#9fb0c7", fontSize: 13 }}>Loading analysis...</p>;
    }

    if (error) {
      return (
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <p style={{ color: "#ff8aa0", fontSize: 13 }}>{error}</p>
          <button type="button" className="link-inline text-xs" onClick={retry}>
            Retry
          </button>
        </div>
      );
    }

    if (!detail) {
      return <p style={{ color: "#7f8faa", fontSize: 13 }}>No analysis is available for this document.</p>;
    }

    return (
      <div style={{ display: "grid", gap: 18 }}>
        {analysisBlock}
        <div style={{ height: 1, background: "rgba(112, 142, 187, 0.12)" }} />
        <div style={{ display: "grid", gridTemplateColumns: mainGridColumns, gap: compact ? 14 : 18 }}>
        <div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            <span className={statusClass(detail.enrichment.status)}>{detail.enrichment.status || "not_enriched"}</span>
            <span className="tone-chip">Review: {detail.review.decision || "pending"}</span>
            <span className="tone-chip" title={documentModelTitle}>Model: {documentModelLabel}</span>
            {primaryAnalysis.kind === "position" ? (
              <span className={analysisChipClass(primaryAnalysis.tone)}>
                Position: {formatAnalysisLabel(primaryAnalysis.label)}
              </span>
            ) : primaryAnalysis.kind === "stance" ? (
              <span className={analysisChipClass(primaryAnalysis.tone)}>
                Stance: {formatAnalysisLabel(primaryAnalysis.label)}
              </span>
            ) : null}
            {primaryAnalysis.confidence > 0 ? (
              <span className="tone-chip">Confidence: {Math.round(primaryAnalysis.confidence * 100)}%</span>
            ) : null}
          </div>
          <p style={{ marginTop: 10, color: "#c6d4e6", fontSize: 13, lineHeight: 1.65 }}>
            {detail.enrichment.summary || "No summary is available for this document yet."}
          </p>
          {primaryAnalysis.rationale ? (
            <p style={{ marginTop: 8, color: "#8899b1", fontSize: 12, lineHeight: 1.55 }}>{primaryAnalysis.rationale}</p>
          ) : null}
        </div>
        <div style={{ display: "grid", gap: 12 }}>
          <div>
            <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
              Tags
            </p>
            {renderAnalysisChips(detail.enrichment.tags, "No tags yet")}
          </div>
          <div>
            <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
              Keywords
            </p>
            {renderAnalysisChips(detail.enrichment.keywords, "No keywords yet")}
          </div>
        </div>
      </div>
      </div>
    );
  }

  return (
    <div style={{ display: "grid", gridTemplateColumns: mainGridColumns, gap: compact ? 14 : 18 }}>
      <div>
        {analysisBlock}
        <p style={{ marginTop: 10, color: "#c6d4e6", fontSize: 13, lineHeight: 1.65 }}>{articleSummary}</p>
      </div>
      <div style={{ display: "grid", gap: 12 }}>
        <div>
          <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
            Matched Topics
          </p>
          {renderAnalysisChips(topicLabels, "No mapped topics")}
        </div>
        {matchedFirms.length > 0 ? (
          <div>
            <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
              FINRA Firm Matches
            </p>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              <FinraFirmMatchPills firms={matchedFirms} />
            </div>
          </div>
        ) : null}
        <div>
          <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
            Follow-Up
          </p>
          <p style={{ color: "#8ea0ba", fontSize: 12, lineHeight: 1.55 }}>
            Review the source article, then save it if it should be promoted into research follow-up or briefing coverage.
          </p>
        </div>
      </div>
    </div>
  );
}

export function IntelBetaDashboard({
  initialArticles,
  initialTopicRules,
  initialDocuments = [],
}: {
  initialArticles: StoredRssArticle[];
  initialTopicRules: StoredRssTopicRule[];
  initialDocuments?: DocumentListItem[];
}) {
  const [articles, setArticles] = useState<StoredRssArticle[]>(initialArticles);
  const [documents, setDocuments] = useState<DocumentListItem[]>(initialDocuments);
  const documentFeedItems = useMemo(() => documents.map(documentToFeedItem), [documents]);
  const feedItems = useMemo<FeedItem[]>(
    () => dedupeFeedItems(
      [...articles, ...documentFeedItems]
        .sort((a, b) => feedItemDateMs(b) - feedItemDateMs(a))
    ),
    [articles, documentFeedItems]
  );
  const [topicRules, setTopicRules] = useState<StoredRssTopicRule[]>(initialTopicRules);
  const [selectedTopic, setSelectedTopic] = useState<TopicFilter>("ALL");
  const [selectedSource, setSelectedSource] = useState<SourceFilter>("ALL");
  const [visibleItemLimit, setVisibleItemLimit] = useState(FEED_RENDER_BATCH_SIZE);
  const [search, setSearch] = useState("");
  const [selectedArticleId, setSelectedArticleId] = useState<number | null>(feedItems[0]?.id ?? null);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [newCount, setNewCount] = useState(0);
  const newestFetchedAtRef = useRef<string>(initialArticles[0] ? feedItemDate(initialArticles[0]) : "");
  const articleSignatureRef = useRef(articleListSignature(initialArticles));
  const documentSignatureRef = useRef(documentListSignature(initialDocuments));
  const topicRulesSignatureRef = useRef(topicRulesSignature(initialTopicRules));
  const savedItems = useSavedItems();
  const [expandedAnalysis, setExpandedAnalysis] = useState<Record<string, boolean>>({});
  const [docDetails, setDocDetails] = useState<Record<string, DocumentDetailData>>({});
  const [docDetailLoading, setDocDetailLoading] = useState<Record<string, boolean>>({});
  const [docDetailError, setDocDetailError] = useState<Record<string, string>>({});
  const [fullArticleDocId, setFullArticleDocId] = useState("");
  const [feedAnalyses, setFeedAnalyses] = useState<Record<string, FeedItemAnalysis>>(() => analysisMapFromFeedItems(feedItems));
  const [feedAnalysisLoading, setFeedAnalysisLoading] = useState<Record<string, boolean>>({});
  const [feedAnalysisError, setFeedAnalysisError] = useState<Record<string, string>>({});

  const visibleTopicRules = useMemo(() => normalizeTopicRules(topicRules), [topicRules]);
  const topicIndex = useMemo(() => {
    const topicMatchesByArticleId = new Map<number, TopicRuleView[]>();
    const topicCounts = new Map<string, number>();
    const matchedArticles: FeedItem[] = [];

    for (const article of feedItems) {
      const directMatches = getMatchingTopics(article, visibleTopicRules);
      const matches = directMatches.length > 0 ? directMatches : fallbackDocumentTopicMatches(article, visibleTopicRules);
      topicMatchesByArticleId.set(article.id, matches);
      if (matches.length === 0 && article.item_type !== "document") {
        continue;
      }
      matchedArticles.push(article);
      for (const topic of matches) {
        topicCounts.set(topic.topic_key, (topicCounts.get(topic.topic_key) ?? 0) + 1);
      }
    }

    return { topicMatchesByArticleId, topicCounts, matchedArticles };
  }, [feedItems, visibleTopicRules]);
  const matchedArticles = topicIndex.matchedArticles;
  const sourceIndex = useMemo(() => {
    const sourceMatchesByArticleId = new Map<number, Set<SourceFilter>>();
    const counts = new Map<SourceFilter, number>();
    for (const article of matchedArticles) {
      const matches = new Set<SourceFilter>();
      for (const source of SOURCE_FILTERS) {
        if (matchesSourceFilter(article, source.key)) {
          matches.add(source.key);
          counts.set(source.key, (counts.get(source.key) ?? 0) + 1);
        }
      }
      sourceMatchesByArticleId.set(article.id, matches);
    }
    return { sourceCounts: counts, sourceMatchesByArticleId };
  }, [matchedArticles]);
  const sourceCounts = sourceIndex.sourceCounts;
  const selectedRule = selectedTopic === "ALL"
    ? null
    : visibleTopicRules.find((rule) => rule.topic_key === selectedTopic) ?? null;
  const selectedSourceLabel = selectedSource === "ALL"
    ? "All Sources"
    : SOURCE_FILTERS.find((source) => source.key === selectedSource)?.label ?? selectedSource;
  const searchTerm = search.trim().toLowerCase();
  const deferredSearchTerm = useDeferredValue(searchTerm);
  const isMobile = useMediaQuery("(max-width: 760px)");

  useEffect(() => {
    if (selectedTopic !== "ALL" && !visibleTopicRules.some((rule) => rule.topic_key === selectedTopic)) {
      setSelectedTopic("ALL");
    }
  }, [selectedTopic, visibleTopicRules]);

  useEffect(() => {
    const savedAnalyses = analysisMapFromFeedItems(feedItems);
    setFeedAnalyses((prev) => {
      const current = Object.fromEntries(Object.entries(prev).filter(([, analysis]) => !shouldRegenerateFeedAnalysis(analysis)));
      if (Object.keys(savedAnalyses).length === 0 && Object.keys(current).length === Object.keys(prev).length) {
        return prev;
      }
      return { ...savedAnalyses, ...current };
    });
  }, [feedItems]);

  useEffect(() => {
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let errStreak = 0;
    let mounted = true;

    const poll = async () => {
      try {
        const res = await fetch(`/api/intel/feed?limit=${LIVE_FEED_REFRESH_LIMIT}&includeDocuments=1`);
        if (!res.ok) { errStreak++; }
        else {
          const json = (await res.json()) as {
            ok: boolean;
            data: {
              articles: StoredRssArticle[];
              topicRules: StoredRssTopicRule[];
              documents?: DocumentListItem[];
              generatedAt: string;
            };
          };
          if (!json.ok) { errStreak++; }
          else {
            errStreak = 0;
            const fresh = json.data.articles;
            const freshRules = json.data.topicRules;
            const freshDocuments = json.data.documents ?? [];
            const newest = fresh[0] ? feedItemDate(fresh[0]) : "";
            let changed = false;
            if (newest && newest > newestFetchedAtRef.current) {
              const added = fresh.filter((article) => feedItemDate(article) > newestFetchedAtRef.current).length;
              newestFetchedAtRef.current = newest;
              setNewCount((count) => count + added);
              if (!selectedArticleId && fresh[0]?.id) {
                setSelectedArticleId(fresh[0].id);
              }
            }
            const nextArticleSignature = articleListSignature(fresh);
            if (nextArticleSignature !== articleSignatureRef.current) {
              articleSignatureRef.current = nextArticleSignature;
              changed = true;
              setArticles(fresh);
            }
            const nextDocumentSignature = documentListSignature(freshDocuments);
            if (json.data.documents && nextDocumentSignature !== documentSignatureRef.current) {
              documentSignatureRef.current = nextDocumentSignature;
              changed = true;
              setDocuments(freshDocuments);
            }
            const nextTopicRulesSignature = topicRulesSignature(freshRules);
            if (nextTopicRulesSignature !== topicRulesSignatureRef.current) {
              topicRulesSignatureRef.current = nextTopicRulesSignature;
              changed = true;
              setTopicRules(freshRules);
            }
            if (changed) setLastUpdated(new Date());
          }
        }
      } catch {
        errStreak++;
      }
      if (mounted) {
        // Retry transient failures promptly, but successful feed refreshes run hourly.
        const delay = errStreak > 0
          ? Math.min(15_000 * (2 ** (errStreak - 1)), 120_000)
          : LIVE_FEED_POLL_INTERVAL_MS;
        timeoutId = setTimeout(() => { void poll(); }, delay);
      }
    };

    void poll();
    return () => { mounted = false; if (timeoutId) clearTimeout(timeoutId); };
  }, [selectedArticleId]);

  const filtered = useMemo(
    () =>
      matchedArticles.filter(
        (article) =>
          matchesTopic(article, selectedRule, topicIndex.topicMatchesByArticleId) &&
          (selectedSource === "ALL" || !!sourceIndex.sourceMatchesByArticleId.get(article.id)?.has(selectedSource)) &&
          matchesSearch(article, deferredSearchTerm)
      ),
    [deferredSearchTerm, matchedArticles, selectedRule, selectedSource, sourceIndex.sourceMatchesByArticleId, topicIndex.topicMatchesByArticleId]
  );
  const visibleFiltered = useMemo(
    () => filtered.slice(0, visibleItemLimit),
    [filtered, visibleItemLimit]
  );
  const hasMoreFiltered = visibleFiltered.length < filtered.length;

  useEffect(() => {
    setVisibleItemLimit(FEED_RENDER_BATCH_SIZE);
  }, [deferredSearchTerm, selectedSource, selectedTopic]);

  useEffect(() => {
    if (filtered.length === 0) {
      setSelectedArticleId(null);
      return;
    }
    if (!filtered.some((article) => article.id === selectedArticleId)) {
      setSelectedArticleId(filtered[0].id);
    }
  }, [filtered, selectedArticleId]);

  const featured = filtered.find((article) => article.id === selectedArticleId) ?? filtered[0] ?? null;

  const loadDocDetail = useCallback(async (documentId: string) => {
    const docId = String(documentId || "").trim();
    if (!docId) return;
    if (docDetails[docId] || docDetailLoading[docId]) return;

    setDocDetailLoading((prev) => ({ ...prev, [docId]: true }));
    setDocDetailError((prev) => ({ ...prev, [docId]: "" }));
    try {
      const detail = await fetchJson<DocumentDetailData>(`/api/documents/${encodeURIComponent(docId)}`);
      setDocDetails((prev) => ({ ...prev, [docId]: detail }));
    } catch (err) {
      setDocDetailError((prev) => ({
        ...prev,
        [docId]: err instanceof Error ? err.message : "Failed to load analysis.",
      }));
    } finally {
      setDocDetailLoading((prev) => ({ ...prev, [docId]: false }));
    }
  }, [docDetailLoading, docDetails]);

  const loadFeedAnalysis = useCallback(async (article: FeedItem, force = false) => {
    const itemKey = savedArticleId(article);
    if ((!force && feedAnalyses[itemKey]) || feedAnalysisLoading[itemKey]) return;
    const source = getFeedMeta(article.feed_key, article.feed_label);
    const sourceLabel = feedSourceLabel(article, source);
    const topics = topicIndex.topicMatchesByArticleId.get(article.id)?.map((topic) => topic.label) ?? [];

    setFeedAnalysisLoading((prev) => ({ ...prev, [itemKey]: true }));
    setFeedAnalysisError((prev) => ({ ...prev, [itemKey]: "" }));
    try {
      const payload = await fetchJson<{ analysis: FeedItemAnalysis }>("/api/intel/analyze", {
        method: "POST",
        body: JSON.stringify({
          article_id: article.item_type === "document" ? "" : article.id,
          guid: article.guid || "",
          title: decodeEntities(article.title || ""),
          description: decodeEntities(article.description || ""),
          url: article.url || "",
          source: sourceLabel,
          author: article.author || "",
          published_at: feedItemDate(article),
          tone_label: article.tone_label || "",
          topics,
          item_type: article.item_type || "article",
        }),
      });
      setFeedAnalyses((prev) => ({ ...prev, [itemKey]: payload.analysis }));
    } catch (err) {
      setFeedAnalysisError((prev) => ({
        ...prev,
        [itemKey]: err instanceof Error ? err.message : "Failed to generate analysis.",
      }));
    } finally {
      setFeedAnalysisLoading((prev) => ({ ...prev, [itemKey]: false }));
    }
  }, [feedAnalyses, feedAnalysisLoading, topicIndex.topicMatchesByArticleId]);

  const toggleFeedAnalysis = useCallback((article: FeedItem) => {
    const key = savedArticleId(article);
    setExpandedAnalysis((prev) => {
      const shouldOpen = !prev[key];
      if (isMobile) return shouldOpen ? { [key]: true } : {};
      return { ...prev, [key]: shouldOpen };
    });

    if ((!feedAnalyses[key] || shouldRegenerateFeedAnalysis(feedAnalyses[key])) && !feedAnalysisLoading[key]) {
      void loadFeedAnalysis(article, Boolean(feedAnalyses[key]));
    }
    if (article.item_type === "document" && article.document_id && !docDetails[article.document_id] && !docDetailLoading[article.document_id]) {
      void loadDocDetail(article.document_id);
    }
  }, [docDetailLoading, docDetails, feedAnalyses, feedAnalysisLoading, isMobile, loadDocDetail, loadFeedAnalysis]);

  const openFullArticle = useCallback((article: FeedItem) => {
    const docId = article.document_id || "";
    if (!docId) return;
    setFullArticleDocId(docId);
    if (!docDetails[docId] && !docDetailLoading[docId]) {
      void loadDocDetail(docId);
    }
  }, [docDetailLoading, docDetails, loadDocDetail]);

  const fullArticle = fullArticleDocId
    ? feedItems.find((item) => item.document_id === fullArticleDocId) ?? null
    : null;

  useEffect(() => {
    if (!fullArticleDocId) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setFullArticleDocId("");
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [fullArticleDocId]);

  const toggleArticleSave = (article: FeedItem) => {
    const source = getFeedMeta(article.feed_key, article.feed_label);
    const sourceLabel = feedSourceLabel(article, source);
    const primaryTopic = topicIndex.topicMatchesByArticleId.get(article.id)?.[0]?.label;
    savedItems.toggle({
      id: savedArticleId(article),
      type: article.item_type === "document" ? "doc" : "article",
      title: decodeEntities(article.title || "Untitled article"),
      url: article.url,
      source: sourceLabel,
      topic: primaryTopic,
      metadata: {
        feedKey: article.feed_key,
        author: article.author,
        publishedAt: article.published_at || "",
        toneLabel: article.tone_label,
        documentId: article.document_id || "",
        sourceKind: article.source_kind || "",
      },
    });
  };

  const selectSourceFilter = (source: SourceFilter) => {
    setVisibleItemLimit(FEED_RENDER_BATCH_SIZE);
    setSelectedSource(source);
  };

  return (
    <div
      style={{
        minHeight: "82vh",
        color: "#dbe7f5",
        fontFamily: 'var(--font-body), "Segoe UI", sans-serif',
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: isMobile ? "1fr" : "168px minmax(0, 1fr)",
          gap: 0,
          border: "1px solid rgba(99, 127, 170, 0.18)",
          background: "linear-gradient(180deg, rgba(8,16,28,0.96), rgba(9,20,31,0.96))",
          boxShadow: "0 24px 64px rgba(0,0,0,0.28)",
          overflow: "hidden",
        }}
      >
        <aside
          style={{
            borderRight: isMobile ? "none" : "1px solid rgba(99, 127, 170, 0.16)",
            borderBottom: isMobile ? "1px solid rgba(99, 127, 170, 0.16)" : "none",
            padding: isMobile ? "12px" : "14px 10px 18px",
            background: "linear-gradient(180deg, rgba(8,17,29,0.92), rgba(10,21,34,0.98))",
          }}
        >
          {isMobile ? (
            <div style={{ display: "grid", gap: 10 }}>
              <a
                href="/research"
                style={{
                  display: "block",
                  padding: "9px 10px",
                  borderRadius: 8,
                  border: "1px solid rgba(79,213,255,0.15)",
                  background: "rgba(79,213,255,0.05)",
                  color: "#4fd5ff",
                  fontSize: 12,
                  fontWeight: 600,
                  textDecoration: "none",
                  lineHeight: 1.35,
                }}
              >
                Search all regulatory documents
              </a>
              <label style={{ display: "grid", gap: 6 }}>
                <span style={{ color: "#5f7390", fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase" }}>
                  Topics
                </span>
                <select
                  value={selectedTopic}
                  onChange={(e) => setSelectedTopic(e.target.value)}
                  style={{
                    width: "100%",
                    minHeight: 40,
                    background: "rgba(14, 24, 39, 0.9)",
                    border: "1px solid rgba(90, 118, 162, 0.28)",
                    color: "#d9e7f7",
                    borderRadius: 6,
                    padding: "8px 10px",
                    fontSize: 13,
                  }}
                >
                  <option value="ALL">All Topics ({matchedArticles.length})</option>
                  {visibleTopicRules.map((rule) => (
                    <option key={rule.topic_key} value={rule.topic_key}>
                      {rule.label} ({topicIndex.topicCounts.get(rule.topic_key) ?? 0})
                    </option>
                  ))}
                </select>
              </label>
              <label style={{ display: "grid", gap: 6 }}>
                <span style={{ color: "#5f7390", fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase" }}>
                  Sources
                </span>
                <select
                  value={selectedSource}
                  onChange={(e) => selectSourceFilter(e.target.value as SourceFilter)}
                  style={{
                    width: "100%",
                    minHeight: 40,
                    background: "rgba(14, 24, 39, 0.9)",
                    border: "1px solid rgba(90, 118, 162, 0.28)",
                    color: "#d9e7f7",
                    borderRadius: 6,
                    padding: "8px 10px",
                    fontSize: 13,
                  }}
                >
                  <option value="ALL">All Sources ({matchedArticles.length})</option>
                  {SOURCE_FILTERS.map((source) => (
                    <option key={source.key} value={source.key}>
                      {source.label} ({sourceCounts.get(source.key) ?? 0})
                    </option>
                  ))}
                </select>
              </label>
            </div>
          ) : (
            <>
              <a
                href="/research"
                style={{
                  display: "block",
                  marginBottom: 14,
                  padding: "7px 10px",
                  borderRadius: 8,
                  border: "1px solid rgba(79,213,255,0.15)",
                  background: "rgba(79,213,255,0.05)",
                  color: "#4fd5ff",
                  fontSize: 11,
                  fontWeight: 500,
                  textDecoration: "none",
                  lineHeight: 1.35,
                }}
              >
                Search all regulatory documents
              </a>
              <div style={{ color: "#5f7390", fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 10 }}>
                Topics
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <TopicButton
                  label="All Topics"
                  active={selectedTopic === "ALL"}
                  onClick={() => setSelectedTopic("ALL")}
                  count={matchedArticles.length}
                />
                {visibleTopicRules.map((rule) => (
                  <TopicButton
                    key={rule.topic_key}
                    label={rule.label}
                    active={selectedTopic === rule.topic_key}
                    onClick={() => setSelectedTopic(rule.topic_key)}
                    count={topicIndex.topicCounts.get(rule.topic_key) ?? 0}
                  />
                ))}
              </div>
              <div style={{ color: "#5f7390", fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase", margin: "18px 0 10px" }}>
                Sources
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <TopicButton
                  label="All Sources"
                  active={selectedSource === "ALL"}
                  onClick={() => selectSourceFilter("ALL")}
                  count={matchedArticles.length}
                />
                {SOURCE_FILTERS.map((source) => (
                  <TopicButton
                    key={source.key}
                    label={source.label}
                    active={selectedSource === source.key}
                    onClick={() => selectSourceFilter(source.key)}
                    count={sourceCounts.get(source.key) ?? 0}
                  />
                ))}
              </div>
            </>
          )}

          {!isMobile ? <div style={{ marginTop: 22 }}>
            <div style={{ color: "#5f7390", fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 10 }}>
              Legend
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 12, color: "#8d9fb7" }}>
              {Object.entries(TONE_STYLE).map(([key, value]) => (
                <div key={key} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ width: 6, height: 6, borderRadius: 1, background: value.color }} />
                  <span>{value.label}</span>
                </div>
              ))}
            </div>
          </div> : null}
        </aside>

        <main style={{ minWidth: 0 }}>
          <div
            style={{
              display: "flex",
              alignItems: isMobile ? "stretch" : "center",
              justifyContent: "space-between",
              gap: isMobile ? 10 : 16,
              padding: isMobile ? "12px" : "14px 16px 10px",
              borderBottom: "1px solid rgba(99, 127, 170, 0.16)",
              flexWrap: "wrap",
              flexDirection: isMobile ? "column" : "row",
            }}
          >
            <div
              style={{
                color: "#91a8c7",
                fontSize: 11,
                letterSpacing: "0.16em",
                textTransform: "uppercase",
                fontFamily: '"IBM Plex Mono", "SFMono-Regular", Consolas, monospace',
              }}
            >
              News Feed / {selectedRule ? selectedRule.label : "All"} / {selectedSourceLabel} / {filtered.length} matched ({feedItems.length} total)
            </div>

            <div style={{ display: "flex", alignItems: isMobile ? "stretch" : "center", gap: isMobile ? 10 : 14, flexWrap: "wrap" }}>
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="search..."
                style={{
                  width: isMobile ? "100%" : 220,
                  minHeight: isMobile ? 40 : undefined,
                  background: "rgba(14, 24, 39, 0.9)",
                  border: "1px solid rgba(90, 118, 162, 0.18)",
                  color: "#d9e7f7",
                  borderRadius: 5,
                  padding: "8px 12px",
                  fontSize: 12,
                }}
              />
              <div
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 8,
                  color: "#4dd39f",
                  fontSize: 12,
                  letterSpacing: "0.10em",
                  textTransform: "uppercase",
                  fontFamily: '"IBM Plex Mono", "SFMono-Regular", Consolas, monospace',
                }}
              >
                <span style={{ width: 6, height: 6, borderRadius: 999, background: "#4dd39f", boxShadow: "0 0 10px rgba(77,211,159,0.7)" }} />
                <span>Live {formatClock(lastUpdated)}</span>
              </div>
            </div>
          </div>

          {newCount > 0 ? (
            <button
              onClick={() => setNewCount(0)}
              style={{
                margin: "10px 16px 0",
                border: "1px solid rgba(77, 211, 159, 0.25)",
                background: "rgba(77, 211, 159, 0.08)",
                color: "#4dd39f",
                padding: "7px 12px",
                borderRadius: 4,
                fontSize: 12,
                cursor: "pointer",
              }}
            >
              {newCount} new item{newCount === 1 ? "" : "s"} available
            </button>
          ) : null}

          <div style={{ padding: isMobile ? "0 12px 14px" : "12px 16px 18px" }}>
            {!isMobile ? <div
              style={{
                display: "grid",
                gridTemplateColumns: "80px 150px 66px minmax(0, 1fr) 220px 24px",
                gap: 14,
                paddingBottom: 8,
                color: "#5f7390",
                fontSize: 10,
                letterSpacing: "0.18em",
                textTransform: "uppercase",
                fontFamily: '"IBM Plex Mono", "SFMono-Regular", Consolas, monospace',
              }}
            >
              <div>Time</div>
              <div>Source</div>
              <div>Snt</div>
              <div>Headline</div>
              <div style={{ textAlign: "right" }}>Tags</div>
              <div aria-hidden="true" />
            </div> : null}

            {filtered.length === 0 ? (
              <div style={{ color: "#72839d", fontSize: 13, padding: "28px 0" }}>
                {feedItems.length === 0 ? "No feed items yet." : "No feed items match the current filters."}
              </div>
            ) : (
              <>
              {filtered.length > visibleFiltered.length ? (
                <div style={{ color: "#72839d", fontSize: 12, padding: "4px 0 10px" }}>
                  Showing {visibleFiltered.length} of {filtered.length} matched items. Refine search or load more for older results.
                </div>
              ) : null}
              {visibleFiltered.map((article) => {
                const itemKey = savedArticleId(article);
                const matchedTopicsForArticle = topicIndex.topicMatchesByArticleId.get(article.id) ?? [];
                const analysisOpen = !!expandedAnalysis[itemKey];
                const docId = article.document_id || "";
                const detailLoading = docId ? !!docDetailLoading[docId] : false;
                const detailError = docId ? docDetailError[docId] || "" : "";
                const itemAnalysisLoading = !!feedAnalysisLoading[itemKey];
                const itemAnalysisError = feedAnalysisError[itemKey] || "";
                const analysisLabel = itemAnalysisLoading || detailLoading ? "Analyzing..." : analysisOpen ? "Hide Analysis" : "Open Analysis";

                return (
                  <Fragment key={itemKey}>
                    {article.id === featured?.id ? (
                      <FeaturedCard
                        article={article}
                        matchedTopics={matchedTopicsForArticle}
                        saved={savedItems.isSaved(itemKey)}
                        onToggleSave={() => toggleArticleSave(article)}
                        analysisOpen={analysisOpen}
                        analysisLabel={analysisLabel}
                        onToggleAnalysis={() => toggleFeedAnalysis(article)}
                        onOpenFullArticle={() => openFullArticle(article)}
                        compact={isMobile}
                      />
                    ) : (
                      <FeedRow
                        article={article}
                        matchedTopics={matchedTopicsForArticle}
                        active={article.id === selectedArticleId}
                        onSelect={() => setSelectedArticleId(article.id)}
                        saved={savedItems.isSaved(itemKey)}
                        onToggleSave={() => toggleArticleSave(article)}
                        analysisOpen={analysisOpen}
                        analysisLabel={analysisLabel}
                        onToggleAnalysis={() => toggleFeedAnalysis(article)}
                        onOpenFullArticle={() => openFullArticle(article)}
                        compact={isMobile}
                      />
                    )}
                    {analysisOpen ? (
                      <div
                        style={{
                          borderTop: "1px solid rgba(112, 142, 187, 0.12)",
                          background: "rgba(5, 13, 23, 0.62)",
                          padding: isMobile ? "12px 0 16px" : "14px 16px 16px",
                        }}
                      >
                        <FeedAnalysisPanel
                          article={article}
                          matchedTopics={matchedTopicsForArticle}
                          analysis={feedAnalyses[itemKey]}
                          analysisLoading={itemAnalysisLoading}
                          analysisError={itemAnalysisError}
                          retryAnalysis={() => void loadFeedAnalysis(article, true)}
                          detail={docId ? docDetails[docId] : undefined}
                          loading={detailLoading}
                          error={detailError}
                          retry={() => {
                            if (docId) void loadDocDetail(docId);
                          }}
                          compact={isMobile}
                        />
                      </div>
                    ) : null}
                  </Fragment>
                );
              })}
              {hasMoreFiltered ? (
                <div style={{ display: "flex", justifyContent: "center", padding: "16px 0 4px" }}>
                  <button
                    type="button"
                    onClick={() => setVisibleItemLimit((limit) => limit + FEED_RENDER_BATCH_SIZE)}
                    style={{
                      minHeight: 36,
                      border: "1px solid rgba(79,213,255,0.28)",
                      background: "rgba(79,213,255,0.08)",
                      color: "#d9e7f7",
                      borderRadius: 8,
                      padding: "8px 14px",
                      fontSize: 12,
                      fontWeight: 700,
                      cursor: "pointer",
                    }}
                  >
                    Load {Math.min(FEED_RENDER_BATCH_SIZE, filtered.length - visibleFiltered.length)} more
                  </button>
                </div>
              ) : null}
              </>
            )}
          </div>
        </main>
      </div>

      {fullArticle ? (
        <FullArticleModal
          article={fullArticle}
          detail={fullArticleDocId ? docDetails[fullArticleDocId] : undefined}
          loading={fullArticleDocId ? !!docDetailLoading[fullArticleDocId] : false}
          error={fullArticleDocId ? docDetailError[fullArticleDocId] || "" : ""}
          onClose={() => setFullArticleDocId("")}
          retry={() => {
            if (fullArticleDocId) void loadDocDetail(fullArticleDocId);
          }}
          compact={isMobile}
        />
      ) : null}

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          gap: 8,
          flexDirection: isMobile ? "column" : "row",
          padding: "10px 4px 0",
          color: "#5d708a",
          fontSize: 11,
        }}
      >
        <div>Updated {formatUpdated(lastUpdated.toISOString())}</div>
        <div>{articles.length} articles + {documentFeedItems.length} research documents</div>
      </div>
    </div>
  );
}
