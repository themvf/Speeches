export type RssArticle = {
  guid: string;
  title: string;
  url: string;
  description: string;
  author: string;
  publishedAt: Date | null;
};

export const WSJ_FEEDS: Record<string, { label: string; feedUrl: string }> = {
  wsj_us_business: {
    label: "WSJ US Business",
    feedUrl: "https://feeds.content.dowjones.io/public/rss/WSJcomUSBusinessNews",
  },
  wsj_markets: {
    label: "WSJ Markets",
    feedUrl: "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
  },
  wsj_opinion: {
    label: "WSJ Opinion",
    feedUrl: "https://feeds.content.dowjones.io/public/rss/RSSOpinion",
  },
  mw_top_stories: {
    label: "MarketWatch Top Stories",
    feedUrl: "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
  },
};

export const DEFAULT_RSS_FEEDS: Record<string, { label: string; feedUrl: string }> = {
  ...WSJ_FEEDS,
  sec_press_releases: {
    label: "SEC Press Releases",
    feedUrl: "https://www.sec.gov/news/pressreleases.rss",
  },
  sec_speeches_statements: {
    label: "SEC Speeches and Statements",
    feedUrl: "https://www.sec.gov/news/speeches-statements.rss",
  },
  sec_litigation_releases: {
    label: "SEC Litigation Releases",
    feedUrl: "https://www.sec.gov/enforcement-litigation/litigation-releases/rss",
  },
  sec_administrative_proceedings: {
    label: "SEC Administrative Proceedings",
    feedUrl: "https://www.sec.gov/enforcement-litigation/administrative-proceedings/rss",
  },
  sec_trading_suspensions: {
    label: "SEC Trading Suspensions",
    feedUrl: "https://www.sec.gov/enforcement-litigation/trading-suspensions/rss",
  },
  finra_notices: {
    label: "FINRA Regulatory Notices",
    feedUrl: "http://feeds.finra.org/FINRANotices",
  },
  finra_rule_filings: {
    label: "FINRA Rule Filings",
    feedUrl: "http://feeds.finra.org/FINRARuleFilings",
  },
  finra_dispute_resolution_rule_filings: {
    label: "FINRA Dispute Resolution Rule Filings",
    feedUrl: "http://feeds.finra.org/DisputeResolutionRuleFilings",
  },
  finra_news: {
    label: "FINRA News Releases and Speeches",
    feedUrl: "http://feeds.finra.org/FINRANews",
  },
  finra_upc_advisories: {
    label: "FINRA UPC Advisories",
    feedUrl: "http://feeds.finra.org/FINRAUPCAdvisories",
  },
  cftc_general_press_releases: {
    label: "CFTC General Press Releases",
    feedUrl: "https://www.cftc.gov/RSS/RSSGP/rssgp.xml",
  },
  cftc_enforcement_press_releases: {
    label: "CFTC Enforcement Press Releases",
    feedUrl: "https://www.cftc.gov/RSS/RSSENF/rssenf.xml",
  },
  cftc_speeches_testimony: {
    label: "CFTC Speeches and Testimony",
    feedUrl: "https://www.cftc.gov/RSS/RSSST/rssst.xml",
  },
  cftc_federal_register_proposed_rules: {
    label: "CFTC Federal Register Proposed Rules",
    feedUrl: "http://comments.cftc.gov/handlers/RSSHandler.ashx?type=Releases&category=Proposed%20Rule",
  },
  cftc_federal_register_final_rules: {
    label: "CFTC Federal Register Final Rules",
    feedUrl: "http://comments.cftc.gov/handlers/RSSHandler.ashx?type=Releases&category=Final%20Rule",
  },
  fed_all_press_releases: {
    label: "Federal Reserve All Press Releases",
    feedUrl: "https://www.federalreserve.gov/feeds/press_all.xml",
  },
  fed_banking_consumer_regulatory_policy: {
    label: "Federal Reserve Banking and Consumer Regulatory Policy",
    feedUrl: "https://www.federalreserve.gov/feeds/press_bcreg.xml",
  },
  fed_enforcement_actions: {
    label: "Federal Reserve Enforcement Actions",
    feedUrl: "https://www.federalreserve.gov/feeds/press_enforcement.xml",
  },
  fed_supervision_regulation_letters: {
    label: "Federal Reserve Supervision and Regulation Letters",
    feedUrl: "https://www.federalreserve.gov/feeds/bankinginfo-rss.xml",
  },
  occ_news_releases: {
    label: "OCC News Releases",
    feedUrl: "https://www.occ.gov/rss/occ_news.xml",
  },
  occ_bulletins: {
    label: "OCC Bulletins",
    feedUrl: "https://www.occ.gov/rss/occ_bulletins.xml",
  },
  occ_speeches: {
    label: "OCC Speeches",
    feedUrl: "https://www.occ.gov/rss/occ-speeches.xml",
  },
  occ_congressional_testimony: {
    label: "OCC Congressional Testimony",
    feedUrl: "https://www.occ.gov/rss/occ-congressional-testimony.xml",
  },
  cfpb_newsroom: {
    label: "CFPB Newsroom",
    feedUrl: "https://www.consumerfinance.gov/about-us/newsroom/feed/",
  },
  ftc_consumer_protection_press_releases: {
    label: "FTC Consumer Protection Press Releases",
    feedUrl: "https://www.ftc.gov/feeds/press-release-consumer-protection.xml",
  },
  coindesk: {
    label: "CoinDesk",
    feedUrl: "https://www.coindesk.com/arc/outboundfeeds/rss/",
  },
  cointelegraph: {
    label: "Cointelegraph",
    feedUrl: "https://cointelegraph.com/rss",
  },
  decrypt: {
    label: "Decrypt",
    feedUrl: "https://decrypt.co/feed",
  },
  the_block: {
    label: "The Block",
    feedUrl: "https://www.theblock.co/rss.xml",
  },
  cisa_cybersecurity_advisories: {
    label: "CISA Cybersecurity Advisories",
    feedUrl: "https://www.cisa.gov/cybersecurity-advisories/all.xml",
  },
  krebs_on_security: {
    label: "Krebs on Security",
    feedUrl: "https://krebsonsecurity.com/feed/",
  },
  gibson_dunn_sec_sentinel: {
    label: "Gibson Dunn SEC Sentinel",
    feedUrl: "https://secsentinel.gibsondunn.com/feed/",
  },
  gibson_dunn_securities_regulation_monitor: {
    label: "Gibson Dunn Securities Regulation and Corporate Governance Monitor",
    feedUrl: "https://themonitor.gibsondunn.com/feed/",
  },
  cleary_enforcement_watch: {
    label: "Cleary Enforcement Watch",
    feedUrl: "https://www.clearyenforcementwatch.com/feed/",
  },
  cooley_pubco: {
    label: "Cooley PubCo",
    feedUrl: "https://cooleypubco.com/feed/",
  },
  cooley_cyber_data_privacy: {
    label: "Cooley Cyber/Data/Privacy",
    feedUrl: "https://cdp.cooley.com/feed/",
  },
  cooley_governance_beat: {
    label: "Cooley Governance Beat",
    feedUrl: "https://governancebeat.cooley.com/feed/",
  },
  latham_global_financial_regulatory_blog: {
    label: "Latham Global Financial Regulatory Blog",
    feedUrl: "https://www.globalfinregblog.com/feed/",
  },
  latham_london: {
    label: "Latham.London",
    feedUrl: "https://www.latham.london/feed/",
  },
  covington_inside_privacy: {
    label: "Covington Inside Privacy",
    feedUrl: "https://www.insideprivacy.com/feed/",
  },
  covington_global_policy_watch: {
    label: "Covington Global Policy Watch",
    feedUrl: "https://www.globalpolicywatch.com/feed/",
  },
  covington_inside_government_contracts: {
    label: "Covington Inside Government Contracts",
    feedUrl: "https://www.insidegovernmentcontracts.com/feed/",
  },
  ballard_spahr_consumer_finance_monitor: {
    label: "Ballard Spahr Consumer Finance Monitor",
    feedUrl: "https://www.consumerfinancemonitor.com/feed/",
  },
  kelley_drye_ad_law_access: {
    label: "Kelley Drye Ad Law Access",
    feedUrl: "https://www.kelleydrye.com/viewpoints/blogs/ad-law-access/rss",
  },
  norton_rose_fulbright_data_protection_report: {
    label: "Norton Rose Fulbright Data Protection Report",
    feedUrl: "https://www.dataprotectionreport.com/feed/",
  },
  squire_patton_boggs_privacy_world: {
    label: "Squire Patton Boggs Privacy World",
    feedUrl: "https://www.privacyworld.blog/feed/",
  },
  bradley_financial_services_perspectives: {
    label: "Bradley Financial Services Perspectives",
    feedUrl: "https://www.financialservicesperspectives.com/feed/",
  },
  bradley_eye_on_enforcement: {
    label: "Bradley Eye on Enforcement",
    feedUrl: "https://www.eyeonenforcement.com/feed/",
  },
};

function decodeEntities(text: string): string {
  return text
    .replace(/&#x([0-9a-fA-F]+);/gi, (_, hex) => String.fromCharCode(parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, dec) => String.fromCharCode(parseInt(dec, 10)))
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&nbsp;/g, " ");
}

function extractTag(xml: string, tag: string): string {
  const cdataRe = new RegExp(`<${tag}[^>]*><!\\[CDATA\\[([\\s\\S]*?)\\]\\]><\\/${tag}>`, "i");
  const plainRe = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, "i");
  const m = xml.match(cdataRe) ?? xml.match(plainRe);
  return m ? m[1].trim() : "";
}

function extractAttr(xml: string, tag: string, attr: string): string {
  const re = new RegExp(`<${tag}[^>]*\\s${attr}="([^"]*)"`, "i");
  const m = xml.match(re);
  return m ? m[1].trim() : "";
}

function stripHtml(text: string): string {
  return text.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
}

function parseRssDate(text: string): Date | null {
  if (!text) return null;
  const d = new Date(text);
  return Number.isFinite(d.getTime()) ? d : null;
}

function normalizeGuid(raw: string, fallbackUrl: string, title: string): string {
  const s = raw.trim() || fallbackUrl.trim();
  if (s) return s;
  // Deterministic fallback so the same article always gets the same GUID
  const seed = `${title}:${fallbackUrl}`.slice(0, 200);
  let h = 0;
  for (let i = 0; i < seed.length; i++) { h = (Math.imul(31, h) + seed.charCodeAt(i)) | 0; }
  return `rss:fallback:${(h >>> 0).toString(16)}`;
}

export async function fetchRssFeed(feedUrl: string, maxItems = 50, timeoutMs = 10_000): Promise<RssArticle[]> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let resp: Response;
  try {
    resp = await fetch(feedUrl, {
      headers: { "User-Agent": "Mozilla/5.0 (compatible; PolicyHubBot/1.0)" },
      next: { revalidate: 0 },
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timer);
  }
  if (!resp.ok) throw new Error(`RSS fetch failed: ${resp.status} ${feedUrl}`);
  const xml = await resp.text();
  if (xml.length > 2_000_000) throw new Error(`RSS feed response too large (${xml.length} bytes): ${feedUrl}`);

  const itemRe = /<item[\s>]([\s\S]*?)<\/item>/gi;
  const results: RssArticle[] = [];
  let match: RegExpExecArray | null;

  while ((match = itemRe.exec(xml)) !== null && results.length < maxItems) {
    const block = match[1];
    const title = decodeEntities(stripHtml(extractTag(block, "title")));
    const url = extractTag(block, "link") || extractAttr(block, "link", "href");
    const description = decodeEntities(stripHtml(extractTag(block, "description") || extractTag(block, "summary")));
    const author = decodeEntities(extractTag(block, "dc:creator") || extractTag(block, "author"));
    const pubDate = extractTag(block, "pubDate") || extractTag(block, "published") || extractTag(block, "updated");
    const guid = normalizeGuid(extractTag(block, "guid"), url, title);

    if (!title || !url) continue;

    results.push({
      guid,
      title,
      url,
      description,
      author,
      publishedAt: parseRssDate(pubDate),
    });
  }

  return results;
}
