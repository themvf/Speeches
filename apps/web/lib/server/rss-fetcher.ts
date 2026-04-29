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

function normalizeGuid(raw: string, fallbackUrl: string): string {
  const s = raw.trim() || fallbackUrl.trim();
  return s || `rss:${Date.now()}:${Math.random()}`;
}

export async function fetchRssFeed(feedUrl: string, maxItems = 50): Promise<RssArticle[]> {
  const resp = await fetch(feedUrl, {
    headers: { "User-Agent": "Mozilla/5.0 (compatible; PolicyHubBot/1.0)" },
    next: { revalidate: 0 },
  });
  if (!resp.ok) throw new Error(`RSS fetch failed: ${resp.status} ${feedUrl}`);
  const xml = await resp.text();

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
    const guid = normalizeGuid(extractTag(block, "guid"), url);

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
