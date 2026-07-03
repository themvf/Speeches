import { decodeEntities, getTopicMatches, normalizeMatchText, normalizeTopicRules, type TopicArticleInput, type TopicRuleInput } from "@/lib/intel-topic-matching";
import { isEnglishRssArticle, shouldEnglishOnlyFilterFeed } from "@/lib/server/rss-language-filter";
import type { RssArticle } from "@/lib/server/rss-fetcher";

const KEYWORD_FILTERED_FEED_PREFIXES = ["prnewswire_", "google_news_"];
const NOISY_RELEVANCE_FILTERED_FEED_PREFIXES = ["prnewswire_"];
const US_ONLY_FRAUD_FEED_KEYS = new Set(["google_news_ponzi_investor_fraud"]);
const REQUIRED_TOPIC_KEYS_BY_FEED_KEY: Record<string, string[]> = {
  google_news_ponzi_investor_fraud: ["PONZI_INVESTOR_FRAUD"],
};
const GAMBLING_RE = /\b(?:gambling|casino|casinos|slot|slots|sportsbook|sportsbooks|wager|wagering|betting|bookmaker|bookmakers|lottery|lotteries|poker|blackjack|roulette|sweepstakes)\b/i;
const GAMBLING_ALLOWED_RE = /\b(?:prediction market|prediction markets|event contract|event contracts|prediction exchange|kalshi|polymarket|predictit|cftc|commodity futures trading commission|binary option|binary options|securities|security-based|sec|securities and exchange commission|broker-dealer|exchange act|securities act|market structure)\b/i;
const PRNEWSWIRE_LEGAL_SOLICITATION_RE = /\b(?:shareholder alert|investor alert|investor deadline|deadline alert|investor notice|shareholder notice|class action attorney|class action law firm|m&a class action firm|investor rights law firm|securities litigation firm|law offices? of|lead plaintiff|lead plaintiff deadline|lead class action|opportunity to lead|appointment as lead plaintiff|securities class action|securities lawsuit|class action lawsuit|class action investigation|announces the filing of a class action|announces that a class action|class action (?:has been )?filed|encourages .* investors to (?:inquire|contact|secure counsel)|investors? (?:with )?(?:substantial )?losses? (?:have|has) opportunity|investigating claims|continues to investigate|announces investigation of|reminds investors|encourages investors|contact the firm|if you (?:purchased|acquired|bought) .* securities|seek appointment as lead plaintiff|upcoming deadline)\b/i;
const NOISY_SOURCE_RELEVANCE_RE = /\b(?:sec|securities and exchange commission|cftc|commodity futures trading commission|finra|federal reserve|treasury|ofac|fincen|cfpb|occ|fdic|ftc|doj|pcaob|msrb|securities|security-based|investment adviser|investment advisor|broker-dealer|investor fraud|investment fraud|ponzi|offering fraud|public offering|registered offering|private placement|initial public offering|ipo|etf|mutual fund|hedge fund|private fund|asset manager|wealth management|shareholder vote|shareholder lawsuit|shareholder rights|stock exchange|equity market|derivative|swap|futures|options|anti-money laundering|aml|kyc|sanctions compliance|illicit finance|cybersecurity|ransomware|data breach|operational resilience|reg sci|crypto asset|digital asset|stablecoin|bitcoin|ethereum|tokenized securities|blockchain|prediction market|event contract|kalshi|polymarket|predictit|binary options|enforcement action|investigation|settlement|lawsuit|complaint|civil penalty|market manipulation|insider trading|regulatory compliance|compliance platform|risk management)\b/i;
const AI_CONTEXT_RE = /\b(?:financial services|bank|banking|securities|investment|investor|broker|adviser|advisor|regulatory|compliance|risk management|governance|cybersecurity|privacy|fraud|aml|kyc|trading|market surveillance)\b/i;
const GEOPOLITICAL_CONTEXT_RE = /\b(?:tariff|tariffs|trade policy|export controls|import restrictions|sanctions|supply chain|national security|foreign policy|shipping lanes|maritime security|semiconductor controls|cross-border restrictions|trade war)\b/i;
const ECONOMIC_CONTEXT_RE = /\b(?:gdp|inflation|cpi|pce|interest rate|rate cut|rate hike|federal reserve|fomc|monetary policy|recession|labor market|jobs report|unemployment|tariff|fiscal policy)\b/i;
const US_ABBREVIATION_JURISDICTION_RE = /\b(?:U\.S\.|U\.S|US|USA)\b/;
const US_FRAUD_JURISDICTION_RE = /\b(?:united states|american|securities and exchange commission|department of justice|federal bureau of investigation|internal revenue service|commodity futures trading commission|federal trade commission|consumer financial protection bureau|financial industry regulatory authority|u\.s\. attorney|us attorney|sec charges|sec sues|sec settles|doj charges|fbi|finra|cftc|cfpb|ftc|irs|fdic|alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|district of columbia|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|new mexico|new york|north carolina|north dakota|ohio|oklahoma|oregon|pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|utah|vermont|virginia|washington|west virginia|wisconsin|wyoming|manhattan|brooklyn|bronx|queens|los angeles|san francisco|san diego|miami|tampa|orlando|atlanta|chicago|boston|philadelphia|dallas|houston|austin|phoenix|seattle|denver|las vegas|charlotte)\b/i;

export type RssIngestionFilterResult = {
  articles: RssArticle[];
  fetched: number;
  matched: number;
  filtered: number;
};

export type RssFilterArticle = TopicArticleInput & {
  author?: string | null;
  url?: string | null;
  guid?: string | null;
  publishedAt?: Date | null;
};

export function shouldKeywordFilterFeed(feedKey: string): boolean {
  const key = String(feedKey || "").trim().toLowerCase();
  return KEYWORD_FILTERED_FEED_PREFIXES.some((prefix) => key.startsWith(prefix));
}

function shouldNoisyRelevanceFilterFeed(feedKey: string): boolean {
  const key = String(feedKey || "").trim().toLowerCase();
  return NOISY_RELEVANCE_FILTERED_FEED_PREFIXES.some((prefix) => key.startsWith(prefix));
}

function shouldRequireUsFraudJurisdiction(feedKey: string): boolean {
  const key = String(feedKey || "").trim().toLowerCase();
  return US_ONLY_FRAUD_FEED_KEYS.has(key);
}

export function rssFetchLimitForFeed(feedKey: string): number {
  return shouldKeywordFilterFeed(feedKey) ? 100 : 50;
}

function articleText(article: RssFilterArticle): string {
  return decodeEntities([
    article.title,
    article.description,
    article.author,
    article.url,
  ].filter(Boolean).join(" "));
}

export function isDisallowedGamblingArticle(article: RssFilterArticle): boolean {
  const text = articleText(article);
  return GAMBLING_RE.test(text) && !GAMBLING_ALLOWED_RE.test(text);
}

export function isDisallowedNoisySourceArticle(feedKey: string, article: RssFilterArticle): boolean {
  if (!shouldNoisyRelevanceFilterFeed(feedKey)) return false;
  return PRNEWSWIRE_LEGAL_SOLICITATION_RE.test(articleText(article));
}

export function hasRequiredFraudJurisdiction(feedKey: string, article: RssFilterArticle): boolean {
  if (!shouldRequireUsFraudJurisdiction(feedKey)) return true;
  const text = articleText(article);
  return US_ABBREVIATION_JURISDICTION_RE.test(text) || US_FRAUD_JURISDICTION_RE.test(text);
}

function passesNoisyRelevanceGate(feedKey: string, article: RssFilterArticle, topicKeys: string[]): boolean {
  if (!shouldNoisyRelevanceFilterFeed(feedKey)) return true;

  const text = articleText(article);
  if (NOISY_SOURCE_RELEVANCE_RE.test(text)) return true;

  const normalized = normalizeMatchText(text);
  if (topicKeys.includes("PONZI_INVESTOR_FRAUD") || topicKeys.includes("AML")) return true;
  if (topicKeys.includes("AI_TECH") && AI_CONTEXT_RE.test(normalized)) return true;
  if (topicKeys.includes("GEOPOLITICAL_TRADE_RISK") && GEOPOLITICAL_CONTEXT_RE.test(normalized)) return true;
  if (topicKeys.includes("ECONOMIC_GROWTH") && ECONOMIC_CONTEXT_RE.test(normalized)) return true;

  return false;
}

export function isAllowedRssArticleForIngestion(
  feedKey: string,
  article: RssFilterArticle,
  topicRules: TopicRuleInput[]
): boolean {
  if (shouldEnglishOnlyFilterFeed(feedKey) && !isEnglishRssArticle(article)) return false;
  if (isDisallowedGamblingArticle(article)) return false;
  if (isDisallowedNoisySourceArticle(feedKey, article)) return false;
  if (!hasRequiredFraudJurisdiction(feedKey, article)) return false;

  if (!shouldKeywordFilterFeed(feedKey)) {
    return true;
  }

  const rules = normalizeTopicRules(topicRules);
  if (rules.length === 0) {
    return false;
  }

  const matches = getTopicMatches(article, rules);
  if (matches.length === 0) return false;

  const requiredTopicKeys = REQUIRED_TOPIC_KEYS_BY_FEED_KEY[String(feedKey || "").trim().toLowerCase()] || [];
  const topicKeys = matches.map((match) => match.rule.topic_key);
  if (requiredTopicKeys.length > 0 && !topicKeys.some((topicKey) => requiredTopicKeys.includes(topicKey))) {
    return false;
  }

  return passesNoisyRelevanceGate(feedKey, article, topicKeys);
}

export function filterRssArticlesForIngestion(
  feedKey: string,
  articles: RssArticle[],
  topicRules: TopicRuleInput[]
): RssIngestionFilterResult {
  if (!shouldKeywordFilterFeed(feedKey) && !shouldEnglishOnlyFilterFeed(feedKey)) {
    const allowedArticles = articles.filter((article) => !isDisallowedGamblingArticle(article));
    return {
      articles: allowedArticles,
      fetched: articles.length,
      matched: allowedArticles.length,
      filtered: articles.length - allowedArticles.length,
    };
  }

  const filteredArticles = articles.filter((article) => isAllowedRssArticleForIngestion(feedKey, article, topicRules));
  if (shouldKeywordFilterFeed(feedKey) && normalizeTopicRules(topicRules).length === 0) {
    return {
      articles: [],
      fetched: articles.length,
      matched: 0,
      filtered: articles.length,
    };
  }
  return {
    articles: filteredArticles,
    fetched: articles.length,
    matched: filteredArticles.length,
    filtered: articles.length - filteredArticles.length,
  };
}
