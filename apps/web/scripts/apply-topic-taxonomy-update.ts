import { neon } from "@neondatabase/serverless";

type TopicRuleUpdate = {
  topicKey: string;
  label: string;
  keywords: string[];
  sortOrder: number;
};

const predictionMarketKeywords = [
  "prediction market",
  "prediction markets",
  "event contract",
  "event contracts",
  "binary options",
  "binary event",
  "prediction exchange",
  "betting market",
  "forecast market",
  "Kalshi",
  "Polymarket",
  "PredictIt",
  "CFTC event contracts",
  "CFTC prediction markets",
  "political betting",
  "sports event contract",
  "odds contract",
  "Commodity Futures Trading Commission",
  "CME Group",
  "Robinhood",
  "prediction-markets",
];

const newTopics: TopicRuleUpdate[] = [
  {
    topicKey: "COMMODITIES_ENERGY_MARKETS",
    label: "Commodities & Energy Markets",
    sortOrder: 160,
    keywords: [
      "commodities",
      "commodities futures",
      "commodity futures",
      "oil prices",
      "crude oil",
      "brent crude",
      "WTI",
      "natural gas",
      "energy markets",
      "energy prices",
      "power markets",
      "electricity markets",
      "CFTC",
      "Strait of Hormuz",
      "shipping chokepoint",
      "supply disruption",
      "geopolitical risk premium",
    ],
  },
  {
    topicKey: "GEOPOLITICAL_TRADE_RISK",
    label: "Geopolitical & Trade Risk",
    sortOrder: 170,
    keywords: [
      "geopolitical risk",
      "trade policy",
      "tariff",
      "tariffs",
      "export controls",
      "import restrictions",
      "supply chain",
      "supply chains",
      "sanctions risk",
      "foreign policy",
      "shipping lanes",
      "Strait of Hormuz",
      "Iran",
      "China",
      "national security",
      "cross-border restrictions",
      "trade war",
      "maritime security",
    ],
  },
  {
    topicKey: "SRO_RULEMAKING_ARBITRATION",
    label: "SRO Rulemaking & Arbitration",
    sortOrder: 180,
    keywords: [
      "proposed rule change",
      "rule filing",
      "SR-FINRA",
      "SR-CBOE",
      "SR-NYSE",
      "SR-NASDAQ",
      "FINRA arbitration",
      "dispute resolution",
      "arbitration procedure",
      "self-regulatory organization",
      "SRO rulemaking",
      "FINRA rules",
      "arbitration forum",
      "customer arbitration",
      "industry arbitration",
    ],
  },
  {
    topicKey: "BANKING_PAYMENTS",
    label: "Banking & Payments",
    sortOrder: 190,
    keywords: [
      "bank",
      "banking",
      "deposits",
      "deposit insurance",
      "payments",
      "payment rails",
      "card network",
      "ACH",
      "wire transfer",
      "cross-border payments",
      "real-time payments",
      "instant payments",
      "bank merger",
      "bank capital",
      "liquidity coverage",
      "net interest income",
      "Basel",
      "Federal Reserve supervision",
    ],
  },
  {
    topicKey: "CONSUMER_PROTECTION_DECEPTIVE_PRACTICES",
    label: "Consumer Protection & Deceptive Practices",
    sortOrder: 200,
    keywords: [
      "FTC",
      "consumer protection",
      "deceptive advertising",
      "deceptive claims",
      "subscription scheme",
      "unfair practices",
      "unfair or deceptive",
      "junk fees",
      "refunds",
      "consumer fraud",
      "misleading claims",
      "negative option",
      "dark patterns",
      "telemarketing",
      "consumer redress",
    ],
  },
  {
    topicKey: "DATA_PRIVACY_DIGITAL_IDENTITY",
    label: "Data Privacy & Digital Identity",
    sortOrder: 210,
    keywords: [
      "data privacy",
      "personal data",
      "consumer data",
      "digital identity",
      "identity verification",
      "eID",
      "biometrics",
      "privacy rule",
      "data broker",
      "data security",
      "identity fraud",
      "AI identity",
      "personal identification code",
      "authentication",
      "credential theft",
    ],
  },
  {
    topicKey: "INVESTMENT_PRODUCTS_DERIVATIVES",
    label: "Investment Products & Derivatives",
    sortOrder: 220,
    keywords: [
      "ETF",
      "exchange-traded fund",
      "options",
      "futures",
      "swaps",
      "derivatives",
      "structured product",
      "annuity",
      "mutual fund",
      "closed-end fund",
      "interval fund",
      "leveraged ETF",
      "inverse ETF",
      "covered call ETF",
      "yield product",
      "retail structured notes",
    ],
  },
];

function formatKeywords(keywords: string[]): string {
  return keywords.join(", ");
}

async function main(): Promise<void> {
  const databaseUrl = process.env.DATABASE_URL;
  if (!databaseUrl) {
    throw new Error("DATABASE_URL is required.");
  }

  const sql = neon(databaseUrl);

  await sql`
    UPDATE rss_topic_rules
    SET
      keywords = ${formatKeywords(predictionMarketKeywords)},
      active = true,
      sort_order = 100,
      updated_at = NOW()
    WHERE topic_key = 'PREDICTION_MARKETS'
  `;

  await sql`
    UPDATE rss_topic_rules
    SET active = false, updated_at = NOW()
    WHERE topic_key = 'PREMARKETS'
  `;

  for (const topic of newTopics) {
    await sql`
      INSERT INTO rss_topic_rules (topic_key, label, keywords, active, sort_order, updated_at)
      VALUES (${topic.topicKey}, ${topic.label}, ${formatKeywords(topic.keywords)}, true, ${topic.sortOrder}, NOW())
      ON CONFLICT (topic_key) DO UPDATE
      SET
        label = EXCLUDED.label,
        keywords = EXCLUDED.keywords,
        active = true,
        sort_order = EXCLUDED.sort_order,
        updated_at = NOW()
    `;
  }

  const rows = await sql`
    SELECT topic_key, label, active, sort_order
    FROM rss_topic_rules
    WHERE topic_key IN (
      'PREMARKETS',
      'PREDICTION_MARKETS',
      'COMMODITIES_ENERGY_MARKETS',
      'GEOPOLITICAL_TRADE_RISK',
      'SRO_RULEMAKING_ARBITRATION',
      'BANKING_PAYMENTS',
      'CONSUMER_PROTECTION_DECEPTIVE_PRACTICES',
      'DATA_PRIVACY_DIGITAL_IDENTITY',
      'INVESTMENT_PRODUCTS_DERIVATIVES'
    )
    ORDER BY sort_order ASC, label ASC
  `;

  console.log(JSON.stringify({ ok: true, updated: rows }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
