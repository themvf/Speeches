export type FinraMemberFirmAffiliateAliasDefinition = {
  firmName: string;
  aliases: string[];
  sourceUrls: string[];
  notes?: string;
};

export const FINRA_MEMBER_FIRM_AFFILIATE_ALIASES = [
  {
    firmName: "ETORO USA SECURITIES INC.",
    aliases: ["eToro", "eToro USA", "eToro Securities"],
    sourceUrls: [
      "https://brokercheck.finra.org/firm/summary/298361",
      "https://www.etoro.com/en-us/customer-service/disclosures/inapp/",
    ],
    notes: "eToro's U.S. securities broker-dealer; crypto services are separate from FINRA/SIPC brokerage coverage.",
  },
  {
    firmName: "KRAKEN SECURITIES",
    aliases: ["Kraken", "Kraken Securities LLC", "Payward", "Payward Inc.", "Payward, Inc."],
    sourceUrls: [
      "https://brokercheck.finra.org/firm/summary/324343",
      "https://www.kraken.com/",
      "https://www.kraken.com/social-media-disclosure",
    ],
    notes: "Kraken Securities is the FINRA/SIPC broker-dealer affiliate; Payward Interactive crypto services are not FINRA/SIPC.",
  },
  {
    firmName: "ROBINHOOD FINANCIAL, LLC",
    aliases: ["Robinhood", "Robinhood Financial"],
    sourceUrls: [
      "https://brokercheck.finra.org/firm/summary/165998",
      "https://robinhood.com/us/en/about/legal/",
    ],
  },
  {
    firmName: "ROBINHOOD SECURITIES, LLC",
    aliases: ["Robinhood Securities"],
    sourceUrls: [
      "https://brokercheck.finra.org/",
      "https://robinhood.com/us/en/about/legal/",
    ],
  },
  {
    firmName: "WEBULL FINANCIAL LLC",
    aliases: ["Webull", "Webull Financial"],
    sourceUrls: [
      "https://brokercheck.finra.org/firm/summary/289063",
      "https://www.webull.com/disclosures",
      "https://www.webull.com/policy",
    ],
  },
  {
    firmName: "WEBULL SECURITIES (US) LLC",
    aliases: ["Webull Securities", "Webull Securities US", "Webull Securities (US)"],
    sourceUrls: [
      "https://brokercheck.finra.org/",
      "https://www.webull.com/disclosures",
    ],
  },
  {
    firmName: "SOFI SECURITIES LLC",
    aliases: ["SoFi", "SoFi Invest", "SoFi Securities"],
    sourceUrls: [
      "https://brokercheck.finra.org/firm/summary/151717",
      "https://www.sofi.com/invest/account-protection-sipc/",
    ],
  },
  {
    firmName: "OPEN TO THE PUBLIC INVESTING, INC.",
    aliases: ["Public.com", "Public Investing", "Open to the Public"],
    sourceUrls: [
      "https://public.com/about-us",
    ],
    notes: "Plain 'Public' is intentionally excluded because it creates false positives in headlines.",
  },
  {
    firmName: "BETTERMENT SECURITIES",
    aliases: ["Betterment", "Betterment Securities"],
    sourceUrls: [
      "https://brokercheck.finra.org/firm/47788",
      "https://www.betterment.com/help/regulated-entity",
    ],
  },
  {
    firmName: "WEALTHFRONT BROKERAGE LLC",
    aliases: ["Wealthfront", "Wealthfront Brokerage"],
    sourceUrls: [
      "https://brokercheck.finra.org/firm/summary/153407",
      "https://www.wealthfront.com/",
    ],
  },
  {
    firmName: "ACORNS SECURITIES, LLC",
    aliases: ["Acorns", "Acorns Securities"],
    sourceUrls: [
      "https://brokercheck.finra.org/firm/summary/168172",
      "https://www.acorns.com/disclosures/",
    ],
  },
  {
    firmName: "M1 FINANCE LLC",
    aliases: ["M1 Finance", "M1 Invest"],
    sourceUrls: [
      "https://m1.com/legal/disclosures/",
    ],
    notes: "Plain 'M1' is intentionally excluded because it can refer to monetary aggregates or other non-firm contexts.",
  },
  {
    firmName: "TASTYTRADE, INC.",
    aliases: ["tastytrade", "tastyworks"],
    sourceUrls: [
      "https://brokercheck.finra.org/firm/summary/277027",
      "https://tastytrade.com/disclosures/",
    ],
  },
  {
    firmName: "ALPACA SECURITIES LLC",
    aliases: ["Alpaca", "Alpaca Securities", "Alpaca Clearing", "AlpacaDB"],
    sourceUrls: [
      "https://brokercheck.finra.org/firm/summary/288202",
      "https://alpaca.markets/disclosures",
    ],
    notes: "Alpaca Crypto is separate and not FINRA/SIPC.",
  },
  {
    firmName: "MOOMOO FINANCIAL INC.",
    aliases: ["Moomoo", "Moomoo Financial", "Moomoo app"],
    sourceUrls: [
      "https://brokercheck.finra.org/firm/summary/283078",
      "https://www.moomoo.com/us/support/topic4_483",
    ],
  },
  {
    firmName: "FUTU CLEARING INC",
    aliases: ["Futu", "Futu Clearing", "Futu Holdings"],
    sourceUrls: [
      "https://brokercheck.finra.org/firm/summary/298769",
      "https://www.futuclearing.com/disclosures",
    ],
  },
  {
    firmName: "COINBASE CAPITAL MARKETS CORP",
    aliases: ["Coinbase", "Coinbase Capital Markets"],
    sourceUrls: [
      "https://brokercheck.finra.org/firm/summary/10722",
      "https://www.coinbase.com/ccm",
    ],
    notes: "Broker-dealer affiliate for securities products; Coinbase crypto activity may involve other entities.",
  },
  {
    firmName: "INTERACTIVE BROKERS LLC",
    aliases: ["Interactive Brokers", "IBKR"],
    sourceUrls: [
      "https://brokercheck.finra.org/firm/summary/36418",
      "https://www.interactivebrokers.com/en/general/security-investor-protection.php",
    ],
  },
  {
    firmName: "CHARLES SCHWAB & CO., INC.",
    aliases: ["Schwab", "Charles Schwab", "Schwab Brokerage"],
    sourceUrls: [
      "https://files.brokercheck.finra.org/firm/firm_5393.pdf",
      "https://www.schwab.com/legal/account-protection",
    ],
  },
  {
    firmName: "FIDELITY BROKERAGE SERVICES LLC",
    aliases: ["Fidelity Investments", "Fidelity Brokerage", "Fidelity Brokerage Services"],
    sourceUrls: [
      "https://brokercheck.finra.org/Firm/Summary/7784",
      "https://www.fidelity.com/customer-service/Important-Legal-and-Regulatory-Disclosures",
    ],
    notes: "Plain 'Fidelity' is intentionally excluded because multiple FINRA member firms use that term.",
  },
] as const satisfies readonly FinraMemberFirmAffiliateAliasDefinition[];
