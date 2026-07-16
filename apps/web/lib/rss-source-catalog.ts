export type RssFeedDefinition = {
  label: string;
  feedUrl: string;
  refreshIntervalMinutes?: number;
};

/**
 * Existing feeds recovered from the deployed registry or earlier source
 * configuration. Keeping them here makes deployments reproducible without
 * changing their established feed keys.
 */
export const EXISTING_RSS_SOURCE_PROMOTIONS = {
  harvard_corp_gov_forum: {
    label: "Harvard Corporate Governance Forum",
    feedUrl: "https://corpgov.law.harvard.edu/feed/",
    refreshIntervalMinutes: 60,
  },
  cls_blue_sky_blog: {
    label: "CLS Blue Sky Blog",
    feedUrl: "https://clsbluesky.law.columbia.edu/feed/",
    refreshIntervalMinutes: 60,
  },
  the_corporate_counsel_net: {
    label: "The Corporate Counsel",
    feedUrl: "https://www.thecorporatecounsel.net/blog/feed/",
    refreshIntervalMinutes: 60,
  },
  rss_nytimes_com_services_xml_rss_nyt_economy_xml: {
    label: "NYT Economy",
    feedUrl: "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml",
    refreshIntervalMinutes: 60,
  },
  google_news_senate_banking_committee: {
    label: "Google News: Senate Banking Committee",
    feedUrl: "https://news.google.com/rss/search?q=%22Senate%20Banking%20Committee%22%20OR%20%22Senate%20Committee%20on%20Banking%2C%20Housing%2C%20and%20Urban%20Affairs%22%20OR%20site%3Abanking.senate.gov%20when%3A7d&hl=en-US&gl=US&ceid=US:en",
    refreshIntervalMinutes: 180,
  },
  google_news_senate_finance_committee: {
    label: "Google News: Senate Finance Committee",
    feedUrl: "https://news.google.com/rss/search?q=%22Senate%20Finance%20Committee%22%20OR%20%22Senate%20Committee%20on%20Finance%22%20OR%20site%3Afinance.senate.gov%20when%3A7d&hl=en-US&gl=US&ceid=US:en",
    refreshIntervalMinutes: 180,
  },
  google_news_senate_agriculture_committee: {
    label: "Google News: Senate Agriculture Committee",
    feedUrl: "https://news.google.com/rss/search?q=%22Senate%20Agriculture%20Committee%22%20OR%20%22Senate%20Committee%20on%20Agriculture%2C%20Nutrition%2C%20and%20Forestry%22%20OR%20site%3Aagriculture.senate.gov%20when%3A7d&hl=en-US&gl=US&ceid=US:en",
    refreshIntervalMinutes: 180,
  },
  google_news_senate_judiciary_committee: {
    label: "Google News: Senate Judiciary Committee",
    feedUrl: "https://news.google.com/rss/search?q=%22Senate%20Judiciary%20Committee%22%20OR%20%22Senate%20Committee%20on%20the%20Judiciary%22%20OR%20site%3Ajudiciary.senate.gov%20when%3A7d&hl=en-US&gl=US&ceid=US:en",
    refreshIntervalMinutes: 180,
  },
  google_news_senate_hsgac: {
    label: "Google News: Senate Homeland Security Committee",
    feedUrl: "https://news.google.com/rss/search?q=%22Senate%20Homeland%20Security%20and%20Governmental%20Affairs%20Committee%22%20OR%20%22Senate%20HSGAC%22%20OR%20site%3Ahsgac.senate.gov%20when%3A7d&hl=en-US&gl=US&ceid=US:en",
    refreshIntervalMinutes: 180,
  },
  google_news_senate_commerce_committee: {
    label: "Google News: Senate Commerce Committee",
    feedUrl: "https://news.google.com/rss/search?q=%22Senate%20Commerce%20Committee%22%20OR%20%22Senate%20Committee%20on%20Commerce%2C%20Science%2C%20and%20Transportation%22%20OR%20site%3Acommerce.senate.gov%20when%3A7d&hl=en-US&gl=US&ceid=US:en",
    refreshIntervalMinutes: 180,
  },
  american_banker: {
    label: "American Banker",
    feedUrl: "https://www.americanbanker.com/feed?rss=true",
    refreshIntervalMinutes: 60,
  },
  search_cnbc_com_rs_search_combinedcms_view_xml: {
    label: "CNBC",
    feedUrl: "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    refreshIntervalMinutes: 30,
  },
  rss_nytimes_com_services_xml_rss_nyt_business_xml: {
    label: "NYT Business",
    feedUrl: "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    refreshIntervalMinutes: 60,
  },
  rss_nytimes_com_services_xml_rss_nyt_dealbook_xml: {
    label: "NYT DealBook",
    feedUrl: "https://rss.nytimes.com/services/xml/rss/nyt/DealBook.xml",
    refreshIntervalMinutes: 60,
  },
  www_centralbanking_com_feeds_rss_category_central_banks_fina: {
    label: "Central Banking",
    feedUrl: "https://www.centralbanking.com/feeds/rss/category/central-banks/financial-stability",
    refreshIntervalMinutes: 60,
  },
} as const satisfies Record<string, RssFeedDefinition>;

export const EXISTING_RSS_SOURCE_PROMOTION_KEYS = Object.freeze(
  Object.keys(EXISTING_RSS_SOURCE_PROMOTIONS),
) as ReadonlyArray<keyof typeof EXISTING_RSS_SOURCE_PROMOTIONS>;
