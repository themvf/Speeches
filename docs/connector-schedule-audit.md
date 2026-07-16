# Connector Schedule Audit

Generated: 2026-06-28

Re-audit: 2026-06-29

## 2026-06-29 Addendum

The broad custom-document pipeline is mostly covered: scheduled extraction exists for the main official, trade-media, Reddit, CRS, Bloomberg, Substack, and NewsAPI paths, and `.github/workflows/connector-enrichment-6hour.yml` retries most scheduled custom-document source kinds every 6 hours.

2026-06-29 update: trade-association durable connectors were added for `ici_news_item`, `isda_news_item`, `mfa_news_item`, `fia_news_item`, `aba_news_item`, `bpi_news_item`, `icba_news_item`, and `lsta_news_item`. They run through `.github/workflows/connector-gap-6hour.yml` and retry through `.github/workflows/connector-enrichment-6hour.yml`.

Remaining automatic-ingestion/enrichment gaps:

1. `bloomberg_public_article` and `substack_public_article` are enriched inside their dedicated extraction workflows only when that run reports new or updated documents. They are not in the 6-hour connector enrichment backstop matrix, so missing, failed, or fallback enrichments may not retry automatically when a later extraction run has no changes.
2. `finra_comment_letter` is supported by the connector runner and the manual workflow, but it has no stable default source list and no schedule. It still requires a FINRA notice/rule URL override.
3. Streamlit/manual source kinds `sec_rule_release`, `sec_rule_comment`, `regulations_gov_rule`, and `regulations_gov_comment` are not in `SUPPORTED_CONNECTORS`, so they have no headless scheduled ingestion path and no source-kind entry in the scheduled enrichment matrix.
4. The AML-specific NewsAPI workflow is manual only. It writes normal `newsapi_article` records, so the NewsAPI enrichment backstop can eventually enrich those records, but the AML query itself is not continuously ingested.
5. The deployed app's RSS refresh stores feed rows in Neon and runs lightweight RSS item analysis, but most app RSS feeds do not become durable `custom_documents.json` records and therefore do not receive full document enrichment, review, vector indexing, or corpus-backed briefing treatment.
6. `knowledge-index-sync.yml` runs after SEC speech and policy extraction workflows only. It does not automatically run after `financial-news-daily`, `financial-news-enrich-scheduled`, `bloomberg-public-hourly`, `substack-public-3hour`, `connector-gap-6hour`, `connector-enrichment-6hour`, `securities-market-sources-daily`, or `crs-daily`, so newly ingested/enriched non-policy sources may lag in vector search until manual sync or another triggering workflow.

High-priority feed families that are currently RSS-analysis-only unless promoted by another connector:

- FINRA rule filings, dispute-resolution rule filings, news, and UPC advisories.
- Federal Reserve press releases, enforcement actions, and supervision/regulation letters.
- OCC news, bulletins, speeches, and congressional testimony.
- CFPB newsroom and FTC consumer-protection press releases.
- CFTC Federal Register proposed/final rule RSS feeds.
- WSJ Markets, WSJ Opinion, and MarketWatch Top Stories. The durable WSJ connector currently schedules only the default Dow Jones business RSS feed.
- Crypto feeds: CoinDesk, Cointelegraph, Decrypt, and The Block.
- Cyber feeds: CISA advisories and Krebs on Security.
- Law-firm and legal-analysis RSS feeds that are only present in the app RSS layer.

Recommended next changes:

1. Add `bloomberg_public_article` and `substack_public_article` to `.github/workflows/connector-enrichment-6hour.yml`.
2. Decide which RSS-only feeds should be promoted into durable custom-document connectors, starting with OCC, CFPB, FTC, Fed enforcement/SR letters, FINRA rule filings, CFTC Federal Register rules, and the remaining Dow Jones/MarketWatch feeds.
3. Add source-list configuration for URL-dependent comment sources: FINRA comment letters, SEC rule comments/releases, and Regulations.gov rules/comments.
4. Expand `knowledge-index-sync.yml` triggers or add a lower-frequency scheduled sync that covers all custom-document ingestion/enrichment workflows.
5. Keep AML as manual if it is only an ad hoc research query; otherwise add a scheduled AML NewsAPI ingest and either same-run enrichment or a clearly labeled backstop check.

This audit separates four different concepts that currently overlap in the codebase:

- Supported connector: accepted by `run_connector_extraction_pipeline.py`.
- On-demand workflow option: selectable in `.github/workflows/policy-extraction.yml` or a dedicated workflow.
- Scheduled workflow: has a GitHub Actions `schedule` cron or Vercel cron.
- App RSS source: refreshed by the deployed app, not persisted through `custom_documents.json` unless a separate document connector exists.

All GitHub Actions cron expressions are UTC. The Vercel cron in `apps/web/vercel.json` is also UTC.

## Scheduled Workflows

| Workflow | Purpose | Cron | Connectors or source family |
|---|---|---:|---|
| `.github/workflows/financial-news-daily.yml` | NewsAPI ingest plus enrichment | `0 8 * * *`, `17 11 * * *`, `17 12 * * *` | `newsapi_article` |
| `.github/workflows/financial-news-enrich-scheduled.yml` | Backstop enrichment for NewsAPI articles | `30 9 * * *` | `newsapi_article` |
| `.github/workflows/bloomberg-public-hourly.yml` | Bloomberg public feed extraction plus enrichment | `0 * * * *` | `bloomberg_public_latest` discovers `bloomberg_public_article` |
| `.github/workflows/substack-public-3hour.yml` | Substack public search/feed extraction plus enrichment | `0 */3 * * *` | `substack_public_article` |
| `.github/workflows/sec-speech-sync.yml` | Dedicated SEC speech sync | `0 3,11,19 * * *` | `sec_speech` |
| `.github/workflows/policy-extraction-scheduled.yml` | Core policy document extraction | `0 10 * * *`, `0 22 * * *` | `doj_usao_press_release`, `finra_awc`, `sec_enforcement_litigation`, `sec_speech` |
| `.github/workflows/securities-market-sources-daily.yml` | Securities market official sources | `30 12 * * *` | `finra_regulatory_notice`, `cftc_press_release`, `cftc_public_statement_remark`, `sec_press_release_rss`, `sec_administrative_proceeding`, `sec_trading_suspension`, `sec_federal_register`, `sec_pcaob_rulemaking`, `pcaob_update`, `msrb_press_release`, `sifma_news_item` |
| `.github/workflows/connector-gap-6hour.yml` | Remaining runnable connectors and durable trade/RSS/social documents | `0 */6 * * *` | `federal_reserve_speech_testimony`, `treasury_statement_remark`, `treasury_press_release`, `treasury_featured_story`, `sec_tm_faq`, `jdsupra_article`, `investmentnews_article`, `citywire_article`, `therecord_media_article`, `wired_article`, `tripwire_article`, `akamai_blog_article`, `ritholtz_article`, `ft_portfolios_market_commentary`, `liberty_street_economics_article`, `wealth_of_common_sense_article`, `wsj_dow_jones`, `reddit_post` |
| `.github/workflows/connector-enrichment-6hour.yml` | Backstop enrichment for scheduled custom-document connectors | `30 */6 * * *` | `only_missing_or_failed` enrichment for scheduled official, trade, RSS, social, CRS, SEC speech, DOJ, FINRA, CFTC, Treasury, Fed, SIFMA, and securities-market source kinds |
| `.github/workflows/crs-daily.yml` | CRS report extraction | `30 13 * * *` | `congress_crs_product` |
| `.github/workflows/trends-daily.yml` | Daily trend aggregation from enriched docs | `45 13 * * *` | Derived output, not a source connector |
| `.github/workflows/intelligence-evidence.yml` | GDELT evidence smoke tests | `0 9 * * *` | Live evidence verification, not a source connector |
| `.github/workflows/daily-health-check.yml` | Workflow/GCS/RSS health report | `0 9 * * *` | Health checks, not a source connector |
| `apps/web/vercel.json` | App RSS refresh endpoint | `*/10 * * * *` | `DEFAULT_RSS_FEEDS` into Neon RSS tables |

## On-Demand Only Workflows

| Workflow | Purpose | Scheduled? | Notes |
|---|---|---:|---|
| `.github/workflows/financial-news-ingest.yml` | Manual NewsAPI ingest | No | Scheduled equivalent exists in `financial-news-daily.yml`. |
| `.github/workflows/financial-news-enrich.yml` | Manual financial-news enrichment | No | Scheduled equivalent exists in `financial-news-enrich-scheduled.yml`. |
| `.github/workflows/policy-extraction.yml` | Manual connector extraction | No | Choice list is narrower than `SUPPORTED_CONNECTORS`. |
| `.github/workflows/knowledge-index-sync.yml` | Manual/vector sync after ingestion | No | Also triggers from workflow runs, but no cron. |
| `.github/workflows/aml-news-ingest.yml` | Manual AML-specific NewsAPI ingest | No | No continuous schedule. |
| `.github/workflows/python-tests.yml` | Tests on push/PR/manual | No cron | Not an ingestion source. |

## Connector Coverage

| Connector or source kind | Source | Source URL | On-demand option? | Scheduled? | Cron / workflow | Status |
|---|---|---|---:|---:|---|---|
| `newsapi_article` | NewsAPI financial news | Configured in `data/news_connector_settings.json`; domains include Reuters, WSJ, Bloomberg, FT, CNBC, AP, MarketWatch, CoinDesk | Yes, `financial-news-ingest.yml` | Yes | `financial-news-daily.yml`: `0 8 * * *`, `17 11 * * *`, `17 12 * * *`; enrichment: `30 9 * * *` | Covered |
| `aml_newsapi_article` | NewsAPI AML query | Query is embedded in `aml-news-ingest.yml` | Yes, dedicated manual workflow | No | None | Gap if AML-specific feed must be continuous |
| `bloomberg_public_latest` | Bloomberg public latest feed | Connector default is empty; scraper discovers latest public feed | Yes | Yes | `bloomberg-public-hourly.yml`: `0 * * * *` | Covered |
| `bloomberg_public_article` | Bloomberg public article extraction | Article URLs discovered by `bloomberg_public_latest` | Yes | Indirect | `bloomberg-public-hourly.yml`: `0 * * * *` | Covered through latest feed; direct article connector is manual |
| `bloomberg_latest_apify` | Legacy Bloomberg alias | No default URL | No | No | None | Legacy alias; deprecate or map intentionally |
| `bloomberg_apify_article` | Legacy Bloomberg alias | No default URL | No | No | None | Legacy alias; deprecate or map intentionally |
| `substack_public_article` | Substack public search/feed | `https://substack.com/api/v1/post/search` | Yes, dedicated scheduled workflow dispatch | Yes | `substack-public-3hour.yml`: `0 */3 * * *` | Covered |
| `sec_youtube_video` | SEC YouTube video transcripts | `https://www.youtube.com/user/SECViews` | Yes | Yes | `sec-youtube-videos-daily.yml`: `20 14 * * *` | Covered |
| `youtube_video` | Configurable YouTube channel transcripts | Requires a YouTube channel URL, handle, channel ID, or uploads RSS URL override | Yes, with `base_url` override | No | None | On-demand source for selected public channels |
| `jdsupra_article` | JD Supra legal/regulatory analysis | `https://www.jdsupra.com/` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered |
| `investmentnews_article` | InvestmentNews wealth-management news | `https://www.investmentnews.com/` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered |
| `citywire_article` | Citywire asset-management news | `https://citywire.com/us/news` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered |
| `therecord_media_article` | The Record cybersecurity news | `https://therecord.media/` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered |
| `wired_article` | WIRED security and technology coverage | `https://www.wired.com/category/security/` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered |
| `tripwire_article` | Tripwire State of Security | `https://www.tripwire.com/state-of-security` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered |
| `akamai_blog_article` | Akamai Blog | `https://www.akamai.com/blog` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered |
| `ritholtz_article` | The Big Picture / Ritholtz market commentary | `https://ritholtz.com/` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered |
| `ft_portfolios_market_commentary` | First Trust Portfolios market commentary | `https://www.ftportfolios.com/retail/blogs/marketcommentary/index.aspx` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered |
| `liberty_street_economics_article` | Liberty Street Economics | `https://libertystreeteconomics.newyorkfed.org/` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered |
| `wealth_of_common_sense_article` | A Wealth of Common Sense | `https://awealthofcommonsense.com/` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered |
| `wsj_dow_jones` | WSJ / Dow Jones durable RSS documents | `https://feeds.content.dowjones.io/public/rss/WSJcomUSBusinessNews` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered |
| `reddit_post` | Reddit keyword/social discovery | Configured by `data/news_connector_settings.json`; defaults in `reddit_scraper.py` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered; Reddit credentials recommended |
| `sec_speech` | SEC speeches and statements | `https://www.sec.gov/newsroom/speeches-statements` | Yes | Yes | `sec-speech-sync.yml`: `0 3,11,19 * * *`; also `policy-extraction-scheduled.yml`: `0 10 * * *`, `0 22 * * *` | Covered, possibly duplicated |
| `sec_enforcement_litigation` | SEC litigation releases | `https://www.sec.gov/enforcement-litigation/litigation-releases` | Yes | Yes | `policy-extraction-scheduled.yml`: `0 10 * * *`, `0 22 * * *` | Covered |
| `sec_tm_faq` | SEC trading markets FAQ | `https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered |
| `sec_press_release_rss` | SEC press releases | `https://www.sec.gov/news/pressreleases.rss` | Yes | Yes | `securities-market-sources-daily.yml`: `30 12 * * *` | Covered |
| `sec_administrative_proceeding` | SEC administrative proceedings | `https://www.sec.gov/enforcement-litigation/administrative-proceedings/rss` | Yes | Yes | `securities-market-sources-daily.yml`: `30 12 * * *` | Covered |
| `sec_trading_suspension` | SEC trading suspensions | `https://www.sec.gov/enforcement-litigation/trading-suspensions/rss` | Yes | Yes | `securities-market-sources-daily.yml`: `30 12 * * *` | Covered |
| `sec_federal_register` | SEC Federal Register materials | `https://www.federalregister.gov/articles/search.rss?conditions%5Bagency_ids%5D%5B%5D=466&order=newest` | Yes | Yes | `securities-market-sources-daily.yml`: `30 12 * * *` | Covered |
| `sec_pcaob_rulemaking` | SEC PCAOB rulemaking | `https://www.sec.gov/rules-regulations/public-company-accounting-oversight-board-rulemaking` | Yes | Yes | `securities-market-sources-daily.yml`: `30 12 * * *` | Covered |
| `doj_usao_press_release` | DOJ USAO press releases | `https://www.justice.gov/usao/pressreleases` | Yes | Yes | `policy-extraction-scheduled.yml`: `0 10 * * *`, `0 22 * * *` | Covered |
| `finra_awc` | FINRA disciplinary actions / AWC | `https://www.finra.org/rules-guidance/oversight-enforcement/finra-disciplinary-actions` | Yes | Yes | `policy-extraction-scheduled.yml`: `0 10 * * *`, `0 22 * * *` | Covered |
| `finra_regulatory_notice` | FINRA regulatory notices | `https://www.finra.org/rules-guidance/notices` | Yes | Yes | `securities-market-sources-daily.yml`: `30 12 * * *` | Covered |
| `finra_comment_letter` | FINRA comment letters for a notice URL | Requires a FINRA notice/rule URL override | Yes, with `base_url` override | No | None | Gap until a stable source URL list exists |
| `federal_reserve_speech_testimony` | Federal Reserve speeches/testimony | `https://www.federalreserve.gov/newsevents/speeches-testimony.htm` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered |
| `cftc_press_release` | CFTC press releases | `https://www.cftc.gov/PressRoom/PressReleases` | Yes | Yes | `securities-market-sources-daily.yml`: `30 12 * * *` | Covered |
| `cftc_public_statement_remark` | CFTC public statements and remarks | `https://www.cftc.gov/PressRoom/SpeechesTestimony/index.htm` | Yes | Yes | `securities-market-sources-daily.yml`: `30 12 * * *` | Covered |
| `treasury_featured_story` | Treasury featured stories | `https://home.treasury.gov/news/featured-stories` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered |
| `treasury_press_release` | Treasury press releases | `https://home.treasury.gov/news/press-releases` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered |
| `treasury_statement_remark` | Treasury statements and remarks | `https://home.treasury.gov/news/press-releases/statements-remarks` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *` | Covered |
| `sifma_news_item` | SIFMA news | `https://www.sifma.org/news` | Yes | Yes | `securities-market-sources-daily.yml`: `30 12 * * *` | Covered with `SIFMA_PROXY_URL` / `RESIDENTIAL_PROXY_URL` support |
| `ici_news_item` | Investment Company Institute news releases | `https://www.ici.org/news_%26_opinions/news-releases` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *`; enrichment: `connector-enrichment-6hour.yml` | Covered |
| `isda_news_item` | ISDA news | `https://www.isda.org/category/news/?subcategories=24` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *`; enrichment: `connector-enrichment-6hour.yml` | Covered |
| `mfa_news_item` | Managed Funds Association newsroom | `https://www.mfaalts.org/newsroom/` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *`; enrichment: `connector-enrichment-6hour.yml` | Covered |
| `fia_news_item` | FIA news | `https://www.fia.org/news` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *`; enrichment: `connector-enrichment-6hour.yml` | Covered |
| `aba_news_item` | American Bankers Association press releases | `https://www.aba.com/about-us/press-room/press-releases` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *`; enrichment: `connector-enrichment-6hour.yml` | Covered |
| `bpi_news_item` | Bank Policy Institute news | `https://bpi.com/news/` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *`; enrichment: `connector-enrichment-6hour.yml` | Covered |
| `icba_news_item` | ICBA news and articles | `https://www.icba.org/newsroom/news-and-articles` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *`; enrichment: `connector-enrichment-6hour.yml` | Covered |
| `lsta_news_item` | LSTA news and resources | `https://www.lsta.org/news-resources/` | Yes | Yes | `connector-gap-6hour.yml`: `0 */6 * * *`; enrichment: `connector-enrichment-6hour.yml` | Covered |
| `congress_crs_product` | Congressional Research Service reports | `https://www.congress.gov/crs-products` | Yes | Yes | `crs-daily.yml`: `30 13 * * *` | Covered |
| `pcaob_update` | PCAOB updates | `https://pcaobus.org/all-updates-and-news-releases` | Yes | Yes | `securities-market-sources-daily.yml`: `30 12 * * *` | Covered |
| `msrb_press_release` | MSRB press releases | `https://www.msrb.org/Press-Releases` | Yes | Yes | `securities-market-sources-daily.yml`: `30 12 * * *` | Covered |

## Scraper Modules Outside Full Schedule Coverage

These files exist as scraper/source concepts but are not all scheduled as durable document-ingest connectors.

| Scraper/module | Source family | Runner support | Schedule status | Gap |
|---|---|---:|---:|---|
| `trade_media_scraper.py` | `jdsupra_article`, `investmentnews_article`, `citywire_article`, `therecord_media_article`, `wired_article`, `tripwire_article`, `akamai_blog_article`, `ritholtz_article`, `ft_portfolios_market_commentary`, `liberty_street_economics_article`, `wealth_of_common_sense_article` | Yes | Yes | Runs through `connector-gap-6hour.yml` every 6 hours. |
| `wsj_rss_scraper.py` | `wsj_dow_jones` RSS article extraction | Yes | Yes | App RSS also refreshes every 10 minutes; durable document ingestion now runs through `connector-gap-6hour.yml` every 6 hours. |
| `reddit_scraper.py` | `reddit_post` social/news discovery | Yes | Yes | Runs through `connector-gap-6hour.yml` every 6 hours using configured Reddit settings; Reddit API credentials are recommended for cloud reliability. |
| `sec_rule_comments_scraper.py` | SEC rule comments | No, not in `SUPPORTED_CONNECTORS` | No | Rule-comment source kinds exist elsewhere, but this scraper is not scheduled through the connector runner. |
| `regulations_gov_manual_scraper.py` | Regulations.gov rule/comment documents | No, not in `SUPPORTED_CONNECTORS` | No | Manual source only unless wired into runner and schedule. |
| `sec_scraper_free.py`, `sec_speech_extractor.py`, `extract_all_speeches.py` | Legacy SEC speech extraction | Not the current scheduled runner path | Replaced by `sec_speech` scheduled runner | Keep as legacy/backfill unless needed. |
| `apify_bloomberg_scraper.py` | Legacy Bloomberg/Apify extraction | Legacy aliases exist in runner | No direct schedule | Prefer public Bloomberg connector or remove aliases. |

## App RSS Feed Sources

These are app-level RSS feeds defined in `apps/web/lib/server/rss-fetcher.ts`. They are refreshed by Vercel cron `*/10 * * * *` through `/api/intel/rss-refresh`, stored in Neon RSS tables, and are separate from `custom_documents.json` document ingestion.

The maintained catalog also restores 15 feeds that already existed in the deployed registry or earlier source configuration. Their established feed keys are retained so historical rows remain continuous. Similar coverage from different publishers is intentionally preserved: repeated independent reporting is treated as a corroboration and attention signal. Only replay of the same GUID within the same feed is collapsed.

| Existing source promoted to maintained defaults | Feed key | Default cadence |
|---|---|---:|
| Harvard Corporate Governance Forum | `harvard_corp_gov_forum` | 60m |
| CLS Blue Sky Blog | `cls_blue_sky_blog` | 60m |
| The Corporate Counsel | `the_corporate_counsel_net` | 60m |
| NYT Economy | `rss_nytimes_com_services_xml_rss_nyt_economy_xml` | 60m |
| Google News: Senate Banking Committee | `google_news_senate_banking_committee` | 180m |
| Google News: Senate Finance Committee | `google_news_senate_finance_committee` | 180m |
| Google News: Senate Agriculture Committee | `google_news_senate_agriculture_committee` | 180m |
| Google News: Senate Judiciary Committee | `google_news_senate_judiciary_committee` | 180m |
| Google News: Senate Homeland Security Committee | `google_news_senate_hsgac` | 180m |
| Google News: Senate Commerce Committee | `google_news_senate_commerce_committee` | 180m |
| American Banker | `american_banker` | 60m |
| CNBC | `search_cnbc_com_rs_search_combinedcms_view_xml` | 30m |
| NYT Business | `rss_nytimes_com_services_xml_rss_nyt_business_xml` | 60m |
| NYT DealBook | `rss_nytimes_com_services_xml_rss_nyt_dealbook_xml` | 60m |
| Central Banking | `www_centralbanking_com_feeds_rss_category_central_banks_fina` | 60m |

| Feed key | Label | Feed URL | Cron |
|---|---|---|---|
| `wsj_us_business` | WSJ US Business | `https://feeds.content.dowjones.io/public/rss/WSJcomUSBusinessNews` | `*/10 * * * *` |
| `wsj_markets` | WSJ Markets | `https://feeds.content.dowjones.io/public/rss/RSSMarketsMain` | `*/10 * * * *` |
| `wsj_opinion` | WSJ Opinion | `https://feeds.content.dowjones.io/public/rss/RSSOpinion` | `*/10 * * * *` |
| `mw_top_stories` | MarketWatch Top Stories | `https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines` | `*/10 * * * *` |
| `sec_press_releases` | SEC Press Releases | `https://www.sec.gov/news/pressreleases.rss` | `*/10 * * * *` |
| `sec_speeches_statements` | SEC Speeches and Statements | `https://www.sec.gov/news/speeches-statements.rss` | `*/10 * * * *` |
| `sec_litigation_releases` | SEC Litigation Releases | `https://www.sec.gov/enforcement-litigation/litigation-releases/rss` | `*/10 * * * *` |
| `sec_administrative_proceedings` | SEC Administrative Proceedings | `https://www.sec.gov/enforcement-litigation/administrative-proceedings/rss` | `*/10 * * * *` |
| `sec_trading_suspensions` | SEC Trading Suspensions | `https://www.sec.gov/enforcement-litigation/trading-suspensions/rss` | `*/10 * * * *` |
| `finra_notices` | FINRA Regulatory Notices | `http://feeds.finra.org/FINRANotices` | `*/10 * * * *` |
| `finra_rule_filings` | FINRA Rule Filings | `http://feeds.finra.org/FINRARuleFilings` | `*/10 * * * *` |
| `finra_dispute_resolution_rule_filings` | FINRA Dispute Resolution Rule Filings | `http://feeds.finra.org/DisputeResolutionRuleFilings` | `*/10 * * * *` |
| `finra_news` | FINRA News Releases and Speeches | `http://feeds.finra.org/FINRANews` | `*/10 * * * *` |
| `finra_upc_advisories` | FINRA UPC Advisories | `http://feeds.finra.org/FINRAUPCAdvisories` | `*/10 * * * *` |
| `cftc_general_press_releases` | CFTC General Press Releases | `https://www.cftc.gov/RSS/RSSGP/rssgp.xml` | `*/10 * * * *` |
| `cftc_enforcement_press_releases` | CFTC Enforcement Press Releases | `https://www.cftc.gov/RSS/RSSENF/rssenf.xml` | `*/10 * * * *` |
| `cftc_speeches_testimony` | CFTC Speeches and Testimony | `https://www.cftc.gov/RSS/RSSST/rssst.xml` | `*/10 * * * *` |
| `cftc_federal_register_proposed_rules` | CFTC Federal Register Proposed Rules | `http://comments.cftc.gov/handlers/RSSHandler.ashx?type=Releases&category=Proposed%20Rule` | `*/10 * * * *` |
| `cftc_federal_register_final_rules` | CFTC Federal Register Final Rules | `http://comments.cftc.gov/handlers/RSSHandler.ashx?type=Releases&category=Final%20Rule` | `*/10 * * * *` |
| `fed_all_press_releases` | Federal Reserve All Press Releases | `https://www.federalreserve.gov/feeds/press_all.xml` | `*/10 * * * *` |
| `fed_banking_consumer_regulatory_policy` | Federal Reserve Banking and Consumer Regulatory Policy | `https://www.federalreserve.gov/feeds/press_bcreg.xml` | `*/10 * * * *` |
| `fed_enforcement_actions` | Federal Reserve Enforcement Actions | `https://www.federalreserve.gov/feeds/press_enforcement.xml` | `*/10 * * * *` |
| `fed_supervision_regulation_letters` | Federal Reserve Supervision and Regulation Letters | `https://www.federalreserve.gov/feeds/bankinginfo-rss.xml` | `*/10 * * * *` |
| `occ_news_releases` | OCC News Releases | `https://www.occ.gov/rss/occ_news.xml` | `*/10 * * * *` |
| `occ_bulletins` | OCC Bulletins | `https://www.occ.gov/rss/occ_bulletins.xml` | `*/10 * * * *` |
| `occ_speeches` | OCC Speeches | `https://www.occ.gov/rss/occ-speeches.xml` | `*/10 * * * *` |
| `occ_congressional_testimony` | OCC Congressional Testimony | `https://www.occ.gov/rss/occ-congressional-testimony.xml` | `*/10 * * * *` |
| `cfpb_newsroom` | CFPB Newsroom | `https://www.consumerfinance.gov/about-us/newsroom/feed/` | `*/10 * * * *` |
| `ftc_consumer_protection_press_releases` | FTC Consumer Protection Press Releases | `https://www.ftc.gov/feeds/press-release-consumer-protection.xml` | `*/10 * * * *` |
| `coindesk` | CoinDesk | `https://www.coindesk.com/arc/outboundfeeds/rss/` | `*/10 * * * *` |
| `cointelegraph` | Cointelegraph | `https://cointelegraph.com/rss` | `*/10 * * * *` |
| `decrypt` | Decrypt | `https://decrypt.co/feed` | `*/10 * * * *` |
| `the_block` | The Block | `https://www.theblock.co/rss.xml` | `*/10 * * * *` |
| `cisa_cybersecurity_advisories` | CISA Cybersecurity Advisories | `https://www.cisa.gov/cybersecurity-advisories/all.xml` | `*/10 * * * *` |
| `krebs_on_security` | Krebs on Security | `https://krebsonsecurity.com/feed/` | `*/10 * * * *` |
| `gibson_dunn_sec_sentinel` | Gibson Dunn SEC Sentinel | `https://secsentinel.gibsondunn.com/feed/` | `*/10 * * * *` |
| `gibson_dunn_securities_regulation_monitor` | Gibson Dunn Securities Regulation and Corporate Governance Monitor | `https://themonitor.gibsondunn.com/feed/` | `*/10 * * * *` |
| `cleary_enforcement_watch` | Cleary Enforcement Watch | `https://www.clearyenforcementwatch.com/feed/` | `*/10 * * * *` |
| `cooley_pubco` | Cooley PubCo | `https://cooleypubco.com/feed/` | `*/10 * * * *` |
| `cooley_cyber_data_privacy` | Cooley Cyber/Data/Privacy | `https://cdp.cooley.com/feed/` | `*/10 * * * *` |
| `cooley_governance_beat` | Cooley Governance Beat | `https://governancebeat.cooley.com/feed/` | `*/10 * * * *` |
| `latham_global_financial_regulatory_blog` | Latham Global Financial Regulatory Blog | `https://www.globalfinregblog.com/feed/` | `*/10 * * * *` |
| `latham_london` | Latham.London | `https://www.latham.london/feed/` | `*/10 * * * *` |
| `covington_inside_privacy` | Covington Inside Privacy | `https://www.insideprivacy.com/feed/` | `*/10 * * * *` |
| `covington_global_policy_watch` | Covington Global Policy Watch | `https://www.globalpolicywatch.com/feed/` | `*/10 * * * *` |
| `covington_inside_government_contracts` | Covington Inside Government Contracts | `https://www.insidegovernmentcontracts.com/feed/` | `*/10 * * * *` |
| `ballard_spahr_consumer_finance_monitor` | Ballard Spahr Consumer Finance Monitor | `https://www.consumerfinancemonitor.com/feed/` | `*/10 * * * *` |
| `kelley_drye_ad_law_access` | Kelley Drye Ad Law Access | `https://www.kelleydrye.com/viewpoints/blogs/ad-law-access/rss` | `*/10 * * * *` |
| `norton_rose_fulbright_data_protection_report` | Norton Rose Fulbright Data Protection Report | `https://www.dataprotectionreport.com/feed/` | `*/10 * * * *` |
| `squire_patton_boggs_privacy_world` | Squire Patton Boggs Privacy World | `https://www.privacyworld.blog/feed/` | `*/10 * * * *` |
| `bradley_financial_services_perspectives` | Bradley Financial Services Perspectives | `https://www.financialservicesperspectives.com/feed/` | `*/10 * * * *` |
| `bradley_eye_on_enforcement` | Bradley Eye on Enforcement | `https://www.eyeonenforcement.com/feed/` | `*/10 * * * *` |

## Gaps To Close

Priority 1, supported by runner but not safely schedulable without source inputs:

- `finra_comment_letter` requires a FINRA notice/rule URL override.

Priority 2, source concepts exist but still require curated URL lists or a crawler design:

- `sec_rule_comment`
- `regulations_gov_comment`

Priority 3, cleanup or document as intentionally legacy:

- `bloomberg_latest_apify`
- `bloomberg_apify_article`
- `sec_scraper_free.py`
- `sec_speech_extractor.py`
- `extract_all_speeches.py`

## Recommended Scheduling Changes

1. Keep `.github/workflows/connector-enrichment-6hour.yml` enabled so newly ingested or failed custom-document enrichments are retried every 6 hours.
2. Add a source-list config for URL-dependent comment connectors:
   - FINRA comment letters need one or more source notice URLs.
   - SEC rule comments and Regulations.gov comments need one or more rule/docket URLs.
3. Add connector freshness metrics for every scheduled source kind, not just NewsAPI, so Admin can show `last workflow success`, `newest document`, `documents in feed`, and `stale` per connector.
4. Consider whether OCC, CFPB, FTC, law-firm blogs, crypto feeds, and cyber feeds should remain app RSS rows only or become durable `custom_documents.json` document connectors.
