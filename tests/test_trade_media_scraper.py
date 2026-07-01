import unittest
from unittest.mock import patch

from trade_media_scraper import (
    TRADE_MEDIA_SOURCES,
    TradeMediaScraper,
    _looks_like_access_challenge,
    _passes_source_url_filters,
    _url_key,
)


class _FakeResponse:
    def __init__(self, text="", url="", headers=None):
        self.text = text
        self.url = url or "https://example.com/article"
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}

    def raise_for_status(self):
        return None


class TradeMediaScraperTests(unittest.TestCase):
    def test_requested_sources_are_registered_with_defaults(self):
        for source_key in [
            "therecord_media_article",
            "wired_article",
            "tripwire_article",
            "akamai_blog_article",
            "ritholtz_article",
            "ft_portfolios_market_commentary",
            "liberty_street_economics_article",
            "wealth_of_common_sense_article",
        ]:
            cfg = TRADE_MEDIA_SOURCES[source_key]
            self.assertTrue(cfg["label"])
            self.assertTrue(cfg["default_url"])
            self.assertTrue(cfg["search_domain"])

    def test_url_key_preserves_meaningful_query_parameters(self):
        key = _url_key("https://www.ftportfolios.com/retail/blogs/marketcommentary/index.aspx?id=123&utm_source=x")

        self.assertEqual(
            key,
            "https://www.ftportfolios.com/retail/blogs/marketcommentary/index.aspx?id=123",
        )

    def test_requested_source_url_filters_reject_navigation_pages(self):
        self.assertFalse(
            _passes_source_url_filters(
                "https://therecord.media/news/cybercrime",
                TRADE_MEDIA_SOURCES["therecord_media_article"],
            )
        )
        self.assertFalse(
            _passes_source_url_filters(
                "https://www.ftportfolios.com/retail/blogs/marketcommentary/index.aspx",
                TRADE_MEDIA_SOURCES["ft_portfolios_market_commentary"],
            )
        )
        self.assertFalse(
            _passes_source_url_filters(
                "https://www.wired.com/story/ulta-promo-codes-july-2026/",
                TRADE_MEDIA_SOURCES["wired_article"],
            )
        )
        self.assertFalse(
            _passes_source_url_filters(
                "https://www.wired.com/story/nike-promo-codes-and-discounts-july-2026/",
                TRADE_MEDIA_SOURCES["wired_article"],
            )
        )
        self.assertTrue(
            _passes_source_url_filters(
                "https://www.ftportfolios.com/Commentary/MarketCommentary/2026/6/30/just-a-smidge",
                TRADE_MEDIA_SOURCES["ft_portfolios_market_commentary"],
            )
        )
        self.assertTrue(
            _passes_source_url_filters(
                "https://www.wired.com/story/cybersecurity-regulators-ransomware-banks/",
                TRADE_MEDIA_SOURCES["wired_article"],
            )
        )

    def test_wired_feed_discovery_excludes_coupon_articles(self):
        scraper = TradeMediaScraper(min_delay_seconds=0)
        rss_text = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Ulta Promo Codes: Up to 50% Off in July 2026</title>
      <link>https://www.wired.com/story/ulta-promo-codes-july-2026/</link>
      <pubDate>Wed, 01 Jul 2026 11:00:00 GMT</pubDate>
      <description>Coupon page.</description>
    </item>
    <item>
      <title>Ransomware Operators Target Financial Firms</title>
      <link>https://www.wired.com/story/ransomware-financial-firms-2026/</link>
      <pubDate>Wed, 01 Jul 2026 12:00:00 GMT</pubDate>
      <description>Security article.</description>
    </item>
  </channel>
</rss>
"""

        with patch.object(scraper, "_fetch", return_value=_FakeResponse(text=rss_text)):
            docs = scraper._discover_from_feed(
                feed_url="https://www.wired.com/feed/category/security/latest/rss",
                source_key="wired_article",
                source_label="WIRED",
                source_url="https://www.wired.com/category/security/",
            )

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["title"], "Ransomware Operators Target Financial Firms")
        self.assertEqual(docs[0]["url"], "https://www.wired.com/story/ransomware-financial-firms-2026/")

    def test_wired_discovery_uses_security_feed_not_sitewide_feed(self):
        scraper = TradeMediaScraper(min_delay_seconds=0)
        listing_html = """
<html><head>
  <link rel="alternate" type="application/rss+xml" href="https://www.wired.com/feed/rss" />
</head><body></body></html>
"""
        rss_text = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Security Story</title>
      <link>https://www.wired.com/story/security-story/</link>
      <pubDate>Wed, 01 Jul 2026 12:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""
        fetched = []

        def fake_fetch(url, timeout=45):
            fetched.append(url)
            if url.endswith("/category/security/"):
                return _FakeResponse(text=listing_html, url=url)
            return _FakeResponse(text=rss_text, url=url)

        with patch.object(scraper, "_fetch", side_effect=fake_fetch):
            docs = scraper.discover_documents("wired_article", max_pages=1, include_rss=True)

        self.assertEqual(len(docs), 1)
        self.assertIn("https://www.wired.com/feed/category/security/latest/rss", fetched)
        self.assertNotIn("https://www.wired.com/feed/rss", fetched)

    def test_build_google_news_query_uses_domain_and_optional_terms(self):
        scraper = TradeMediaScraper(min_delay_seconds=0)

        query = scraper._build_google_news_query(
            "investmentnews_article",
            "https://www.investmentnews.com/",
            "SEC enforcement",
        )

        self.assertEqual(query, "site:investmentnews.com SEC enforcement")

    def test_build_google_news_query_uses_path_scoped_site_query_when_configured(self):
        scraper = TradeMediaScraper(min_delay_seconds=0)

        query = scraper._build_google_news_query(
            "akamai_blog_article",
            "https://www.akamai.com/blog",
            "API security",
        )

        self.assertEqual(query, "site:akamai.com/blog API security")

    def test_google_news_search_fallback_decodes_real_urls(self):
        scraper = TradeMediaScraper(min_delay_seconds=0)
        rss_text = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>SEC ordered $17.9bn in enforcement-related relief in FY2025 - Citywire</title>
      <link>https://news.google.com/rss/articles/example-1</link>
      <pubDate>Tue, 07 Apr 2026 18:41:44 GMT</pubDate>
      <description><![CDATA[<a href="https://news.google.com/rss/articles/example-1">SEC ordered $17.9bn in enforcement-related relief in FY2025</a><font color="#6f6f6f">Citywire</font>]]></description>
      <source>Citywire</source>
    </item>
  </channel>
</rss>
"""

        with patch.object(scraper, "_fetch", return_value=_FakeResponse(text=rss_text)):
            with patch.object(
                scraper,
                "_decode_google_news_url",
                return_value="https://citywire.com/ria/news/sec-ordered-17-9bn-in-enforcement-related-relief-in-fy2025/a2454863",
            ):
                docs = scraper._discover_from_google_news_search(
                    source_key="citywire_article",
                    source_label="Citywire",
                    source_url="https://citywire.com/us/news",
                    search_query="SEC",
                    max_results=10,
                )

        self.assertEqual(len(docs), 1)
        self.assertEqual(
            docs[0]["url"],
            "https://citywire.com/ria/news/sec-ordered-17-9bn-in-enforcement-related-relief-in-fy2025/a2454863",
        )
        self.assertEqual(docs[0]["title"], "SEC ordered $17.9bn in enforcement-related relief in FY2025")
        self.assertEqual(docs[0]["discovery_source"], "google_news_search")
        self.assertEqual(docs[0]["search_query"], "site:citywire.com SEC")

    def test_google_news_search_keeps_google_link_when_decode_is_unavailable(self):
        scraper = TradeMediaScraper(min_delay_seconds=0)
        rss_text = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>SEC update - InvestmentNews</title>
      <link>https://news.google.com/rss/articles/example-2</link>
      <pubDate>Tue, 07 Apr 2026 18:41:44 GMT</pubDate>
      <description><![CDATA[<a href="https://news.google.com/rss/articles/example-2">SEC update</a><font color="#6f6f6f">InvestmentNews</font>]]></description>
      <source>InvestmentNews</source>
    </item>
  </channel>
</rss>
"""

        with patch.object(scraper, "_fetch", return_value=_FakeResponse(text=rss_text)):
            with patch.object(scraper, "_decode_google_news_url", return_value=""):
                docs = scraper._discover_from_google_news_search(
                    source_key="investmentnews_article",
                    source_label="InvestmentNews",
                    source_url="https://www.investmentnews.com/",
                    search_query="SEC",
                    max_results=10,
                )

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["url"], "https://news.google.com/rss/articles/example-2")
        self.assertEqual(docs[0]["source_url"], "")

    def test_extract_document_uses_snippet_when_access_challenge_detected(self):
        scraper = TradeMediaScraper(min_delay_seconds=0)
        blocked_html = """
<html><head><title>Blocked</title></head>
<body>Request unsuccessful. Incapsula incident ID: 123</body></html>
"""

        with patch.object(
            scraper,
            "_fetch",
            return_value=_FakeResponse(
                text=blocked_html,
                url="https://citywire.com/us/news/example/a123",
            ),
        ):
            result = scraper.extract_document(
                "https://citywire.com/us/news/example/a123",
                fallback_title="Blocked article headline",
                fallback_date="April 01, 2026",
                fallback_description="Search discovery found this result, but the site challenged the request.",
                fallback_source_name="Citywire",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["source_format"], "snippet")
        self.assertIn("blocked or returned a protection page", result["data"]["full_text"])
        self.assertGreaterEqual(result["data"]["word_count"], 30)

    def test_extract_document_returns_snippet_when_google_news_url_cannot_be_decoded(self):
        scraper = TradeMediaScraper(min_delay_seconds=0)

        with patch.object(scraper, "_decode_google_news_url", return_value=""):
            result = scraper.extract_document(
                "https://news.google.com/rss/articles/example-3",
                fallback_title="Google News fallback headline",
                fallback_date="April 01, 2026",
                fallback_description="Discovery captured this article via Google News when direct source discovery was unavailable.",
                fallback_source_name="InvestmentNews",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["source_format"], "snippet")
        self.assertEqual(result["data"]["url"], "https://news.google.com/rss/articles/example-3")
        self.assertGreaterEqual(result["data"]["word_count"], 30)

    def test_access_challenge_detector_handles_cloudflare_and_incapsula_markers(self):
        self.assertTrue(_looks_like_access_challenge("Attention Required! | Cloudflare"))
        self.assertTrue(_looks_like_access_challenge("Request unsuccessful. Incapsula incident ID: 123"))
        self.assertFalse(_looks_like_access_challenge("<html><body><article><p>Normal content.</p></article></body></html>"))


if __name__ == "__main__":
    unittest.main()
