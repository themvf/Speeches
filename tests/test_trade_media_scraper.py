import unittest
from unittest.mock import patch

from trade_media_scraper import TradeMediaScraper, _looks_like_access_challenge


class _FakeResponse:
    def __init__(self, text="", url="", headers=None):
        self.text = text
        self.url = url or "https://example.com/article"
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}

    def raise_for_status(self):
        return None


class TradeMediaScraperTests(unittest.TestCase):
    def test_build_google_news_query_uses_domain_and_optional_terms(self):
        scraper = TradeMediaScraper(min_delay_seconds=0)

        query = scraper._build_google_news_query(
            "investmentnews_article",
            "https://www.investmentnews.com/",
            "SEC enforcement",
        )

        self.assertEqual(query, "site:investmentnews.com SEC enforcement")

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
