import unittest

from newsapi_financial_scraper import NewsAPIFinancialScraper


class NewsAPIFinancialScraperDiscoveryTests(unittest.TestCase):
    def test_discovers_broader_sources_when_domain_allowlist_is_sparse(self):
        scraper = NewsAPIFinancialScraper(api_key="test-key")
        calls = []

        def fake_fetch_json(endpoint, params):
            calls.append(dict(params))
            domain = str(params.get("domains", "") or "")
            if domain == "cnbc.com,coindesk.com":
                return {
                    "status": "ok",
                    "totalResults": 1,
                    "articles": [
                        {
                            "url": "https://www.cnbc.com/2026/04/19/existing-sanctions.html",
                            "title": "Existing sanctions article",
                            "publishedAt": "2026-04-19T12:00:00Z",
                            "source": {"name": "CNBC", "id": "cnbc"},
                        }
                    ],
                }
            if domain == "cnbc.com":
                return {"status": "ok", "totalResults": 0, "articles": []}
            if domain == "coindesk.com":
                return {"status": "ok", "totalResults": 0, "articles": []}
            return {
                "status": "ok",
                "totalResults": 2,
                "articles": [
                    {
                        "url": "https://www.cnbc.com/2026/04/19/existing-sanctions.html",
                        "title": "Existing sanctions article",
                        "publishedAt": "2026-04-19T12:00:00Z",
                        "source": {"name": "CNBC", "id": "cnbc"},
                    },
                    {
                        "url": "https://example.com/2026/04/22/fincen-aml-update.html",
                        "title": "FinCEN AML update",
                        "publishedAt": "2026-04-22T12:00:00Z",
                        "source": {"name": "Example News", "id": "example"},
                    },
                ],
            }

        scraper._fetch_json = fake_fetch_json  # type: ignore[method-assign]

        docs = scraper.discover_documents(
            query="AML OR FinCEN",
            domains="cnbc.com,coindesk.com",
            max_pages=1,
            page_size=10,
            target_count=5,
        )

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["title"], "FinCEN AML update")
        self.assertTrue(scraper.last_discovery_debug["fallback_no_domains_used"])
        self.assertTrue(any("domains" not in call for call in calls))


if __name__ == "__main__":
    unittest.main()
