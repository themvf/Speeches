import unittest
from unittest.mock import Mock, patch

from congress_crs_products_scraper import (
    CRS_PRODUCTS_SEARCH_URL,
    CRS_PRODUCTS_BROWSE_URL,
    EVERY_CRS_INDEX_URL,
    CongressCRSProductsScraper,
    _infer_doc_type,
    _is_crs_detail_url,
    _normalize_listing_url,
)


class _FakeResponse:
    def __init__(self, text: str, url: str, status_code: int = 200):
        self.text = text
        self.url = url
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class CongressCRSProductsScraperTests(unittest.TestCase):
    def test_listing_url_normalizes_browse_page_to_quick_search(self):
        normalized = _normalize_listing_url(CRS_PRODUCTS_BROWSE_URL)
        self.assertIn("/index.php/quick-search/crs-products", normalized)
        self.assertIn("pageSize=100", normalized)

    def test_detail_url_detection(self):
        self.assertTrue(_is_crs_detail_url("https://www.congress.gov/crs-product/R48860"))
        self.assertFalse(_is_crs_detail_url("https://www.congress.gov/crs-products"))
        self.assertFalse(_is_crs_detail_url("https://www.example.com/crs-product/R48860"))

    def test_doc_type_inference_from_product_number_prefix(self):
        self.assertEqual(_infer_doc_type("R48860"), "Report")
        self.assertEqual(_infer_doc_type("IF12852"), "In Focus")
        self.assertEqual(_infer_doc_type("IN12458"), "Insight")
        self.assertEqual(_infer_doc_type("LSB11328"), "Legal Sidebar")

    @patch.dict("os.environ", {"CRS_PROXY_URL": "http://proxy.example:8000"})
    def test_fetch_html_uses_configured_proxy(self):
        scraper = CongressCRSProductsScraper(min_delay_seconds=0)
        response = _FakeResponse("<html><body>OK</body></html>", "https://www.congress.gov/crs-products")
        scraper.session.get = Mock(return_value=response)

        scraper._fetch_html("https://www.congress.gov/crs-products")

        self.assertEqual(scraper.session.get.call_args.kwargs["proxy"], "http://proxy.example:8000")

    @patch.dict(
        "os.environ",
        {"RESIDENTIAL_PROXY_URL": "http://residential.example:8000"},
        clear=True,
    )
    def test_fetch_html_uses_residential_proxy_fallback(self):
        scraper = CongressCRSProductsScraper(min_delay_seconds=0)
        response = _FakeResponse("<html><body>OK</body></html>", "https://www.congress.gov/crs-products")
        scraper.session.get = Mock(return_value=response)

        scraper._fetch_html("https://www.congress.gov/crs-products")

        self.assertEqual(scraper.session.get.call_args.kwargs["proxy"], "http://residential.example:8000")

    def test_fetch_html_rejects_cloudflare_challenge(self):
        scraper = CongressCRSProductsScraper(min_delay_seconds=0)
        response = _FakeResponse(
            "<html><title>Just a moment...</title>Enable JavaScript and cookies to continue</html>",
            "https://www.congress.gov/crs-products",
        )
        scraper.session.get = Mock(return_value=response)

        with self.assertRaisesRegex(RuntimeError, "Cloudflare challenge"):
            scraper._fetch_html("https://www.congress.gov/crs-products")

    @patch.object(CongressCRSProductsScraper, "_fetch_html")
    def test_discovery_rejects_empty_listing_page(self, mock_fetch_html):
        mock_fetch_html.return_value = _FakeResponse(
            "<html><title>Congress.gov</title><body>No matching results</body></html>",
            "https://www.congress.gov/index.php/quick-search/crs-products?pageSize=100",
        )
        scraper = CongressCRSProductsScraper(min_delay_seconds=0)

        with self.assertRaisesRegex(RuntimeError, "listing returned no products"):
            scraper.discover_documents(base_url=CRS_PRODUCTS_SEARCH_URL, max_pages=1)

    @patch.object(CongressCRSProductsScraper, "_fetch_html")
    def test_discover_documents_parses_listing_results(self, mock_fetch_html):
        html = """
        <html>
          <body>
            <main>
              <section>
                <div class="search-row">
                  <a href="/crs-product/R48860">FY2026 Defense Budget: Funding for Selected Weapon Systems</a>
                  <div>CRS Product Type: Reports</div>
                  <div>CRS Product Number: R48860</div>
                  <div>Publication Date: 02/20/2026</div>
                  <div>Author: Gettinger, Daniel M.</div>
                  <div>Topics: Defense &amp; Intelligence; Appropriations</div>
                </div>
                <div class="search-row">
                  <a href="/crs-product/IF12852">Federal Contract Set-Asides for Small Businesses</a>
                  <div>CRS Product Type: In Focus</div>
                  <div>CRS Product Number: IF12852</div>
                  <div>Publication Date: 02/20/2026</div>
                  <div>Author: Dilger, Robert Jay</div>
                  <div>Topics: Commerce &amp; Small Business</div>
                </div>
                <a rel="next" href="/index.php/quick-search/crs-products?page=2&amp;pageSize=100">Next page</a>
              </section>
            </main>
          </body>
        </html>
        """
        mock_fetch_html.return_value = _FakeResponse(
            html,
            "https://www.congress.gov/index.php/quick-search/crs-products?pageSize=100",
            200,
        )

        scraper = CongressCRSProductsScraper(min_delay_seconds=0)
        docs = scraper.discover_documents(base_url=CRS_PRODUCTS_SEARCH_URL, max_pages=1)

        self.assertEqual(len(docs), 2)
        self.assertTrue(mock_fetch_html.call_args[0][0].startswith("https://www.congress.gov/index.php/quick-search/crs-products"))
        self.assertEqual(docs[0]["product_number"], "R48860")
        self.assertEqual(docs[0]["doc_type"], "Reports")
        self.assertEqual(docs[0]["authors"], "Gettinger, Daniel M.")
        self.assertEqual(docs[0]["topics"], ["Defense & Intelligence", "Appropriations"])
        self.assertEqual(scraper.last_discovery_debug["pages"][0]["returned_items"], 2)
        self.assertFalse(scraper.last_discovery_debug["proxy_configured"])
        self.assertIn("page=2", scraper.last_discovery_debug["pages"][0]["next_page_url"])

    @patch.object(CongressCRSProductsScraper, "_fetch_html")
    def test_default_discovery_uses_static_mirror_and_official_urls(self, mock_fetch_html):
        html = """
        <html><body>
          <div class="crs-report">
            <p class="report-title"><a href="/reports/R48984.html">Advanced Air Mobility</a></p>
            <p class="report-metadata">April 1, 2026 - June 12, 2026 <code>R48984</code></p>
          </div>
        </body></html>
        """
        mock_fetch_html.return_value = _FakeResponse(html, EVERY_CRS_INDEX_URL)
        scraper = CongressCRSProductsScraper(min_delay_seconds=0)

        docs = scraper.discover_documents(max_pages=1)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["url"], "https://www.congress.gov/crs-product/R48984")
        self.assertEqual(docs[0]["mirror_url"], "https://www.everycrsreport.com/reports/R48984.html")
        self.assertEqual(docs[0]["date"], "June 12, 2026")
        self.assertEqual(scraper.last_discovery_debug["discovery_mode"], "everycrsreport_static_index")

    @patch.object(CongressCRSProductsScraper, "_fetch_html")
    def test_mirror_extraction_selects_latest_version_metadata(self, mock_fetch_html):
        html = """
        <html><body><main><h1>Updated CRS Report</h1><article>
        <p>This is the current full report text with enough policy analysis to represent an extracted document.</p>
        </article></main></body></html>
        """
        metadata = """{
          "id": "R40000",
          "topics": ["Economic Policy"],
          "versions": [
            {"date": "2026-06-15", "title": "Updated CRS Report", "formats": [{"format": "PDF", "url": "https://example.test/latest.pdf"}]},
            {"date": "2020-01-01", "title": "Old CRS Report", "formats": [{"format": "PDF", "url": "https://example.test/old.pdf"}]}
          ]
        }"""
        mock_fetch_html.side_effect = [
            _FakeResponse(html, "https://www.everycrsreport.com/reports/R40000.html"),
            _FakeResponse(metadata, "https://www.everycrsreport.com/reports/R40000.json"),
        ]
        scraper = CongressCRSProductsScraper(min_delay_seconds=0)

        result = scraper.extract_document(
            "https://www.everycrsreport.com/reports/R40000.html",
            canonical_url_override="https://www.congress.gov/crs-product/R40000",
        )["data"]

        self.assertEqual(result["url"], "https://www.congress.gov/crs-product/R40000")
        self.assertEqual(result["date"], "June 15, 2026")
        self.assertEqual(result["pdf_url"], "https://example.test/latest.pdf")
        self.assertEqual(result["topics"], ["Economic Policy"])

    @patch.object(CongressCRSProductsScraper, "_fetch_html")
    def test_extract_document_parses_detail_page_and_pdf_link(self, mock_fetch_html):
        html = """
        <html>
          <body>
            <main>
              <nav>Breadcrumb</nav>
              <h1>FY2026 Defense Budget: Funding for Selected Weapon Systems</h1>
              <section>
                <div>CRS Product Type: Reports</div>
                <div>CRS Product Number: R48860</div>
                <div>Publication Date: 02/20/2026</div>
                <div>Author: Gettinger, Daniel M.</div>
                <div>Topics: Defense &amp; Intelligence; Appropriations</div>
                <a href="/119/crs-product/R48860/pdf">Download PDF (891KB)</a>
              </section>
              <article>
                <p>FY2026 Defense Budget: Funding for Selected Weapon Systems</p>
                <p>February 20, 2026 (R48860)</p>
                <p>This report examines selected weapon-system funding levels.</p>
                <p>It provides background and congressional context.</p>
              </article>
              <footer>Site Content</footer>
            </main>
          </body>
        </html>
        """
        mock_fetch_html.return_value = _FakeResponse(
            html,
            "https://www.congress.gov/crs-product/R48860",
            200,
        )

        scraper = CongressCRSProductsScraper(min_delay_seconds=0)
        result = scraper.extract_document("https://www.congress.gov/crs-product/R48860")
        data = result["data"]

        self.assertTrue(result["success"])
        self.assertEqual(data["title"], "FY2026 Defense Budget: Funding for Selected Weapon Systems")
        self.assertEqual(data["date"], "February 20, 2026")
        self.assertEqual(data["doc_type"], "Reports")
        self.assertEqual(data["product_number"], "R48860")
        self.assertEqual(data["authors"], "Gettinger, Daniel M.")
        self.assertEqual(data["topics"], ["Defense & Intelligence", "Appropriations"])
        self.assertEqual(data["pdf_url"], "https://www.congress.gov/119/crs-product/R48860/pdf")
        self.assertIn("selected weapon-system funding levels", data["full_text"])


if __name__ == "__main__":
    unittest.main()
