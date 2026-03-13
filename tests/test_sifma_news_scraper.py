import unittest
from unittest.mock import patch

from sifma_news_scraper import SIFMA_NEWS_URL, SIFMANewsScraper, _infer_sifma_doc_type, _is_sifma_detail_url


class _FakeResponse:
    def __init__(self, text: str, url: str, status_code: int = 200):
        self.text = text
        self.url = url
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class SIFMANewsScraperTests(unittest.TestCase):
    def test_detail_url_detection_excludes_listing_pages(self):
        self.assertFalse(_is_sifma_detail_url("https://www.sifma.org/news"))
        self.assertFalse(_is_sifma_detail_url("https://www.sifma.org/news/blog"))
        self.assertFalse(_is_sifma_detail_url("https://www.sifma.org/news/press-releases"))
        self.assertFalse(_is_sifma_detail_url("https://www.sifma.org/news/speeches"))

    def test_detail_url_detection_keeps_article_pages(self):
        self.assertTrue(
            _is_sifma_detail_url(
                "https://www.sifma.org/news/blog/understanding-securities-ownership-in-the-united-states"
            )
        )
        self.assertTrue(
            _is_sifma_detail_url(
                "https://www.sifma.org/news/press-releases/joint-trades-statement-on-faqs-on-the-capital-treatment-of-tokenized-securities"
            )
        )
        self.assertTrue(
            _is_sifma_detail_url(
                "https://www.sifma.org/news/speeches/chairman-remarks-on-market-structure"
            )
        )

    def test_doc_type_inference(self):
        self.assertEqual(
            _infer_sifma_doc_type(
                "Joint Trades Statement on FAQs on the Capital Treatment of Tokenized Securities",
                category="Press Releases",
            ),
            "Press Release",
        )
        self.assertEqual(
            _infer_sifma_doc_type(
                "Understanding Securities Ownership in the United States",
                category="Pennsylvania + Wall",
                url="https://www.sifma.org/news/blog/understanding-securities-ownership-in-the-united-states",
            ),
            "Blog Post",
        )
        self.assertEqual(
            _infer_sifma_doc_type(
                "Chairman Remarks on Market Structure",
                category="Speeches",
            ),
            "Speech",
        )

    @patch.object(SIFMANewsScraper, "_fetch_html")
    def test_discover_documents_parses_listing_cards(self, mock_fetch_html):
        html = """
        <html>
          <body>
            <main>
              <div class="grid">
                <div class="relative h-full">
                  <div class="relative border-l pl-6">
                    <a aria-label="Understanding Securities Ownership in the United States"
                       href="/news/blog/understanding-securities-ownership-in-the-united-states"></a>
                    <div class="flex w-full flex-col gap-4">
                      <div class="flex flex-col gap-3">
                        <div class="flex flex-col gap-[6px]">
                          <div>Pennsylvania + Wall</div>
                          <div>Mar 06, 2026</div>
                          <h3><span>Understanding Securities Ownership in the United States</span></h3>
                        </div>
                        <ul><li><a href="/issues/investor-education-protection/investor-protection">Investor Protection</a></li></ul>
                      </div>
                    </div>
                  </div>
                </div>
                <div class="relative h-full">
                  <div class="relative border-l pl-6">
                    <a aria-label="Joint Trades Statement on FAQs on the Capital Treatment of Tokenized Securities"
                       href="/news/press-releases/joint-trades-statement-on-faqs-on-the-capital-treatment-of-tokenized-securities"></a>
                    <div class="flex w-full flex-col gap-4">
                      <div class="flex flex-col gap-3">
                        <div class="flex flex-col gap-[6px]">
                          <div>Press Releases</div>
                          <div>Mar 05, 2026</div>
                          <h3><span>Joint Trades Statement on FAQs on the Capital Treatment of Tokenized Securities</span></h3>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                <a href="/news/blog">Pennsylvania + Wall</a>
                <a rel="next" href="/news?page=2">Next page</a>
              </div>
            </main>
          </body>
        </html>
        """
        mock_fetch_html.return_value = _FakeResponse(html, SIFMA_NEWS_URL, 200)

        scraper = SIFMANewsScraper(min_delay_seconds=0)
        docs = scraper.discover_documents(max_pages=1)

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["title"], "Understanding Securities Ownership in the United States")
        self.assertEqual(docs[0]["category"], "Pennsylvania + Wall")
        self.assertEqual(docs[0]["doc_type"], "Blog Post")
        self.assertEqual(docs[1]["doc_type"], "Press Release")
        self.assertEqual(scraper.last_discovery_debug["pages"][0]["returned_items"], 2)

    @patch.object(SIFMANewsScraper, "_fetch_html")
    def test_discover_documents_flags_vercel_checkpoint(self, mock_fetch_html):
        checkpoint_html = "<html><head><title>Vercel Security Checkpoint</title></head><body></body></html>"
        mock_fetch_html.return_value = _FakeResponse(checkpoint_html, SIFMA_NEWS_URL, 403)

        scraper = SIFMANewsScraper(min_delay_seconds=0)
        with self.assertRaises(RuntimeError):
            scraper.discover_documents(max_pages=1)

        self.assertTrue(scraper.last_discovery_debug["checkpoint_blocked"])
        self.assertEqual(scraper.last_discovery_debug["stop_reason"], "checkpoint_blocked")


if __name__ == "__main__":
    unittest.main()
