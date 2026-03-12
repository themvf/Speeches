import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup

from treasury_news_scraper import (
    TREASURY_STATEMENTS_REMARKS_URL,
    TreasuryNewsScraper,
    _find_next_page_url,
    _infer_treasury_doc_type,
    _is_treasury_detail_url,
    _parse_date_text,
)


class TreasuryNewsScraperHelpersTests(unittest.TestCase):
    def test_statement_and_remark_detail_url_detection_excludes_section_pages(self):
        self.assertFalse(
            _is_treasury_detail_url(
                "https://home.treasury.gov/news/press-releases",
                "treasury_statement_remark",
            )
        )
        self.assertFalse(
            _is_treasury_detail_url(
                "https://home.treasury.gov/news/press-releases/readouts",
                "treasury_statement_remark",
            )
        )
        self.assertFalse(
            _is_treasury_detail_url(
                "https://home.treasury.gov/news/press-releases/testimonies",
                "treasury_statement_remark",
            )
        )
        self.assertFalse(
            _is_treasury_detail_url(
                "https://home.treasury.gov/news/press-releases/statements-remarks",
                "treasury_statement_remark",
            )
        )
        self.assertFalse(
            _is_treasury_detail_url(
                "https://home.treasury.gov/news/press-releases/statements-remarks/secretary",
                "treasury_statement_remark",
            )
        )

    def test_statement_and_remark_detail_url_detection_keeps_actual_documents(self):
        self.assertTrue(
            _is_treasury_detail_url(
                "https://home.treasury.gov/news/press-releases/jy3142",
                "treasury_statement_remark",
            )
        )
        self.assertTrue(
            _is_treasury_detail_url(
                "https://home.treasury.gov/news/press-releases/sb0142",
                "treasury_statement_remark",
            )
        )

    def test_doc_type_inference(self):
        self.assertEqual(
            _infer_treasury_doc_type(
                "treasury_statement_remark",
                "Readout of Secretary Bessent's Meeting With G7 Finance Ministers",
            ),
            "Readout",
        )
        self.assertEqual(
            _infer_treasury_doc_type(
                "treasury_statement_remark",
                "Testimony of Deputy Secretary Before the House Financial Services Committee",
            ),
            "Testimony",
        )
        self.assertEqual(
            _infer_treasury_doc_type(
                "treasury_statement_remark",
                "Remarks by Secretary of the Treasury at the Economic Club of New York",
            ),
            "Remarks",
        )
        self.assertEqual(
            _infer_treasury_doc_type(
                "treasury_statement_remark",
                "Statement from Secretary of the Treasury Before the Senate Banking Committee",
                text="Secretary Statements & Remarks",
            ),
            "Statement",
        )

    def test_parse_date_text_handles_treasury_iso_datetime(self):
        parsed = _parse_date_text("2026-03-12T16:30:00Z")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.strftime("%Y-%m-%d %H:%M:%S"), "2026-03-12 16:30:00")

    def test_find_next_page_url_keeps_query_string_pagination(self):
        soup = BeautifulSoup(
            '<nav class="pager"><a href="?page=1" title="Go to next page">Next page</a></nav>',
            "html.parser",
        )
        self.assertEqual(
            _find_next_page_url(soup, TREASURY_STATEMENTS_REMARKS_URL),
            f"{TREASURY_STATEMENTS_REMARKS_URL}?page=1",
        )

    @patch.object(TreasuryNewsScraper, "_discover_from_listing_page")
    def test_statement_and_remark_discovery_scans_all_treasury_sections(self, mock_discover):
        def _fake_discover(page_url, source_key):
            self.assertEqual(source_key, "treasury_statement_remark")
            return (
                [
                    {
                        "url": f"{page_url.rstrip('/').replace('?page=1', '')}/doc",
                        "title": page_url,
                        "date": "March 12, 2026",
                        "speaker": "",
                        "doc_type": "Statement",
                        "source_format": "html",
                        "listing_page": page_url,
                    }
                ],
                "",
            )

        mock_discover.side_effect = _fake_discover

        scraper = TreasuryNewsScraper(min_delay_seconds=0)
        docs = scraper.discover_documents("treasury_statement_remark", max_pages=1)

        scanned_urls = [call.args[0] for call in mock_discover.call_args_list]
        self.assertEqual(
            scanned_urls,
            [
                "https://home.treasury.gov/news/press-releases/statements-remarks",
                "https://home.treasury.gov/news/press-releases/readouts",
                "https://home.treasury.gov/news/press-releases/testimonies",
            ],
        )
        self.assertEqual(len(docs), 3)


if __name__ == "__main__":
    unittest.main()
