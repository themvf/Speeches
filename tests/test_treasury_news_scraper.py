import unittest

from treasury_news_scraper import _infer_treasury_doc_type, _is_treasury_detail_url


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


if __name__ == "__main__":
    unittest.main()
