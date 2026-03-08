import unittest

from regulations_gov_manual_scraper import (
    _download_attachment_url,
    _download_content_url,
    _parse_regulations_url,
    _public_comment_url,
    _public_document_url,
    _extract_urls_from_value,
)


class RegulationsGovManualScraperHelpersTests(unittest.TestCase):
    def test_parse_public_urls(self):
        self.assertEqual(
            _parse_regulations_url("https://www.regulations.gov/docket/CFPB-2025-0037"),
            {
                "url": "https://www.regulations.gov/docket/CFPB-2025-0037",
                "kind": "docket",
                "docket_id": "CFPB-2025-0037",
                "document_id": "",
                "comment_id": "",
                "record_id": "",
                "download_kind": "",
                "download_filename": "",
            },
        )
        self.assertEqual(
            _parse_regulations_url("https://www.regulations.gov/document/CFPB-2025-0037-0001")["document_id"],
            "CFPB-2025-0037-0001",
        )
        self.assertEqual(
            _parse_regulations_url("https://www.regulations.gov/comment/CFPB-2025-0037-4964")["comment_id"],
            "CFPB-2025-0037-4964",
        )

    def test_parse_download_url(self):
        parsed = _parse_regulations_url(
            "https://downloads.regulations.gov/CFPB-2025-0037-4964/attachment_1.pdf"
        )
        self.assertEqual(parsed["kind"], "download")
        self.assertEqual(parsed["record_id"], "CFPB-2025-0037-4964")
        self.assertEqual(parsed["download_kind"], "attachment")
        self.assertEqual(parsed["download_filename"], "attachment_1.pdf")

    def test_download_url_builders(self):
        self.assertEqual(
            _download_content_url("CFPB-2025-0037-0001", "pdf"),
            "https://downloads.regulations.gov/CFPB-2025-0037-0001/content.pdf",
        )
        self.assertEqual(
            _download_attachment_url("CFPB-2025-0037-4964", 1, "pdf"),
            "https://downloads.regulations.gov/CFPB-2025-0037-4964/attachment_1.pdf",
        )
        self.assertEqual(
            _public_document_url("CFPB-2025-0037-0001"),
            "https://www.regulations.gov/document/CFPB-2025-0037-0001",
        )
        self.assertEqual(
            _public_comment_url("CFPB-2025-0037-4964"),
            "https://www.regulations.gov/comment/CFPB-2025-0037-4964",
        )

    def test_extract_urls_from_nested_payload(self):
        payload = {
            "data": {
                "id": "comment-1",
                "attributes": {
                    "downloadUrl": "https://downloads.regulations.gov/CFPB-2025-0037-4964/content.pdf"
                },
            },
            "included": [
                {
                    "id": "attachment-1",
                    "attributes": {
                        "fileFormats": [
                            "https://downloads.regulations.gov/CFPB-2025-0037-4964/attachment_1.pdf",
                            "https://downloads.regulations.gov/CFPB-2025-0037-4964/attachment_1.txt",
                        ]
                    },
                }
            ],
        }

        urls = _extract_urls_from_value(payload)
        self.assertEqual(
            urls,
            [
                "https://downloads.regulations.gov/CFPB-2025-0037-4964/content.pdf",
                "https://downloads.regulations.gov/CFPB-2025-0037-4964/attachment_1.pdf",
                "https://downloads.regulations.gov/CFPB-2025-0037-4964/attachment_1.txt",
            ],
        )


if __name__ == "__main__":
    unittest.main()
