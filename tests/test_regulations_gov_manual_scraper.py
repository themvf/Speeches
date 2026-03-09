import unittest

from regulations_gov_manual_scraper import (
    _download_attachment_url,
    _download_content_url,
    _infer_commenter_identity,
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

    def test_infer_commenter_identity_from_summary_subject(self):
        inferred = _infer_commenter_identity(
            title="CFPB-2025-0037-13872",
            summary=(
                "The National Retail Federation (NRF) files a public comment strongly supporting the "
                "CFPB's reconsideration of its Personal Financial Data Rights rule."
            ),
            body_text="",
        )
        self.assertEqual(inferred["commenter_name"], "")
        self.assertEqual(inferred["commenter_org"], "The National Retail Federation (NRF)")

    def test_infer_commenter_identity_from_title(self):
        inferred = _infer_commenter_identity(
            title="Comment from Cato Institute",
            summary="",
            body_text="",
        )
        self.assertEqual(inferred["commenter_name"], "")
        self.assertEqual(inferred["commenter_org"], "Cato Institute")

    def test_infer_commenter_identity_from_labeled_body_line(self):
        inferred = _infer_commenter_identity(
            title="Public Comment",
            summary="",
            body_text="Submitted by: Consumer Bankers Association\nRe: CFPB-2025-0037",
        )
        self.assertEqual(inferred["commenter_name"], "")
        self.assertEqual(inferred["commenter_org"], "Consumer Bankers Association")

    def test_infer_commenter_identity_person_name(self):
        inferred = _infer_commenter_identity(
            title="Comment from Jane Doe",
            summary="",
            body_text="",
        )
        self.assertEqual(inferred["commenter_name"], "Jane Doe")
        self.assertEqual(inferred["commenter_org"], "")


if __name__ == "__main__":
    unittest.main()
