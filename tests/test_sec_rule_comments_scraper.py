import unittest

from sec_rule_comments_scraper import SECRuleCommentsScraper


class _FakeResponse:
    def __init__(self, url: str, text: str = "", content: bytes = b"", headers=None):
        self.url = url
        self.text = text
        self.content = content if content else text.encode("utf-8")
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}

    def raise_for_status(self):
        return None


class _FakeSession:
    def __init__(self, responses):
        self._responses = responses
        self.headers = {}
        self.requested_urls = []

    def get(self, url: str, timeout: int = 60, allow_redirects: bool = True):
        self.requested_urls.append(url)
        response = self._responses.get(url)
        if response is None:
            raise AssertionError(f"Unexpected URL requested: {url}")
        return response


class _FakeScraper(SECRuleCommentsScraper):
    def __init__(self, responses):
        super().__init__(min_delay_seconds=0.0)
        self._responses = responses

    def _fetch(self, url: str, timeout: int = 60):
        response = self._responses.get(url)
        if response is None:
            raise AssertionError(f"Unexpected URL requested: {url}")
        return response


class SECRuleCommentsScraperTests(unittest.TestCase):
    def test_discover_documents_strips_rule_url_fragment_before_fetch(self):
        input_rule_url = "https://www.sec.gov/rules-regulations/2026/03/s7-2026-09#33-11412interpretive"
        rule_url = "https://www.sec.gov/rules-regulations/2026/03/s7-2026-09"
        rule_html = """
        <html><body>
        <h1>Application of the Federal Securities Laws to Certain Types of Crypto Assets</h1>
        <div>
          <p>File Number</p><p>S7-2026-09</p>
          <p>Release Number</p><p>33-11412</p>
          <p>SEC Issue Date</p><p>March 17, 2026</p>
        </div>
        </body></html>
        """
        scraper = SECRuleCommentsScraper(min_delay_seconds=0.0)
        fake_session = _FakeSession({rule_url: _FakeResponse(rule_url, text=rule_html)})
        scraper.session = fake_session

        docs = scraper.discover_documents(input_rule_url, include_pdfs=False)

        self.assertGreaterEqual(len(fake_session.requested_urls), 1)
        self.assertEqual(fake_session.requested_urls[0], rule_url)
        self.assertTrue(all("#" not in requested_url for requested_url in fake_session.requested_urls))
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["entry_kind"], "rule")
        self.assertEqual(docs[0]["url"], rule_url)
        self.assertEqual(docs[0]["file_number"], "S7-2026-09")

    def test_discover_documents_returns_rule_and_comments(self):
        rule_url = "https://www.sec.gov/rules-regulations/2026/03/s7-2026-09"
        comments_url = "https://www.sec.gov/rules-regulations/public-comments/s7-2026-09"
        comment_pdf = "https://www.sec.gov/comments/s7-2026-09/example-comment.pdf"
        comment_html = "https://www.sec.gov/comments/s7-2026-09/example-comment.htm"
        rule_html = """
        <html><body>
        <h1>Application of the Federal Securities Laws to Certain Types of Crypto Assets</h1>
        <a href="/rules-regulations/public-comments/s7-2026-09">View Received Comments</a>
        <a href="/files/rules/other/2026/33-11412.pdf">Interpretive Release: SEC Issued Version ( pdf )</a>
        <div>
          <p>Rule Type</p><p>Interpretive</p>
          <p>File Number</p><p>S7-2026-09</p>
          <p>Release Number</p><p>33-11412</p><p>34-105020</p>
          <p>SEC Issue Date</p><p>March 17, 2026</p>
          <p>Effective Date</p><p>March 23, 2026</p>
          <p>Federal Register Publish Date</p><p>March 23, 2026</p>
        </div>
        </body></html>
        """
        comments_html = f"""
        <html><body>
        <table>
          <tbody>
            <tr>
              <td>March 24, 2026</td>
              <td>Public Comment</td>
              <td><a href="{comment_pdf}">Anonymous</a></td>
            </tr>
            <tr>
              <td>March 20, 2026</td>
              <td>Public Comment</td>
              <td><a href="{comment_html}">Brandon Ferrick, General Counsel, Douro Labs LLC</a></td>
            </tr>
          </tbody>
        </table>
        </body></html>
        """

        scraper = _FakeScraper(
            {
                rule_url: _FakeResponse(rule_url, text=rule_html),
                comments_url: _FakeResponse(comments_url, text=comments_html),
            }
        )

        docs = scraper.discover_documents(rule_url, include_pdfs=True)

        self.assertEqual(len(docs), 3)
        self.assertEqual(docs[0]["entry_kind"], "rule")
        self.assertEqual(docs[0]["file_number"], "S7-2026-09")
        self.assertEqual(docs[0]["release_numbers"], ["33-11412", "34-105020"])
        self.assertEqual(docs[1]["entry_kind"], "comment")
        self.assertEqual(docs[1]["url"], comment_pdf)
        self.assertEqual(docs[1]["commenter_name"], "Anonymous")
        self.assertEqual(docs[2]["url"], comment_html)
        self.assertEqual(docs[2]["notice_number"], "S7-2026-09")

    def test_extract_comment_handles_plain_text_comment(self):
        comment_url = "https://www.sec.gov/comments/s7-2026-09/example-comment.txt"
        comment_text = (
            "We support the Commission's effort to clarify the application of the securities laws.\n"
            "The agency should preserve this interpretive approach and finalize it promptly."
        )
        scraper = _FakeScraper(
            {
                comment_url: _FakeResponse(
                    comment_url,
                    text=comment_text,
                    headers={"Content-Type": "text/plain; charset=utf-8"},
                )
            }
        )

        result = scraper.extract_comment(
            comment_url,
            fallback_title="Comment from Example Organization",
            fallback_date="March 20, 2026",
            fallback_commenter_name="Example Organization",
            fallback_file_number="S7-2026-09",
            fallback_release_numbers=["33-11412", "34-105020"],
            fallback_rule_title="Application of the Federal Securities Laws to Certain Types of Crypto Assets",
            fallback_rule_url="https://www.sec.gov/rules-regulations/2026/03/s7-2026-09",
            fallback_comments_url="https://www.sec.gov/rules-regulations/public-comments/s7-2026-09",
            fallback_letter_type="Public Comment",
        )

        self.assertTrue(result["success"])
        data = result["data"]
        self.assertEqual(data["source_format"], "txt")
        self.assertEqual(data["file_number"], "S7-2026-09")
        self.assertEqual(data["commenter_name"], "Example Organization")
        self.assertIn("Rule Title:", data["full_text"])
        self.assertIn("We support the Commission's effort", data["full_text"])


if __name__ == "__main__":
    unittest.main()

