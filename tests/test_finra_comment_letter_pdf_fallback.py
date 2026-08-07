"""A scanned comment letter must degrade to metadata, not fail the run.

FINRA's 25-06 comment page carries an image-only PDF (Davis Polk, 4 pages, no
text layer). Raising on it meant the single new item failed, processed_count
stayed 0, and the daily monitor exited 1 - every day, permanently, while the
letter was missing from the notices page entirely.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import finra_comment_letter_scraper as scraper  # noqa: E402
from neon_feeds import METADATA_FALLBACK_TEXT_MARKER  # noqa: E402

PDF_URL = "https://www.finra.org/sites/default/files/NoticeComment/Davis%20Polk_25-06.pdf"


class _Response:
    def __init__(self, url, content=b"%PDF-1.4 fake"):
        self.url = url
        self.content = content
        self.text = ""


@pytest.fixture
def pdf_scraper(monkeypatch):
    instance = scraper.FINRACommentLetterScraper()
    monkeypatch.setattr(instance, "_fetch", lambda url, timeout=60: _Response(PDF_URL))
    return instance


def test_marker_helper_matches_the_shared_constant():
    """A drifted literal would silently re-enable enrichment on stubs."""
    assert scraper._metadata_fallback_marker() == METADATA_FALLBACK_TEXT_MARKER


def test_scanned_pdf_yields_a_metadata_record_instead_of_raising(pdf_scraper, monkeypatch):
    monkeypatch.setattr(pdf_scraper, "_extract_pdf_text", lambda content: "")

    result = pdf_scraper.extract_document(
        PDF_URL,
        fallback_title="Davis Polk & Wardwell LLP",
        fallback_date="2025-06-18",
        fallback_commenter_name="Davis Polk & Wardwell LLP",
        fallback_notice_number="25-06",
        fallback_notice_url="https://www.finra.org/rules-guidance/notices/25-06",
    )

    assert result["success"] is True
    data = result["data"]
    assert data["extraction_mode"] == "metadata_fallback"
    assert data["pdf_url"] == PDF_URL

    text = data["full_text"]
    # The marker is what keeps the stub out of LLM enrichment.
    assert METADATA_FALLBACK_TEXT_MARKER in text
    assert "scanned PDF with no text layer" in text
    # The notices route parses these header lines to name the commenter.
    assert "Commenter: Davis Polk & Wardwell LLP" in text
    assert "Notice Number: 25-06" in text
    assert f"Source URL: {PDF_URL}" in text


def test_extractable_pdf_is_unaffected(pdf_scraper, monkeypatch):
    monkeypatch.setattr(pdf_scraper, "_extract_pdf_text", lambda content: "Real letter body text.")

    data = pdf_scraper.extract_document(PDF_URL, fallback_commenter_name="Someone")["data"]

    assert data["extraction_mode"] == "pdf_text"
    assert "Real letter body text." in data["full_text"]
    assert METADATA_FALLBACK_TEXT_MARKER not in data["full_text"]


def test_html_comment_page_still_raises_when_empty(pdf_scraper, monkeypatch):
    """Only the no-text-layer PDF case is a permanent, known cause."""
    html_url = "https://www.finra.org/rules-guidance/notices/comment/someone-comment"
    monkeypatch.setattr(pdf_scraper, "_fetch", lambda url, timeout=60: _Response(html_url))

    with pytest.raises(RuntimeError, match="No text extracted from FINRA comment page"):
        pdf_scraper.extract_document(html_url)
