from hedge_fund_letter_scraper import HedgeFundLetterScraper
from run_connector_extraction_pipeline import SUPPORTED_CONNECTORS, _default_base_url


class _FakeResponse:
    def __init__(self, text="", url="https://fiscal.ai/fund-letters/", content=None, content_type="text/html"):
        self.text = text
        self.content = content if content is not None else text.encode("utf-8")
        self.url = url
        self.headers = {"content-type": content_type}

    def raise_for_status(self):
        return None


def test_fiscal_discovery_collects_recent_letter_links(monkeypatch):
    html = """
    <main>
      <a href="/super-investors/">Super Investors</a>
      <div>
        <a href="https://givernycapital.com/wp-content/uploads/2026/03/giverny-capital-annual-letter-2025.pdf">
          Giverny Capital Mar 28
        </a>
      </div>
      <div>
        <a href="https://content.haydencapital.com/Hayden-Capital-Quarterly-Letter-2025-Q4.pdf">
          Hayden Capital Feb 26
        </a>
      </div>
    </main>
    """
    scraper = HedgeFundLetterScraper(min_delay_seconds=0)

    def fake_fetch(url, timeout=45):
        if "fiscal.ai" in url:
            return _FakeResponse(text=html, url=url)
        return _FakeResponse(text="<main></main>", url=url)

    monkeypatch.setattr(scraper, "_fetch", fake_fetch)

    docs = scraper.discover_documents(base_url="https://fiscal.ai/fund-letters/", max_pages=1)

    assert len(docs) == 2
    assert docs[0]["source_key"] == "fiscal_ai_fund_letters"
    assert docs[0]["source_label"] == "Fiscal.ai Fund Letters"
    assert docs[0]["date"] == "March 28, 2026"
    assert docs[0]["source_format"] == "pdf"
    assert docs[0]["fund_name"] == "Giverny Capital"


def test_extract_html_letter_body(monkeypatch):
    html = """
    <html><body>
      <main>
        <h1>Example Fund Q1 2026 Letter</h1>
        <p>April 15, 2026</p>
        <p>The fund discussed portfolio performance, market structure, credit spreads, and regulatory uncertainty.</p>
        <p>Management explained position sizing, risk controls, and the opportunity set for long-term investors.</p>
      </main>
    </body></html>
    """
    scraper = HedgeFundLetterScraper(min_delay_seconds=0)
    monkeypatch.setattr(scraper, "_fetch", lambda _url, timeout=90: _FakeResponse(text=html, url=_url))

    result = scraper.extract_document(
        {
            "url": "https://example.com/q1-2026-letter",
            "title": "Fallback Letter",
            "source_label": "Example Source",
            "fund_name": "Example Fund",
        }
    )

    assert result["success"] is True
    assert result["data"]["title"] == "Example Fund Q1 2026 Letter"
    assert result["data"]["date"] == "April 15, 2026"
    assert "portfolio performance" in result["data"]["full_text"]
    assert result["data"]["source_format"] == "html"


def test_runner_supports_hedge_fund_letter_connector():
    assert "hedge_fund_letter" in SUPPORTED_CONNECTORS
    assert _default_base_url("hedge_fund_letter") == "https://fiscal.ai/fund-letters/"
