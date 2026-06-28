from securities_market_sources_scraper import SecuritiesMarketSourcesScraper


class _FakeResponse:
    def __init__(self, text="", content=b"", url="https://example.com/doc"):
        self.text = text
        self.content = content or text.encode("utf-8")
        self.url = url

    def raise_for_status(self):
        return None


def test_sec_rss_discovery_parses_items(monkeypatch):
    rss = """<?xml version="1.0" encoding="utf-8"?>
    <rss version="2.0">
      <channel>
        <item>
          <title>SEC Charges Market Manipulation Scheme</title>
          <link>https://www.sec.gov/newsroom/press-releases/2026-10</link>
          <pubDate>Fri, 26 Jun 2026 12:00:00 -0400</pubDate>
          <description><![CDATA[<p>Market manipulation and disclosure failures.</p>]]></description>
        </item>
      </channel>
    </rss>
    """
    scraper = SecuritiesMarketSourcesScraper(min_delay_seconds=0)
    monkeypatch.setattr(scraper, "_fetch", lambda _url, timeout=45: _FakeResponse(text=rss, url=_url))

    docs = scraper.discover_documents("sec_press_release_rss", max_pages=1)

    assert len(docs) == 1
    assert docs[0]["title"] == "SEC Charges Market Manipulation Scheme"
    assert docs[0]["source_key"] == "sec_press_release_rss"
    assert docs[0]["source_format"] == "html"
    assert docs[0]["summary"] == "Market manipulation and disclosure failures."


def test_sec_pcaob_rulemaking_discovers_pdf_links(monkeypatch):
    html = """
    <table>
      <tr><th>Release No.</th><th>Date</th><th>Details</th></tr>
      <tr>
        <td><a href="/files/rules/pcaob/2026/34-105001.pdf">34-105001</a></td>
        <td>June 26, 2026</td>
        <td>Notice of PCAOB rulemaking on audit standards</td>
      </tr>
    </table>
    """
    scraper = SecuritiesMarketSourcesScraper(min_delay_seconds=0)
    monkeypatch.setattr(scraper, "_fetch", lambda _url, timeout=45: _FakeResponse(text=html, url=_url))

    docs = scraper.discover_documents("sec_pcaob_rulemaking", max_pages=1)

    assert len(docs) == 1
    assert docs[0]["url"] == "https://www.sec.gov/files/rules/pcaob/2026/34-105001.pdf"
    assert docs[0]["source_format"] == "pdf"
    assert docs[0]["date"] == "June 26, 2026"


def test_html_link_discovery_uses_container_title(monkeypatch):
    html = """
    <div class="card">
      <h3>MSRB Advances Market Transparency Initiative</h3>
      <p>June 26, 2026</p>
      <a href="/Press-Releases/MSRB-Advances-Market-Transparency">Read more</a>
    </div>
    """
    scraper = SecuritiesMarketSourcesScraper(min_delay_seconds=0)
    monkeypatch.setattr(scraper, "_fetch", lambda _url, timeout=45: _FakeResponse(text=html, url=_url))

    docs = scraper.discover_documents("msrb_press_release", base_url="https://www.msrb.org/Press-Releases")

    assert len(docs) == 1
    assert docs[0]["url"] == "https://www.msrb.org/Press-Releases/MSRB-Advances-Market-Transparency"
    assert "MSRB Advances Market Transparency" in docs[0]["title"]
    assert docs[0]["date"] == "June 26, 2026"


def test_extract_document_from_html(monkeypatch):
    html = """
    <html><body>
      <main>
        <h1>MSRB Rule Filing</h1>
        <p>June 26, 2026</p>
        <p>The MSRB advanced municipal securities market transparency.</p>
      </main>
    </body></html>
    """
    scraper = SecuritiesMarketSourcesScraper(min_delay_seconds=0)
    monkeypatch.setattr(scraper, "_fetch", lambda _url, timeout=90: _FakeResponse(text=html, url=_url))

    result = scraper.extract_document({"url": "https://www.msrb.org/Press-Releases/example", "title": "Fallback"})

    assert result["success"] is True
    assert result["data"]["title"] == "MSRB Rule Filing"
    assert "municipal securities" in result["data"]["full_text"]


def test_extract_document_keeps_rss_title_when_page_heading_is_generic(monkeypatch):
    html = """
    <html><body>
      <main>
        <h1>Notice</h1>
        <p>June 29, 2026</p>
        <p>The Securities and Exchange Commission is publishing this Federal Register notice.</p>
      </main>
    </body></html>
    """
    scraper = SecuritiesMarketSourcesScraper(min_delay_seconds=0)
    monkeypatch.setattr(scraper, "_fetch", lambda _url, timeout=90: _FakeResponse(text=html, url=_url))

    result = scraper.extract_document(
        {
            "url": "https://www.federalregister.gov/documents/example",
            "title": "Agency Information Collection Activities; Proposed Collection; Comment Request",
        }
    )

    assert result["success"] is True
    assert result["data"]["title"] == "Agency Information Collection Activities; Proposed Collection; Comment Request"


def test_extract_document_prefers_listing_date_over_future_body_date(monkeypatch):
    html = """
    <html><body>
      <main>
        <h1>Financial Data Transparency Act Joint Data Standards</h1>
        <p>Effective Date: October 01, 2026</p>
        <p>The agencies are publishing this final joint rule.</p>
      </main>
    </body></html>
    """
    scraper = SecuritiesMarketSourcesScraper(min_delay_seconds=0)
    monkeypatch.setattr(scraper, "_fetch", lambda _url, timeout=90: _FakeResponse(text=html, url=_url))

    result = scraper.extract_document(
        {
            "url": "https://www.federalregister.gov/documents/example",
            "title": "Financial Data Transparency Act Joint Data Standards",
            "date": "June 25, 2026",
        }
    )

    assert result["success"] is True
    assert result["data"]["date"] == "June 25, 2026"


def test_extract_document_uses_metadata_fallback_when_detail_fetch_fails(monkeypatch):
    scraper = SecuritiesMarketSourcesScraper(min_delay_seconds=0)

    def _raise(_url, timeout=90):
        raise RuntimeError("500 Server Error")

    monkeypatch.setattr(scraper, "_fetch", _raise)

    result = scraper.extract_document(
        {
            "url": "https://www.federalregister.gov/documents/example",
            "title": "Submission for OMB Review; Comment Request; Extension: Rule 30e-1",
            "date": "June 26, 2026",
            "summary": "",
            "source_format": "html",
        }
    )

    assert result["success"] is True
    assert result["data"]["date"] == "June 26, 2026"
    assert result["data"]["extraction_mode"] == "metadata_fallback"
    assert "Submission for OMB Review" in result["data"]["full_text"]
