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


def test_occ_fdic_cfpb_rss_sources_discover_and_are_registered():
    from securities_market_sources_scraper import SECURITIES_MARKET_SOURCES

    for key in ("occ_news_release", "fdic_press_release", "cfpb_newsroom"):
        cfg = SECURITIES_MARKET_SOURCES[key]
        assert cfg["discovery"] == "rss"
        assert cfg["default_url"].startswith("https://")


def test_nydfs_html_link_discovery_filters_by_path_and_skips_listing_page(monkeypatch):
    html = """
    <div class="views-row">
      <a href="/reports_and_publications/press_releases/pr20260701">Governor Hochul Issues New Guidance</a>
      <p>July 1, 2026</p>
    </div>
    <div class="views-row">
      <a href="/reports_and_publications/press_releases">VIEW ALL PRESS RELEASES</a>
    </div>
    <a href="/reports_and_publications/press_releases/pr20260609">Stablecoin Framework Proposed Regulation</a>
    """
    scraper = SecuritiesMarketSourcesScraper(min_delay_seconds=0)
    monkeypatch.setattr(scraper, "_fetch", lambda _url, timeout=45: _FakeResponse(text=html, url=_url))

    docs = scraper.discover_documents(
        "nydfs_press_release", base_url="https://www.dfs.ny.gov/reports_and_publications/press_releases"
    )

    urls = [d["url"] for d in docs]
    assert "https://www.dfs.ny.gov/reports_and_publications/press_releases/pr20260701" in urls
    assert "https://www.dfs.ny.gov/reports_and_publications/press_releases/pr20260609" in urls
    # The "view all" link back to the listing page itself must not be treated as a document.
    assert "https://www.dfs.ny.gov/reports_and_publications/press_releases" not in urls


def test_extract_document_prefers_richest_body_over_first_matching_selector(monkeypatch):
    """Regression test: a plain `or`-chain over selectors picks whichever is
    *present* first, even if it holds almost no text. Some sites (e.g. NYDFS)
    have a near-empty <article> wrapper while the real content lives in a
    later selector like div.field--name-body - this must not truncate every
    extraction from that site to a few words."""
    html = """
    <html><body>
      <article>Back To Newsroom Governor Announces New Policy</article>
      <main>
        <article>Back To Newsroom Governor Announces New Policy</article>
        <div class="field--name-body">
          <p>Governor Announces New Policy</p>
          <p>The department today announced a sweeping new policy affecting thousands of regulated entities
          across the state, following months of stakeholder engagement and public comment on the proposal.</p>
          <p>Officials said the change reflects lessons learned from recent examinations and is intended to
          strengthen consumer protections while preserving a competitive marketplace for financial services.</p>
        </div>
      </main>
    </body></html>
    """
    scraper = SecuritiesMarketSourcesScraper(min_delay_seconds=0)
    monkeypatch.setattr(scraper, "_fetch", lambda _url, timeout=90: _FakeResponse(text=html, url=_url))

    result = scraper.extract_document({"url": "https://www.dfs.ny.gov/example", "title": "Fallback"})

    assert result["success"] is True
    assert "sweeping new policy" in result["data"]["full_text"]
    assert result["data"]["word_count"] > 30


def test_extract_document_still_prefers_article_when_it_has_the_real_content(monkeypatch):
    """No-regression check: when <article> genuinely holds the richest
    content (the common case for most existing sources), it must still win."""
    html = """
    <html><body>
      <article>
        <h1>MSRB Rule Filing</h1>
        <p>The MSRB advanced municipal securities market transparency initiatives this quarter,
        publishing updated guidance for dealers and municipal advisors on recordkeeping obligations.</p>
      </article>
      <div class="field--name-body">short</div>
    </body></html>
    """
    scraper = SecuritiesMarketSourcesScraper(min_delay_seconds=0)
    monkeypatch.setattr(scraper, "_fetch", lambda _url, timeout=90: _FakeResponse(text=html, url=_url))

    result = scraper.extract_document({"url": "https://www.msrb.org/example", "title": "Fallback"})

    assert result["success"] is True
    assert "municipal advisors" in result["data"]["full_text"]


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
