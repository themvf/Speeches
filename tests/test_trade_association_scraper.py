from trade_association_scraper import TradeAssociationScraper
from run_connector_extraction_pipeline import SUPPORTED_CONNECTORS, _default_base_url


class _FakeResponse:
    def __init__(self, text="", url="https://example.org/news"):
        self.text = text
        self.content = text.encode("utf-8")
        self.url = url

    def raise_for_status(self):
        return None


def test_rss_discovery_builds_trade_association_items(monkeypatch):
    rss = """<?xml version="1.0" encoding="utf-8"?>
    <rss version="2.0">
      <channel>
        <item>
          <title>ISDA Publishes Derivatives Market Structure Paper</title>
          <link>https://www.isda.org/a/example-market-structure-paper/</link>
          <pubDate>Mon, 29 Jun 2026 12:00:00 GMT</pubDate>
          <description><![CDATA[<p>Policy recommendations on derivatives market structure.</p>]]></description>
        </item>
      </channel>
    </rss>
    """
    scraper = TradeAssociationScraper(min_delay_seconds=0)
    monkeypatch.setattr(scraper, "_fetch", lambda _url, timeout=45: _FakeResponse(text=rss, url=_url))

    docs = scraper.discover_documents("isda_news_item", include_rss=True, max_pages=1)

    assert len(docs) == 1
    assert docs[0]["source_key"] == "isda_news_item"
    assert docs[0]["source_label"] == "ISDA"
    assert docs[0]["title"] == "ISDA Publishes Derivatives Market Structure Paper"
    assert docs[0]["description"] == "Policy recommendations on derivatives market structure."


def test_html_discovery_filters_same_site_detail_links(monkeypatch):
    html = """
    <main>
      <article>
        <h2>ABA Urges Agencies to Tailor Capital Proposal</h2>
        <p>June 29, 2026</p>
        <a href="/about-us/press-room/press-releases/aba-urges-agencies-to-tailor-capital-proposal">Read more</a>
      </article>
      <a href="/about-us/contact-us">Contact us</a>
    </main>
    """
    scraper = TradeAssociationScraper(min_delay_seconds=0)
    monkeypatch.setattr(scraper, "_fetch", lambda _url, timeout=45: _FakeResponse(text=html, url=_url))

    docs = scraper.discover_documents("aba_news_item", include_rss=False, max_pages=1)

    assert len(docs) == 1
    assert docs[0]["url"] == "https://www.aba.com/about-us/press-room/press-releases/aba-urges-agencies-to-tailor-capital-proposal"
    assert docs[0]["date"] == "June 29, 2026"
    assert "ABA Urges Agencies" in docs[0]["title"]


def test_extract_document_uses_article_body(monkeypatch):
    html = """
    <html><body>
      <article>
        <h1>ICI Comments on ETF Disclosure Proposal</h1>
        <time datetime="2026-06-29">June 29, 2026</time>
        <p>The Investment Company Institute submitted comments on ETF disclosure policy.</p>
        <p>The letter focuses on investor protection, fund operations, and market structure.</p>
      </article>
    </body></html>
    """
    scraper = TradeAssociationScraper(min_delay_seconds=0)
    monkeypatch.setattr(scraper, "_fetch", lambda _url, timeout=90: _FakeResponse(text=html, url=_url))

    result = scraper.extract_document({"url": "https://www.ici.org/news-releases/example", "source_label": "ICI"})

    assert result["success"] is True
    assert result["data"]["title"] == "ICI Comments on ETF Disclosure Proposal"
    assert result["data"]["date"] == "June 29, 2026"
    assert "ETF disclosure policy" in result["data"]["full_text"]


def test_runner_supports_trade_association_connectors():
    for connector in [
        "ici_news_item",
        "isda_news_item",
        "mfa_news_item",
        "fia_news_item",
        "aba_news_item",
        "bpi_news_item",
        "icba_news_item",
        "lsta_news_item",
    ]:
        assert connector in SUPPORTED_CONNECTORS
        assert _default_base_url(connector)
