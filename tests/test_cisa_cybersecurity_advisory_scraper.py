from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cisa_cybersecurity_advisory_scraper as cisa
import run_connector_extraction_pipeline as pipeline


@dataclass
class FakeResponse:
    text: str
    url: str = "https://www.cisa.gov/news-events/cybersecurity-advisories/aa26-097a"
    status_code: int = 200

    @property
    def content(self) -> bytes:
        return self.text.encode("utf-8")

    def raise_for_status(self) -> None:
        return None


def test_cisa_rss_discovery_parses_official_items(monkeypatch):
    rss = """
    <rss>
      <channel>
        <item>
          <title>AA26-097A: Example Threat Advisory</title>
          <link>https://www.cisa.gov/news-events/cybersecurity-advisories/aa26-097a</link>
          <pubDate>Wed, 01 Jul 2026 12:00:00 +0000</pubDate>
          <description>Example advisory summary.</description>
        </item>
        <item>
          <title>ICSA-26-183-02 Example Control System Advisory</title>
          <link>https://www.cisa.gov/news-events/ics-advisories/icsa-26-183-02</link>
          <pubDate>Tue, 30 Jun 2026 12:00:00 +0000</pubDate>
          <description>Example ICS advisory summary.</description>
        </item>
      </channel>
    </rss>
    """
    scraper = cisa.CISACybersecurityAdvisoryScraper(min_delay_seconds=0)
    monkeypatch.setattr(scraper, "_fetch", lambda *args, **kwargs: FakeResponse(rss, url=args[0]))

    docs = scraper._discover_from_rss()

    assert [doc["doc_type"] for doc in docs] == ["Cybersecurity Advisory", "ICS Advisory"]
    assert docs[0]["alert_code"] == "AA26-097A"
    assert docs[0]["date"] == "July 01, 2026"


def test_cisa_listing_discovery_parses_article_rows(monkeypatch):
    html = """
    <html><body>
      <article>
        <time datetime="2026-07-01">July 01, 2026</time>
        <a href="/news-events/alerts/2026/07/01/cisa-adds-one-known-exploited-vulnerability-catalog">
          CISA Adds One Known Exploited Vulnerability to Catalog
        </a>
        <p>Known exploited vulnerability catalog update.</p>
      </article>
    </body></html>
    """
    scraper = cisa.CISACybersecurityAdvisoryScraper(min_delay_seconds=0)
    monkeypatch.setattr(scraper, "_fetch", lambda *args, **kwargs: FakeResponse(html, url=args[0]))

    docs = scraper.discover_documents(include_rss=False, max_pages=1)

    assert len(docs) == 1
    assert docs[0]["doc_type"] == "KEV Alert"
    assert docs[0]["date"] == "July 01, 2026"
    assert docs[0]["url"].startswith("https://www.cisa.gov/news-events/alerts/")


def test_cisa_detail_extraction_returns_full_text(monkeypatch):
    body = " ".join(["CISA"] + ["published"] * 30 + ["guidance"] * 30)
    html = f"""
    <html>
      <head>
        <link rel="canonical" href="https://www.cisa.gov/news-events/cybersecurity-advisories/aa26-097a" />
      </head>
      <body>
        <main>
          <h1>AA26-097A: Example Threat Advisory</h1>
          <div class="c-field--name-field-release-date"><time datetime="2026-07-01">July 01, 2026</time></div>
          <section class="l-page-section"><p>{body}</p></section>
        </main>
      </body>
    </html>
    """
    scraper = cisa.CISACybersecurityAdvisoryScraper(min_delay_seconds=0)
    monkeypatch.setattr(scraper, "_fetch", lambda *args, **kwargs: FakeResponse(html, url=args[0]))

    result = scraper.extract_document("https://www.cisa.gov/news-events/cybersecurity-advisories/aa26-097a")

    assert result["success"] is True
    data = result["data"]
    assert data["title"] == "AA26-097A: Example Threat Advisory"
    assert data["date"] == "July 01, 2026"
    assert data["alert_code"] == "AA26-097A"
    assert data["word_count"] > 40


class CISAScraperStub:
    def extract_document(self, entry: dict[str, Any], **_: Any) -> dict[str, Any]:
        return {
            "success": True,
            "data": {
                "url": entry["url"],
                "title": entry["title"],
                "date": entry["date"],
                "summary": "CISA advisory summary.",
                "doc_type": "Cybersecurity Advisory",
                "alert_code": "AA26-097A",
                "full_text": " ".join(["CISA advisory text"] * 45),
                "source_format": "html",
                "extraction_mode": "cisa_html",
            },
        }


def test_cisa_pipeline_record_builds_document():
    entry = {
        "url": "https://www.cisa.gov/news-events/cybersecurity-advisories/aa26-097a",
        "title": "AA26-097A: Example Threat Advisory",
        "date": "July 01, 2026",
        "doc_type": "Cybersecurity Advisory",
        "alert_code": "AA26-097A",
        "listing_page": cisa.CISA_CYBERSECURITY_ADVISORIES_URL,
    }

    record = pipeline._extract_record(
        connector="cisa_cybersecurity_advisory",
        scraper=CISAScraperStub(),
        entry=entry,
        idx=1,
        base_url=cisa.CISA_CYBERSECURITY_ADVISORIES_URL,
    )

    assert record["metadata"]["source_kind"] == "cisa_cybersecurity_advisory"
    assert record["metadata"]["organization"] == "CISA"
    assert record["metadata"]["source_family"] == "cisa_cybersecurity_advisory"
    assert record["metadata"]["alert_code"] == "AA26-097A"
    assert "cisa" in record["metadata"]["tags"]
