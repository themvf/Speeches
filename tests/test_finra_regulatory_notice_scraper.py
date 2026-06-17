from finra_regulatory_notice_scraper import FINRARegulatoryNoticeScraper, _date_to_display


class DummyResponse:
    def __init__(self, text, url="https://www.finra.org/rules-guidance/notices/26-12"):
        self.text = text
        self.url = url


class StaticFINRAScraper(FINRARegulatoryNoticeScraper):
    def __init__(self, html):
        super().__init__(min_delay_seconds=0)
        self.html = html

    def _fetch(self, url, timeout=60):
        return DummyResponse(self.html, url=url)


def test_relative_finra_date_is_not_preserved_as_published_date():
    assert _date_to_display("20 hours ago") == ""


def test_finra_extract_prefers_actual_page_published_date_over_relative_fallback():
    html = """
    <html>
      <body>
        <article>
          <h1>Regulatory Notice 26-12</h1>
          <div class="field--name-field-core-official-dt">Published Date: Tuesday, June 09, 2026</div>
          <div class="field--name-field-tab-content">
            <p>Guidance Regarding the Application of FINRA Rules in Relation to the SEC No-Action Letter.</p>
            <p>This notice provides member firms with guidance.</p>
          </div>
        </article>
      </body>
    </html>
    """
    result = StaticFINRAScraper(html).extract_document(
        "https://www.finra.org/rules-guidance/notices/26-12",
        fallback_date="20 hours ago",
    )

    assert result["success"] is True
    assert result["data"]["date"] == "June 09, 2026"
    assert "Published Date: June 09, 2026" in result["data"]["full_text"]
