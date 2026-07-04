from senate_committee_scraper import SENATE_COMMITTEE_DEFAULT_URL, SenateCommitteeScraper
import run_connector_extraction_pipeline as pipeline


def test_discovers_hsgac_data_url_and_extracts_detail_page(monkeypatch):
    listing_url = "https://www.hsgac.senate.gov/media/"
    detail_url = "https://www.hsgac.senate.gov/media/reps/chairman-paul-announces-hearing/"
    listing_html = f"""
    <html><body>
      <div class="jet-engine-listing-overlay-wrap" data-url="{detail_url}">
        <h5>Chairman Paul Announces Oversight Hearing</h5>
        <span>May 19, 2026</span>
      </div>
    </body></html>
    """
    detail_html = f"""
    <html>
      <head>
        <meta property="og:title" content="Chairman Paul Announces Oversight Hearing" />
        <meta property="article:published_time" content="2026-05-19T13:00:00Z" />
      </head>
      <body>
        <main>
          <p>WASHINGTON, D.C. - The committee announced an oversight hearing focused on federal agency accountability.</p>
          <p>The hearing will examine policy implementation, congressional oversight, operational risk, and public accountability.</p>
          <p>Members will receive testimony from agency officials and outside witnesses during the public session.</p>
        </main>
      </body>
    </html>
    """

    scraper = SenateCommitteeScraper(sleep_seconds=0)

    def fake_fetch(url):
        if url == detail_url:
            return detail_html, 200, detail_url
        return listing_html, 200, listing_url

    monkeypatch.setattr(scraper, "_fetch_text", fake_fetch)

    docs = scraper.discover_documents(base_url=listing_url, max_pages=1)
    assert len(docs) == 1
    assert docs[0]["url"] == detail_url
    assert docs[0]["source_key"] == "senate_hsgac"
    assert docs[0]["title"] == "Chairman Paul Announces Oversight Hearing"
    assert docs[0]["date"] == "May 19, 2026"

    extracted = scraper.extract_document(docs[0])
    assert extracted["success"] is True
    data = extracted["data"]
    assert data["title"] == "Chairman Paul Announces Oversight Hearing"
    assert data["date"] == "May 19, 2026"
    assert "federal agency accountability" in data["full_text"]


def test_runner_supports_senate_committee_site_connector():
    assert "senate_committee_site" in pipeline.SUPPORTED_CONNECTORS
    assert pipeline._default_base_url("senate_committee_site") == SENATE_COMMITTEE_DEFAULT_URL


def test_senate_committee_extract_record_builds_document():
    entry = {
        "url": "https://www.banking.senate.gov/newsroom/minority/example-release",
        "title": "Banking Committee Requests Agency Update",
        "date": "July 02, 2026",
        "summary": "The committee sent a letter requesting an update from a federal agency.",
        "source_key": "senate_banking",
        "source_label": "Senate Banking Committee",
        "organization": "Senate Committee on Banking, Housing, and Urban Affairs",
        "doc_type": "Press Release",
        "tags_csv": "senate,congress,committee,banking,press-release",
        "listing_page": "https://www.banking.senate.gov/newsroom",
    }

    class SenateScraperStub:
        def extract_document(self, incoming):
            assert incoming is entry
            return {
                "success": True,
                "data": {
                    "url": entry["url"],
                    "title": entry["title"],
                    "date": entry["date"],
                    "summary": entry["summary"],
                    "full_text": (
                        "The Senate Banking Committee requested a detailed agency update "
                        "on supervision, consumer protection, market integrity, and policy "
                        "implementation. The release identifies the committee request, the "
                        "agency response deadline, and the oversight context for Congress."
                    ),
                    "extraction_mode": "senate_committee_html",
                },
            }

    record = pipeline._extract_record(
        connector="senate_committee_site",
        scraper=SenateScraperStub(),
        entry=entry,
        idx=1,
        base_url=SENATE_COMMITTEE_DEFAULT_URL,
    )

    metadata = record["metadata"]
    assert metadata["source_kind"] == "senate_committee_site"
    assert metadata["source_family"] == "senate_committee_site"
    assert metadata["source_name"] == "Senate Banking Committee"
    assert metadata["source_key"] == "senate_banking"
    assert metadata["published_date"] == "July 02, 2026"
    assert "Senate Banking Committee requested" in record["content"]["full_text"]

