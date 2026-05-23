import run_connector_extraction_pipeline as pipeline


class ShortDOJScraper:
    def extract_document(self, url, fallback_title="", fallback_date="", fallback_office=""):
        return {
            "success": True,
            "data": {
                "url": url,
                "title": fallback_title,
                "date": fallback_date,
                "office": fallback_office,
                "updated_date": "",
                "full_text": "Short body.",
                "word_count": 2,
                "source_format": "html",
            },
        }


def test_doj_short_text_is_retained_as_metadata_fallback():
    entry = {
        "url": "https://www.justice.gov/usao-test/pr/example-short-release",
        "title": "Example Short DOJ Press Release",
        "date": "May 23, 2026",
        "office": "U.S. Attorney's Office Test District",
    }

    record = pipeline._extract_record(
        connector="doj_usao_press_release",
        scraper=ShortDOJScraper(),
        entry=entry,
        idx=1,
        base_url="https://www.justice.gov/usao/pressreleases",
    )

    metadata = record["metadata"]
    content = record["content"]

    assert metadata["source_kind"] == "doj_usao_press_release"
    assert metadata["extraction_mode"] == "metadata_fallback"
    assert metadata["extraction_warnings"] == ["body_text_too_short"]
    assert metadata["body_word_count"] == 2
    assert "Example Short DOJ Press Release" in content["full_text"]
    assert "metadata-backed record is retained" in content["full_text"]
