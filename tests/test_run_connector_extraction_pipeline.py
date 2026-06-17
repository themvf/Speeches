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


class ShortDocumentScraper:
    def extract_document(self, url, **kwargs):
        return {
            "success": True,
            "data": {
                "url": url,
                "title": kwargs.get("fallback_title", "Short Document"),
                "date": kwargs.get("fallback_date", "May 23, 2026"),
                "full_text": "Short body.",
                "word_count": 2,
                "source_format": "html",
            },
        }


class ConwayLitigationScraper:
    def extract_document(self, url, **kwargs):
        text = (
            "U.S. SECURITIES AND EXCHANGE COMMISSION\n"
            "Litigation Release No. 26370 / August 11, 2025\n"
            "Securities and Exchange Commission v. Conway\n"
            "SEC Charges Texas Resident with Insider Trading\n"
            "On August 7, 2025, the Securities and Exchange Commission filed charges against "
            "Bruce Cameron Conway, a resident of Dallas, Texas, for insider trading in advance "
            "of the August 24, 2020 announcement that Cancer Genetics, Inc. would merge with a "
            "privately held biotechnology company.\n"
            "According to the SEC's complaint, Conway purchased Cancer Genetics shares in fifteen "
            "accounts belonging to him, various family members, and family-owned trusts."
        )
        return {
            "success": True,
            "data": {
                "url": url,
                "title": "Bruce Cameron Conway",
                "date": "August 11, 2025",
                "release_no": "LR-26370",
                "full_text": text,
                "word_count": len(text.split()),
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


def test_non_doj_short_text_is_retained_instead_of_failing():
    entry = {
        "url": "https://www.sec.gov/rules-regulations/staff-guidance/example-short-faq",
        "title": "Example Short SEC FAQ",
        "updated_date": "May 23, 2026",
    }

    record = pipeline._extract_record(
        connector="sec_tm_faq",
        scraper=ShortDocumentScraper(),
        entry=entry,
        idx=1,
        base_url="https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions",
    )

    assert record["metadata"]["source_kind"] == "sec_tm_faq"
    assert record["metadata"]["title"] == "Example Short SEC FAQ"
    assert record["metadata"]["word_count"] == 2
    assert record["content"]["full_text"] == "Short body."


def test_sec_litigation_record_infers_respondents_and_entities():
    entry = {
        "url": "https://www.sec.gov/enforcement-litigation/litigation-releases/lr-26370",
        "title": "Bruce Cameron Conway",
        "date": "August 11, 2025",
        "release_no": "LR-26370",
    }

    record = pipeline._extract_record(
        connector="sec_enforcement_litigation",
        scraper=ConwayLitigationScraper(),
        entry=entry,
        idx=1,
        base_url="https://www.sec.gov/enforcement-litigation/litigation-releases",
    )

    metadata = record["metadata"]
    assert "Bruce Cameron Conway" in metadata["respondents"]
    assert "Cancer Genetics, Inc." in metadata["entities"]
    assert metadata["action_type"] == "filing"
    assert "Insider Trading" in metadata["alleged_violations"]


def test_summary_with_item_failures_should_fail_process():
    assert pipeline._has_item_failures({"failed_count": 1, "failed": [{"title": "bad"}]}) is True
    assert pipeline._has_item_failures({"failed_count": 0, "failed": []}) is False
    assert pipeline._has_item_failures({"failed_count": "2"}) is True
    assert pipeline._has_item_failures({"failed": [{"title": "bad"}]}) is True
