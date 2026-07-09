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


class TradeMediaScraperStub:
    def extract_document(self, url, **kwargs):
        return {
            "success": True,
            "data": {
                "url": url,
                "title": kwargs.get("fallback_title", "Trade Article"),
                "date": kwargs.get("fallback_date", "June 27, 2026"),
                "source_name": kwargs.get("fallback_source_name", "JD Supra"),
                "description": kwargs.get("fallback_description", ""),
                "full_text": " ".join(["trade media regulatory analysis"] * 45),
                "word_count": 180,
                "source_format": "html",
            },
        }


class WSJScraperStub:
    def extract_document(self, url, **kwargs):
        return {
            "success": True,
            "data": {
                "url": url,
                "title": kwargs.get("fallback_title", "WSJ Article"),
                "date": kwargs.get("fallback_date", "June 27, 2026"),
                "author": kwargs.get("fallback_author", "WSJ Staff"),
                "full_text": " ".join(["dow jones markets article"] * 45),
                "source_format": "html",
                "extraction_mode": "rss_description",
            },
        }


class YouTubeScraperStub:
    def extract_document(self, url, **kwargs):
        return {
            "data": {
                "url": url,
                "video_id": "abc123def45",
                "title": kwargs.get("fallback_title", "Example Video"),
                "date": kwargs.get("fallback_date", "June 27, 2026"),
                "published_at": kwargs.get("fallback_date", "2026-06-27T12:00:00Z"),
                "full_text": " ".join(["youtube transcript text"] * 30),
            }
        }


class HedgeFundLetterScraperStub:
    def extract_document(self, entry):
        return {
            "success": True,
            "data": {
                "url": entry.get("url", ""),
                "title": entry.get("title", "Example Fund Letter"),
                "date": entry.get("date", "March 28, 2026"),
                "summary": entry.get("summary", "Investor letter summary."),
                "full_text": " ".join(["fund letter market commentary portfolio risk"] * 45),
                "source_format": "pdf",
                "extraction_mode": "pdf",
                "fund_name": entry.get("fund_name", "Example Fund"),
                "source_label": entry.get("source_label", "Fiscal.ai Fund Letters"),
                "source_key": entry.get("source_key", "fiscal_ai_fund_letters"),
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


def test_partial_item_failures_are_nonfatal_below_failure_rate_threshold():
    # 2/60 failed (~3%): a small number of item-level failures shouldn't fail
    # an otherwise-mostly-successful run for any connector.
    summary = {"failed_count": 2, "failed": [{"title": "blocked"}], "processed_count": 58}

    assert pipeline._should_fail_for_item_failures("substack_public_article", summary) is False
    assert pipeline._should_fail_for_item_failures("jdsupra_article", summary) is False


def test_item_failures_are_fatal_above_failure_rate_threshold():
    # 40/60 failed (~67%): majority failure should still fail the run even
    # though some items were processed.
    summary = {"failed_count": 40, "failed": [{"title": "blocked"}], "processed_count": 20}

    assert pipeline._should_fail_for_item_failures("jdsupra_article", summary) is True


def test_item_failures_are_fatal_when_no_records_processed():
    summary = {"failed_count": 2, "failed": [{"title": "blocked"}], "processed_count": 0}

    assert pipeline._should_fail_for_item_failures("substack_public_article", summary) is True
    assert pipeline._should_fail_for_item_failures("jdsupra_article", summary) is True


def test_crs_status_treats_equivalent_date_formats_as_existing():
    entry = {
        "url": "https://www.congress.gov/crs-product/R48978",
        "title": "Example CRS Report",
        "date": "June 9, 2026",
        "doc_type": "Report",
        "authors": "",
    }
    existing = {
        "title": "Example CRS Report",
        "published_date": "June 09, 2026",
        "doc_type": "Report",
        "speaker": "Congressional Research Service",
    }

    assert pipeline._status_for_entry("congress_crs_product", entry, existing, set()) == "existing"


def test_crs_status_detects_actual_date_change():
    entry = {
        "url": "https://www.congress.gov/crs-product/R48978",
        "title": "Example CRS Report",
        "date": "June 10, 2026",
        "doc_type": "Report",
        "authors": "",
    }
    existing = {
        "title": "Example CRS Report",
        "published_date": "June 09, 2026",
        "doc_type": "Report",
        "speaker": "Congressional Research Service",
    }

    assert pipeline._status_for_entry("congress_crs_product", entry, existing, set()) == "update_available"


def test_gap_connectors_are_registered_with_defaults():
    for connector in [
        "federal_reserve_speech_testimony",
        "treasury_statement_remark",
        "treasury_press_release",
        "treasury_featured_story",
        "sec_tm_faq",
        "jdsupra_article",
        "investmentnews_article",
        "citywire_article",
        "therecord_media_article",
        "wired_article",
        "tripwire_article",
        "akamai_blog_article",
        "ritholtz_article",
        "ft_portfolios_market_commentary",
        "liberty_street_economics_article",
        "wealth_of_common_sense_article",
        "wsj_dow_jones",
        "reddit_post",
        "hedge_fund_letter",
    ]:
        assert connector in pipeline.SUPPORTED_CONNECTORS
        assert pipeline._default_base_url(connector)


def test_youtube_connectors_are_registered():
    assert "sec_youtube_video" in pipeline.SUPPORTED_CONNECTORS
    assert "youtube_video" in pipeline.SUPPORTED_CONNECTORS
    assert pipeline._default_base_url("sec_youtube_video")
    assert pipeline._default_base_url("youtube_video") == ""


def test_trade_media_extract_record_builds_document():
    entry = {
        "url": "https://www.jdsupra.com/legalnews/example-123/",
        "title": "Example JD Supra Article",
        "date": "June 27, 2026",
        "description": "A regulatory analysis summary.",
    }

    record = pipeline._extract_record(
        connector="jdsupra_article",
        scraper=TradeMediaScraperStub(),
        entry=entry,
        idx=1,
        base_url="https://www.jdsupra.com/",
    )

    assert record["metadata"]["source_kind"] == "jdsupra_article"
    assert record["metadata"]["organization"] == "JD Supra"
    assert record["metadata"]["source_name"] == "JD Supra"
    assert "trade media regulatory analysis" in record["content"]["full_text"]


def test_wsj_dow_jones_extract_record_builds_document():
    entry = {
        "url": "https://www.wsj.com/articles/example",
        "title": "Example WSJ Article",
        "date": "June 27, 2026",
        "description": "RSS summary",
        "author": "WSJ Staff",
    }

    record = pipeline._extract_record(
        connector="wsj_dow_jones",
        scraper=WSJScraperStub(),
        entry=entry,
        idx=1,
        base_url="https://feeds.content.dowjones.io/public/rss/WSJcomUSBusinessNews",
    )

    assert record["metadata"]["source_kind"] == "wsj_dow_jones"
    assert record["metadata"]["organization"] == "WSJ / Dow Jones"
    assert record["metadata"]["extraction_mode"] == "rss_description"


def test_sec_youtube_extract_record_builds_sec_document():
    entry = {
        "url": "https://www.youtube.com/watch?v=abc123def45",
        "video_id": "abc123def45",
        "title": "SEC Roundtable",
        "date": "June 27, 2026",
        "published_at": "2026-06-27T12:00:00Z",
        "channel_id": "UCSEC",
    }

    record = pipeline._extract_record(
        connector="sec_youtube_video",
        scraper=YouTubeScraperStub(),
        entry=entry,
        idx=1,
        base_url="https://www.youtube.com/user/SECViews",
    )

    assert record["metadata"]["organization"] == "SEC"
    assert record["metadata"]["source_kind"] == "sec_youtube_video"
    assert record["metadata"]["youtube_video_id"] == "abc123def45"
    assert "sec" in record["metadata"]["tags"]


def test_generic_youtube_extract_record_builds_youtube_document():
    entry = {
        "url": "https://www.youtube.com/watch?v=abc123def45",
        "video_id": "abc123def45",
        "title": "Market Structure Interview",
        "date": "June 27, 2026",
        "published_at": "2026-06-27T12:00:00Z",
        "channel_id": "UCEXAMPLE",
    }

    record = pipeline._extract_record(
        connector="youtube_video",
        scraper=YouTubeScraperStub(),
        entry=entry,
        idx=1,
        base_url="https://www.youtube.com/@example",
    )

    assert record["metadata"]["organization"] == "YouTube"
    assert record["metadata"]["source_kind"] == "youtube_video"
    assert record["metadata"]["source_family"] == "youtube_video"
    assert record["metadata"]["youtube_channel_id"] == "UCEXAMPLE"


def test_invalid_wired_coupon_records_are_pruned():
    payload = {
        "documents": [
            {
                "metadata": {
                    "source_kind": "wired_article",
                    "title": "Ulta Promo Codes: Up to 50% Off in July 2026",
                    "url": "https://www.wired.com/story/ulta-promo-codes-july-2026/",
                }
            },
            {
                "metadata": {
                    "source_kind": "wired_article",
                    "title": "Ransomware Operators Target Financial Firms",
                    "url": "https://www.wired.com/story/ransomware-financial-firms-2026/",
                }
            },
            {
                "metadata": {
                    "source_kind": "jdsupra_article",
                    "title": "Coupon regulations update",
                    "url": "https://www.jdsupra.com/legalnews/example-123/",
                }
            },
        ]
    }

    removed = pipeline._remove_invalid_wired_coupon_records(payload)

    assert removed == 1
    titles = [item["metadata"]["title"] for item in payload["documents"]]
    assert "Ulta Promo Codes: Up to 50% Off in July 2026" not in titles
    assert "Ransomware Operators Target Financial Firms" in titles
    assert "Coupon regulations update" in titles


def test_hedge_fund_letter_extract_record_builds_document():
    entry = {
        "url": "https://example.com/fund-letter.pdf",
        "title": "Example Fund Q1 Letter",
        "date": "March 28, 2026",
        "summary": "Quarterly investor letter.",
        "source_label": "Fiscal.ai Fund Letters",
        "source_key": "fiscal_ai_fund_letters",
        "organization": "Fiscal.ai",
        "fund_name": "Example Fund",
    }

    record = pipeline._extract_record(
        connector="hedge_fund_letter",
        scraper=HedgeFundLetterScraperStub(),
        entry=entry,
        idx=1,
        base_url="https://fiscal.ai/fund-letters/",
    )

    metadata = record["metadata"]
    assert metadata["source_kind"] == "hedge_fund_letter"
    assert metadata["source_family"] == "hedge_fund_letter"
    assert metadata["source_name"] == "Fiscal.ai Fund Letters"
    assert metadata["fund_name"] == "Example Fund"
    assert metadata["pdf_url"] == "https://example.com/fund-letter.pdf"


def test_sec_federal_register_status_updates_generic_existing_title():
    entry = {
        "url": "https://www.federalregister.gov/documents/2026/06/29/example",
        "title": "Self-Regulatory Organizations; Notice of Filing of Proposed Rule Change",
        "date": "June 29, 2026",
    }
    existing = {
        "title": "Notice",
        "published_date": "June 29, 2026",
    }

    assert pipeline._status_for_entry("sec_federal_register", entry, existing, set()) == "update_available"
