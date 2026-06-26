import json

from substack_public_scraper import SubstackPublicScraper


def test_substack_uses_residential_proxy_fallback(monkeypatch):
    monkeypatch.delenv("SUBSTACK_PROXY_URL", raising=False)
    monkeypatch.delenv("APIFY_PROXY_URL", raising=False)
    monkeypatch.setenv("RESIDENTIAL_PROXY_URL", "http://residential.example:8000")

    scraper = SubstackPublicScraper(min_delay_seconds=0)

    assert scraper.proxy_url == "http://residential.example:8000"


def test_substack_normalizes_residential_proxy_shorthand(monkeypatch):
    monkeypatch.delenv("SUBSTACK_PROXY_URL", raising=False)
    monkeypatch.delenv("APIFY_PROXY_URL", raising=False)
    monkeypatch.setenv("RESIDENTIAL_PROXY_URL", "proxy.example:8000:user:pa:ss")

    scraper = SubstackPublicScraper(min_delay_seconds=0)

    assert scraper.proxy_url == "http://user:pa%3Ass@proxy.example:8000"
    assert scraper.proxy_config_error == ""


def test_substack_rejects_invalid_residential_proxy_port(monkeypatch):
    monkeypatch.delenv("SUBSTACK_PROXY_URL", raising=False)
    monkeypatch.delenv("APIFY_PROXY_URL", raising=False)
    monkeypatch.setenv("RESIDENTIAL_PROXY_URL", "proxy.example:notaport:user:pass")

    scraper = SubstackPublicScraper(min_delay_seconds=0)

    assert scraper.proxy_url == ""
    assert "numeric port" in scraper.proxy_config_error


def _post(post_id, title, slug, keyword_text=""):
    return {
        "id": post_id,
        "publication_id": 10,
        "title": title,
        "slug": slug,
        "canonical_url": f"https://example.substack.com/p/{slug}",
        "post_date": "2026-06-19T12:00:00Z",
        "audience": "everyone",
        "subtitle": keyword_text,
        "truncated_body_text": keyword_text,
        "wordcount": 500,
        "publishedBylines": [{"name": "Test Author"}],
        "postTags": [{"name": "markets"}],
    }


def test_discovery_paginates_and_deduplicates_across_keywords(monkeypatch):
    scraper = SubstackPublicScraper(min_delay_seconds=0)
    payloads = {
        ("securities", 0): {
            "results": [_post(1, "Securities markets", "securities-markets")],
            "publications": [{"id": 10, "name": "Market Notes"}],
            "more": False,
            "feedSessionId": "session-a",
        },
        ("financial industry", 0): {
            "results": [
                _post(1, "Securities markets", "securities-markets"),
                _post(2, "Bank capital", "bank-capital"),
            ],
            "publications": [{"id": 10, "name": "Market Notes"}],
            "more": False,
            "feedSessionId": "session-b",
        },
    }

    def fake_get_json(_url, *, params=None):
        return payloads[(params["query"], params["page"])]

    monkeypatch.setattr(scraper, "_get_json", fake_get_json)
    results = scraper.discover_documents(
        keywords=["securities", "financial industry"], max_pages=1
    )

    assert len(results) == 2
    first = next(item for item in results if item["substack_post_id"] == 1)
    assert first["matched_keywords"] == ["securities", "financial industry"]
    assert first["publication_name"] == "Market Notes"


def test_feed_discovery_parses_curated_rss(monkeypatch):
    scraper = SubstackPublicScraper(min_delay_seconds=0)
    rss = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <channel>
        <title>Bank Reg Blog</title>
        <item>
          <title>Bank capital proposal</title>
          <link>https://bankregblog.substack.com/p/bank-capital-proposal</link>
          <guid>feed-post-1</guid>
          <pubDate>Fri, 19 Jun 2026 12:00:00 GMT</pubDate>
          <description><![CDATA[<p>Bank capital requirements changed materially.</p>]]></description>
          <dc:creator>Policy Author</dc:creator>
          <category>banking</category>
        </item>
      </channel>
    </rss>
    """

    monkeypatch.setattr(scraper, "_get_text", lambda _url: rss)
    results = scraper.discover_feed_documents(
        feeds=[
            {
                "label": "Bank Reg Blog",
                "feed_url": "https://bankregblog.substack.com/feed",
                "tags_csv": "bank-reg-blog,bank-regulation",
            }
        ],
        max_items_per_feed=10,
    )

    assert len(results) == 1
    assert results[0]["slug"] == "bank-capital-proposal"
    assert results[0]["publication_name"] == "Bank Reg Blog"
    assert results[0]["authors"] == ["Policy Author"]
    assert results[0]["feed_tags"] == ["bank-reg-blog", "bank-regulation"]
    assert results[0]["discovery_mode"] == "feed"


def test_discovery_merges_feed_and_search_hits_by_url(monkeypatch):
    scraper = SubstackPublicScraper(min_delay_seconds=0)
    rss = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Market Notes</title>
        <item>
          <title>Securities markets</title>
          <link>https://example.substack.com/p/securities-markets</link>
          <guid>https://example.substack.com/p/securities-markets</guid>
          <pubDate>Fri, 19 Jun 2026 12:00:00 GMT</pubDate>
          <description>Market structure update.</description>
        </item>
      </channel>
    </rss>
    """

    def fake_get_json(_url, *, params=None):
        return {
            "results": [_post(1, "Securities markets", "securities-markets")],
            "publications": [{"id": 10, "name": "Search Publication"}],
            "more": False,
        }

    monkeypatch.setattr(scraper, "_get_json", fake_get_json)
    monkeypatch.setattr(scraper, "_get_text", lambda _url: rss)
    results = scraper.discover_documents(
        keywords=["securities"],
        max_pages=1,
        include_feeds=True,
        feeds=[
            {"label": "Market Notes", "feed_url": "https://example.substack.com/feed"}
        ],
    )

    assert len(results) == 1
    assert results[0]["matched_keywords"] == ["securities"]
    assert results[0]["feed_url"] == "https://example.substack.com/feed"
    assert results[0]["discovery_mode"] == "search+feed"
    assert results[0]["discovery_modes"] == ["search", "feed"]


def test_extract_document_uses_public_body_html(monkeypatch):
    scraper = SubstackPublicScraper(min_delay_seconds=0)

    def fake_get_json(_url, *, params=None):
        return {
            **_post(1, "Institutional markets", "institutional-markets"),
            "body_html": "<h2>Market update</h2><p>Bank capital requirements changed materially today.</p>",
            "free_unlock_required": False,
        }

    monkeypatch.setattr(scraper, "_get_json", fake_get_json)
    result = scraper.extract_document(
        {
            "url": "https://example.substack.com/p/institutional-markets",
            "slug": "institutional-markets",
            "publication_name": "Market Notes",
        }
    )

    assert result["success"] is True
    assert "Bank capital requirements" in result["data"]["full_text"]
    assert result["data"]["access_limited"] is False


def test_extract_document_derives_slug_from_feed_url(monkeypatch):
    scraper = SubstackPublicScraper(min_delay_seconds=0)

    def fake_get_json(url, *, params=None):
        assert url == "https://example.substack.com/api/v1/posts/institutional-markets"
        return {
            **_post(1, "Institutional markets", "institutional-markets"),
            "body_html": "<p>Market structure and capital formation changed materially today.</p>",
            "free_unlock_required": False,
        }

    monkeypatch.setattr(scraper, "_get_json", fake_get_json)
    result = scraper.extract_document(
        {
            "url": "https://example.substack.com/p/institutional-markets",
            "publication_name": "Market Notes",
        }
    )

    assert result["success"] is True
    assert "capital formation" in result["data"]["full_text"]


class _FakeResponse:
    def __init__(self, payload):
        self.output_text = json.dumps(payload)


class _FakeResponses:
    def create(self, *, model, instructions, input):
        candidates = json.loads(input)["candidates"]
        decisions = []
        for candidate in candidates:
            personal = "budget" in candidate["title"].lower()
            decisions.append(
                {
                    "post_id": candidate["post_id"],
                    "classification": "personal_finance"
                    if personal
                    else "institutional_finance",
                    "confidence": 0.96,
                    "reason": "Consumer budgeting advice."
                    if personal
                    else "Institutional markets coverage.",
                }
            )
        return _FakeResponse({"decisions": decisions})


class _FakeClient:
    responses = _FakeResponses()


def test_openai_relevance_filter_excludes_high_confidence_personal_finance():
    scraper = SubstackPublicScraper(min_delay_seconds=0)
    entries = [
        {
            "substack_post_id": 1,
            "url": "https://example.substack.com/p/market-structure",
            "title": "SEC market structure proposal",
            "matched_keywords": ["securities"],
        },
        {
            "substack_post_id": 2,
            "url": "https://example.substack.com/p/family-budget",
            "title": "Build your family budget",
            "matched_keywords": ["financial industry"],
        },
    ]

    included, excluded = scraper.filter_institutional_finance(
        entries, client=_FakeClient()
    )

    assert [item["substack_post_id"] for item in included] == [1]
    assert [item["substack_post_id"] for item in excluded] == [2]
    assert excluded[0]["relevance_reason"] == "Consumer budgeting advice."
