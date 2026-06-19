import json

from substack_public_scraper import SubstackPublicScraper


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
    results = scraper.discover_documents(keywords=["securities", "financial industry"], max_pages=1)

    assert len(results) == 2
    first = next(item for item in results if item["substack_post_id"] == 1)
    assert first["matched_keywords"] == ["securities", "financial industry"]
    assert first["publication_name"] == "Market Notes"


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
                    "classification": "personal_finance" if personal else "institutional_finance",
                    "confidence": 0.96,
                    "reason": "Consumer budgeting advice." if personal else "Institutional markets coverage.",
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

    included, excluded = scraper.filter_institutional_finance(entries, client=_FakeClient())

    assert [item["substack_post_id"] for item in included] == [1]
    assert [item["substack_post_id"] for item in excluded] == [2]
    assert excluded[0]["relevance_reason"] == "Consumer budgeting advice."
