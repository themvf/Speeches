from datetime import UTC, datetime, timedelta

import trend_aggregation as trends


def _iso_days_ago(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).date().isoformat()


def _entry(doc_id: str, status: str, tag: str, days_ago: int) -> dict:
    return {
        "doc_id": doc_id,
        "status": status,
        "date": _iso_days_ago(days_ago),
        "title": f"Doc {doc_id}",
        "url": f"https://example.com/{doc_id}",
        "enrichment": {"tags": [tag], "summary": "Summary text."},
    }


def _build(state):
    return trends.build_trends(
        enrichment_state=state,
        custom_docs=[],
        sec_speeches=[],
        client=None,
        min_mentions=3,
    )


def test_reviewed_entries_are_counted_in_trends():
    # 3 enriched + 2 reviewed, all tagged "bitcoin" (maps to crypto taxonomy).
    entries = {}
    for i in range(3):
        entries[f"e{i}"] = _entry(f"e{i}", "enriched", "bitcoin", days_ago=5 + i)
    for i in range(2):
        entries[f"r{i}"] = _entry(f"r{i}", "reviewed", "bitcoin", days_ago=5 + i)

    payload = _build({"entries": entries})
    crypto = next((t for t in payload["trends"] if t["id"] == "crypto-digital-assets"), None)
    assert crypto is not None, "crypto trend should exist once reviewed docs count"
    # All 5 (3 enriched + 2 reviewed) should be counted, clearing min_mentions=3.
    assert crypto["total_mentions"] == 5


def test_reviewed_only_cluster_still_produces_a_trend():
    # Regression: before the fix, a cluster of only-reviewed docs vanished.
    entries = {f"r{i}": _entry(f"r{i}", "reviewed", "bitcoin", days_ago=5 + i) for i in range(4)}
    payload = _build({"entries": entries})
    assert any(t["id"] == "crypto-digital-assets" for t in payload["trends"])


def test_fallback_enriched_is_not_counted():
    entries = {f"f{i}": _entry(f"f{i}", "fallback_enriched", "bitcoin", days_ago=5 + i) for i in range(5)}
    payload = _build({"entries": entries})
    assert not any(t["id"] == "crypto-digital-assets" for t in payload["trends"])
