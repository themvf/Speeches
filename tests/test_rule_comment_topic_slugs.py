"""SEC rule-comment documents must carry a canonical topic slug.

filterCanonicalTopicMappedDocuments (apps/web/lib/intel-topic-matching.ts)
resolves a stored tag only when its normalized identity exactly equals an
active rule's topic_key or label. Descriptive tags like "public-comment" match
nothing, so a document carrying only those is silently dropped from the feed
after being ingested successfully. These tests pin the slugs that make rule
comments reachable.
"""

import run_connector_extraction_pipeline as pipeline


def _identity(value: str) -> str:
    """Mirror of identity() in filterCanonicalTopicMappedDocuments: lowercase,
    non-alphanumerics collapsed to single spaces."""
    out = []
    for ch in value.lower():
        out.append(ch if ch.isalnum() else " ")
    return " ".join("".join(out).split())


def test_slugs_normalize_onto_real_topic_keys():
    assert _identity(pipeline._TOPIC_SLUG_SECURITIES_REGULATION) == _identity("SECURITIES_REGULATION")
    assert _identity(pipeline._TOPIC_SLUG_CAPITAL_FORMATION) == _identity("CAPITAL_FORMATION")


def test_descriptive_tags_alone_would_not_resolve():
    """Documents the bug being fixed: none of the original tags match a topic."""
    for tag in ("sec", "rulemaking", "public-comment", "rule-release", "file-s7-2026-20"):
        assert _identity(tag) not in {_identity("SECURITIES_REGULATION"), _identity("CAPITAL_FORMATION")}


def test_capital_formation_slug_added_for_capital_formation_rulemaking():
    slugs = pipeline._capital_formation_slug_for(
        "Comment on File No. S7-2026-17",
        "",
        "Re: File No. S7-2026-17, Registered Offering Reform. We support the "
        "Commission's proposal to modernize the shelf registration process.",
    )
    assert slugs == [pipeline._TOPIC_SLUG_CAPITAL_FORMATION]


def test_capital_formation_slug_absent_for_unrelated_rulemaking():
    slugs = pipeline._capital_formation_slug_for(
        "Comment on File No. S7-2026-21",
        "",
        "Re: File No. S7-2026-21, Security-Based Swap Execution. Our comments "
        "concern clearing agency margin methodology and settlement cycles.",
    )
    assert slugs == []


def test_capital_formation_scan_is_bounded():
    """The vocabulary hit must fall inside the scanned prefix; a mention buried
    far into a long comment letter should not tag the whole document."""
    buried = ("filler sentence. " * 400) + " this concerns a non-traded REIT offering"
    assert len(buried) > pipeline._CAPITAL_FORMATION_SCAN_CHARS
    assert pipeline._capital_formation_slug_for("Comment", "", buried) == []


def test_empty_input_is_safe():
    assert pipeline._capital_formation_slug_for("", "", "") == []
    assert pipeline._capital_formation_slug_for(None, None, None) == []
