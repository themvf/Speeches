from run_connector_extraction_pipeline import (
    _annotate_topic_matches,
    _topic_rules_to_search_terms,
    _topic_term_matches,
)


def test_topic_rules_to_search_terms_preserves_current_rule_order():
    rules = [
        {
            "topic_key": "FINANCIAL_MARKETS",
            "label": "Financial Markets",
            "keywords": "market, stock, equity, trading",
            "sort_order": 80,
        },
        {
            "topic_key": "SECURITIES_REGULATION",
            "label": "Securities Regulation",
            "keywords": "sec, securities, disclosure",
            "sort_order": 10,
        },
    ]

    terms = _topic_rules_to_search_terms(rules)

    assert terms[:4] == ["sec", "securities", "disclosure", "securities regulation"]
    assert "financial markets" in terms


def test_topic_rules_to_search_terms_can_be_balanced_and_limited():
    rules = [
        {
            "topic_key": "FINANCIAL_MARKETS",
            "label": "Financial Markets",
            "keywords": "market, stock, equity, trading",
            "sort_order": 80,
        },
        {
            "topic_key": "SECURITIES_REGULATION",
            "label": "Securities Regulation",
            "keywords": "sec, securities, disclosure",
            "sort_order": 10,
        },
        {
            "topic_key": "CRYPTO",
            "label": "Crypto",
            "keywords": "crypto, stablecoin, defi",
            "sort_order": 60,
        },
    ]

    terms = _topic_rules_to_search_terms(rules, max_terms=5)

    assert terms == ["sec", "crypto", "market", "securities", "stablecoin"]


def test_annotate_topic_matches_uses_existing_keywords_and_feed_tags():
    entry = {
        "title": "SEC market structure proposal",
        "summary": "Broker-dealer trading and disclosure update.",
        "matched_keywords": ["sec"],
        "feed_tags": ["market-structure"],
    }
    rules = [
        {
            "topic_key": "SECURITIES_REGULATION",
            "label": "Securities Regulation",
            "keywords": "sec, securities, disclosure, exchange",
            "sort_order": 10,
        },
        {
            "topic_key": "FINANCIAL_MARKETS",
            "label": "Financial Markets",
            "keywords": "market, stock, equity, trading",
            "sort_order": 80,
        },
    ]

    _annotate_topic_matches(entry, rules)

    assert entry["matched_topic_keys"] == ["SECURITIES_REGULATION", "FINANCIAL_MARKETS"]
    assert "Securities Regulation" in entry["matched_topic_labels"]
    assert "trading" in entry["matched_topic_keywords"]


def test_topic_term_matches_uses_word_boundaries_for_short_terms():
    # Short acronyms must not match as mid-word substrings.
    assert not _topic_term_matches("ai", "please check your email inbox")
    assert not _topic_term_matches("sec", "in the second quarter")
    assert not _topic_term_matches("aml", "the enamel coating")
    # But should still match as standalone words.
    assert _topic_term_matches("ai", "new ai governance rules")
    assert _topic_term_matches("sec", "the sec proposed a rule")


def test_topic_term_matches_multiword_with_flexible_separators():
    assert _topic_term_matches("money laundering", "anti money laundering program")
    assert _topic_term_matches("market structure", "equity market-structure reform")
    assert _topic_term_matches("digital asset", "a new digital_asset framework")


def test_annotate_topic_matches_does_not_over_match_short_acronym():
    entry = {"title": "Quarterly email newsletter for the second half", "summary": ""}
    rules = [
        {"topic_key": "AI_TECH", "label": "AI & Tech", "keywords": "ai", "sort_order": 50},
        {"topic_key": "SECURITIES_REGULATION", "label": "Securities Regulation", "keywords": "sec", "sort_order": 10},
    ]
    _annotate_topic_matches(entry, rules)
    # "ai" inside "email" and "sec" inside "second" must not produce matches.
    assert "matched_topic_keys" not in entry
