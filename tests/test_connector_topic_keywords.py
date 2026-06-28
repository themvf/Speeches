from run_connector_extraction_pipeline import _annotate_topic_matches, _topic_rules_to_search_terms


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
