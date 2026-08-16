"""Python side of the shared CAPITAL_FORMATION vocabulary.

Fixtures here are mirrored in apps/web/lib/capital-formation-vocabulary.test.ts.
Change one, change the other - that pairing is what keeps the TS and Python
readers from drifting the way the seven hand-maintained copies did.
"""

from pathlib import Path

import capital_formation_vocabulary as vocab
import neon_feeds

MIRRORED_SORT_ORDER = 12
MIRRORED_TOPIC_KEY = "CAPITAL_FORMATION"
MIRRORED_FOCUS_AREA_IDS = [
    "capital_public_offerings",
    "capital_private_capital",
    "capital_direct_participation",
    "capital_debt_financing",
    "capital_strategic_transactions",
    "capital_access_policy",
]
MIRRORED_REQUIRED_KEYWORDS = [
    "capital formation",
    "Rule 506(b)",
    "Rule 506(c)",
    "Reg CF",
    "non-traded REIT",
    "direct participation program",
    "Delaware statutory trust",
    "unregistered broker",
    "blue sky preemption",
    "no-action letter",
]


def test_real_config_loads_from_disk():
    """A packaging regression that hides the JSON must fail CI loudly rather
    than silently degrading to the fallback keyword list."""
    assert vocab.VOCABULARY_PATH.exists(), f"missing config at {vocab.VOCABULARY_PATH}"
    assert vocab.topic_key() == MIRRORED_TOPIC_KEY
    assert vocab.sort_order() == MIRRORED_SORT_ORDER
    assert vocab.focus_area_ids() == MIRRORED_FOCUS_AREA_IDS


def test_required_keywords_present():
    keywords = vocab.keywords()
    for keyword in MIRRORED_REQUIRED_KEYWORDS:
        assert keyword in keywords, f"missing keyword: {keyword}"


def test_keywords_csv_is_lowercase_and_comma_separated():
    csv = vocab.keywords_csv()
    parts = [part.strip() for part in csv.split(",")]
    assert csv == csv.lower()
    assert "capital formation" in parts
    # parseKeywords() in intel-topic-matching.ts splits on commas/newlines, so
    # a keyword containing a comma would silently become two broken keywords.
    for keyword in vocab.keywords():
        assert "," not in keyword, f"keyword contains a comma: {keyword}"


def test_filing_only_form_types_stay_out_of_keywords():
    lowered = [k.lower() for k in vocab.keywords()]
    for form in ("form 1-k", "form 1-sa", "form 1-u", "form c-ar", "form c-u"):
        assert form not in lowered, f"filing-only form type leaked into keywords: {form}"


def test_neon_feeds_seed_reads_the_shared_config():
    """neon_feeds and the TS side both seed rss_topic_rules; if this rule stops
    tracking the shared config, the live keywords depend on which writer got
    there first - the exact bug this consolidation removed."""
    rule = next(r for r in neon_feeds.DEFAULT_TOPIC_RULES if r["topic_key"] == MIRRORED_TOPIC_KEY)
    assert rule["sort_order"] == MIRRORED_SORT_ORDER
    assert rule["label"] == vocab.label()
    assert rule["keywords"] == vocab.keywords_csv()


def test_missing_config_degrades_to_fallback_without_raising(monkeypatch):
    monkeypatch.setattr(vocab, "_payload", None)
    monkeypatch.setattr(vocab, "VOCABULARY_PATH", Path("does-not-exist-capital-formation.json"))
    try:
        keywords = vocab.keywords()
        assert "capital formation" in keywords
        assert vocab.sort_order() == MIRRORED_SORT_ORDER
        assert vocab.topic_key() == MIRRORED_TOPIC_KEY
    finally:
        monkeypatch.setattr(vocab, "_payload", None)
