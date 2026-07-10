import run_financial_news_pipeline as core


DOC_TEXT = (
    "The Commission today charged Acme Advisers with fraud. "
    "Investors were told their funds were fully insured, which was false. "
    "The scheme raised over forty million dollars from retail investors."
)


def _enrich(spans):
    return core._normalize_enrichment_payload(
        {
            "summary": "s",
            "tags": ["enforcement"],
            "keywords": ["fraud"],
            "evidence_spans": spans,
        },
        doc={"full_text": DOC_TEXT},
    )


def test_verbatim_snippet_is_marked_verified():
    out = _enrich([{"claim": "Investors misled", "snippet": "their funds were fully insured, which was false"}])
    assert out["evidence_spans"][0]["verified"] is True


def test_hallucinated_snippet_is_marked_unverified():
    out = _enrich([{"claim": "Fabricated", "snippet": "the CEO personally apologized to every victim in writing"}])
    assert out["evidence_spans"][0]["verified"] is False


def test_smart_quotes_and_whitespace_differences_still_verify():
    out = _enrich([{"claim": "c", "snippet": "their   funds  were fully insured,\nwhich was false"}])
    assert out["evidence_spans"][0]["verified"] is True


def test_reward_discounts_hallucinated_evidence():
    verified = _enrich([
        {"claim": "a", "snippet": "The Commission today charged Acme Advisers with fraud"},
        {"claim": "b", "snippet": "their funds were fully insured, which was false"},
        {"claim": "c", "snippet": "raised over forty million dollars from retail investors"},
    ])
    hallucinated = _enrich([
        {"claim": "a", "snippet": "the moon landing was staged in a studio somewhere"},
        {"claim": "b", "snippet": "aliens secretly control the federal reserve board"},
        {"claim": "c", "snippet": "this snippet appears nowhere in the source document"},
    ])
    r_verified = core._compute_reward(verified, "pending", status="enriched")
    r_hallucinated = core._compute_reward(hallucinated, "pending", status="enriched")
    assert r_verified["components"]["evidence_quality"] == 1.0
    assert r_hallucinated["components"]["evidence_quality"] == 0.0
    assert r_verified["score"] > r_hallucinated["score"]


def test_reward_backward_compatible_with_unflagged_spans():
    # Spans stored before verification existed (no `verified` key) count as verified.
    legacy = {"tags": ["t"], "keywords": ["k"], "evidence_spans": [{"claim": "a", "snippet": "x"}, {"claim": "b", "snippet": "y"}, {"claim": "c", "snippet": "z"}]}
    assert core._evidence_quality_score(legacy["evidence_spans"]) == 1.0
