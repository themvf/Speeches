"""Tests for kpi_tier_c_extract (SEC-13): evidence verification, extraction
parsing/validation, review-decision preservation across re-runs, exhibit
selection, and Tier C config integrity. Network and LLM are never touched."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import kpi_tier_c_extract as tc
from kpi_config import TIER_C_KPIS

SOURCE_TEXT = (
    "Family daily active people (DAP) - DAP was 3.43 billion on average for "
    "June 2026, an increase of 6% year-over-year. Revenue was $47.5 billion."
)

KPIS = [
    {"kpi_key": "family_dap", "label": "Family Daily Active People (DAP)", "unit": "count",
     "hint": "Average daily active people."},
    {"kpi_key": "other_metric", "label": "Other", "unit": "usd", "hint": "Other."},
]


def test_evidence_verified_normalizes_whitespace_and_quotes():
    assert tc.evidence_verified("DAP was 3.43   billion", SOURCE_TEXT)
    # Smart-quote/dash normalization both sides.
    assert tc.evidence_verified("people (DAP) — DAP was", SOURCE_TEXT.replace("-", "—"))
    assert not tc.evidence_verified("DAP was 9.99 billion", SOURCE_TEXT)
    assert not tc.evidence_verified("", SOURCE_TEXT)


def test_parse_extraction_validates_and_flags():
    raw = {
        "family_dap": {"value": 3.43e9, "period": "Q2 2026",
                       "evidence": "DAP was 3.43 billion on average"},
        "other_metric": {"value": 1.0, "period": "Q2 2026",
                         "evidence": "this quote is not in the text"},
    }
    out = tc.parse_extraction(raw, KPIS, SOURCE_TEXT)
    assert out["family_dap"]["evidenceVerified"] is True
    assert out["family_dap"]["status"] == "pending_review"
    assert out["family_dap"]["value"] == 3.43e9
    # Unverifiable evidence is kept but flagged.
    assert out["other_metric"]["evidenceVerified"] is False


def test_parse_extraction_drops_nulls_junk_and_missing_evidence():
    raw = {
        "family_dap": None,                                       # not stated
        "other_metric": {"value": "N/A", "evidence": "Revenue"},  # non-numeric
    }
    assert tc.parse_extraction(raw, KPIS, SOURCE_TEXT) == {}
    raw2 = {"family_dap": {"value": 5, "period": "Q2", "evidence": ""}}
    assert tc.parse_extraction(raw2, KPIS, SOURCE_TEXT) == {}


def test_merge_preserves_review_decisions_same_period():
    existing = {"kpis": {
        "family_dap": {"value": 3.43e9, "period": "Q2 2026", "status": "approved",
                       "label": "DAP", "unit": "count"},
        "other_metric": {"value": 1.0, "period": "Q2 2026", "status": "rejected"},
    }}
    fresh = {
        "family_dap": {"value": 9.9e9, "period": "Q2 2026", "status": "pending_review"},
        "other_metric": {"value": 2.0, "period": "Q2 2026", "status": "pending_review"},
    }
    merged = tc.merge_company(existing, fresh)
    # Approved AND rejected decisions both survive a same-period re-run.
    assert merged["family_dap"]["status"] == "approved"
    assert merged["family_dap"]["value"] == 3.43e9
    assert merged["other_metric"]["status"] == "rejected"


def test_merge_new_period_resets_to_pending():
    existing = {"kpis": {
        "family_dap": {"value": 3.43e9, "period": "Q2 2026", "status": "approved"},
    }}
    fresh = {"family_dap": {"value": 3.5e9, "period": "Q3 2026", "status": "pending_review"}}
    merged = tc.merge_company(existing, fresh)
    assert merged["family_dap"]["status"] == "pending_review"
    assert merged["family_dap"]["value"] == 3.5e9


def test_pick_exhibit_prefers_largest_ex99_html():
    listing = {"directory": {"item": [
        {"name": "form8k.htm", "size": "50000"},
        {"name": "ex99-1.htm", "size": "120000"},
        {"name": "ex99-2.htm", "size": "30000"},
        {"name": "index.htm", "size": "5000"},
    ]}}
    with patch.object(tc.requests, "get") as mock_get:
        mock_get.return_value.json.return_value = listing
        url, note = tc.pick_exhibit("320193", "0000320193-26-000001", "form8k.htm")
    assert url is not None and url.endswith("/ex99-1.htm")
    assert note == "ex99"


def test_pick_exhibit_skips_pdf_only_exhibits():
    listing = {"directory": {"item": [
        {"name": "form8k.htm", "size": "50000"},
        {"name": "ex99-1.pdf", "size": "900000"},
    ]}}
    with patch.object(tc.requests, "get") as mock_get:
        mock_get.return_value.json.return_value = listing
        url, note = tc.pick_exhibit("320193", "0000320193-26-000001", "form8k.htm")
    assert url is None
    assert "PDF" in note


def test_tier_c_config_integrity():
    state = json.loads(Path("kpi_state.json").read_text(encoding="utf-8"))
    ciks = state.get("ciks", {})
    valid_units = {"usd", "usd_per_share", "percent", "count"}
    for ticker, kpis in TIER_C_KPIS.items():
        assert ticker in ciks, f"{ticker} missing from kpi_state.json CIK map"
        keys = [k["kpi_key"] for k in kpis]
        assert len(keys) == len(set(keys)), f"duplicate kpi_key for {ticker}"
        for kpi in kpis:
            assert kpi["unit"] in valid_units
            assert kpi["hint"].strip()


def test_committed_tier_c_json_shape():
    payload = json.loads(
        Path("apps/web/lib/server/kpi-tier-c-data.json").read_text(encoding="utf-8")
    )
    assert isinstance(payload.get("companies"), dict)
    for company in payload["companies"].values():
        for kpi in company.get("kpis", {}).values():
            assert kpi.get("status") in ("pending_review", "approved", "rejected")
