"""Tests for check_stale_tickers.py (SEC-53 weekly upkeep: flagging Industries
universe tickers that are no longer public). All network calls mocked -
classify_candidate is pure and tested directly; check_universe is tested with
monkeypatched fetchers."""

from __future__ import annotations

import json
import os

import check_stale_tickers as checker


def test_classify_candidate_cik_not_found_is_high_confidence():
    result = checker.classify_candidate("DEAD", "0000000001", None)
    assert result["reason"] == "cik_not_found"
    assert result["confidence"] == "high"


def test_classify_candidate_detects_rename():
    submissions = {"tickers": ["NEWCO"], "filings": {"recent": {"form": []}}}
    result = checker.classify_candidate("OLDCO", "0000000001", submissions)
    assert result["reason"] == "renamed"
    assert result["suggestedNewTicker"] == "NEWCO"


def test_classify_candidate_detects_deregistration_form():
    submissions = {"tickers": ["ABC"], "filings": {"recent": {"form": ["8-K", "15-12B"]}}}
    result = checker.classify_candidate("ABC", "0000000001", submissions)
    assert result["reason"] == "deregistered"
    assert result["confidence"] == "high"


def test_classify_candidate_no_active_ticker():
    submissions = {"tickers": [], "filings": {"recent": {"form": ["10-K"]}}}
    result = checker.classify_candidate("ABC", "0000000001", submissions)
    assert result["reason"] == "no_active_ticker"


def test_classify_candidate_uncertain_when_still_listed_and_no_dereg():
    # Missing from the bulk file, but the CIK's own feed still shows the same
    # ticker and no deregistration filing - flagged as low-confidence, not a
    # delisting classification.
    submissions = {"tickers": ["ABC"], "filings": {"recent": {"form": ["10-Q"]}}}
    result = checker.classify_candidate("ABC", "0000000001", submissions)
    assert result["reason"] == "uncertain"
    assert result["confidence"] == "low"


def test_check_universe_skips_tickers_matching_bulk_file(tmp_path, monkeypatch):
    config_path = tmp_path / "industry-config.json"
    config_path.write_text(json.dumps({
        "industries": [
            {"tickers": [
                {"ticker": "OK", "name": "Still Public Co", "cik": "1"},
                {"ticker": "GONE", "name": "Delisted Co", "cik": "2"},
            ]}
        ]
    }))

    monkeypatch.setattr(checker, "_fetch_bulk_tickers", lambda: {"OK": "0000000001"})
    monkeypatch.setattr(checker, "_fetch_submissions", lambda cik: None)  # GONE's CIK 404s
    monkeypatch.setattr(checker, "_corroborate_with_yahoo", lambda flagged: None)
    monkeypatch.setattr(checker, "THROTTLE_S", 0)

    result = checker.check_universe(str(config_path))
    assert result["checkedCount"] == 2
    assert result["flaggedCount"] == 1
    assert result["candidates"][0]["ticker"] == "GONE"
    assert result["candidates"][0]["reason"] == "cik_not_found"


def test_check_universe_one_ticker_network_failure_does_not_abort_run(tmp_path, monkeypatch):
    config_path = tmp_path / "industry-config.json"
    config_path.write_text(json.dumps({
        "industries": [{"tickers": [
            {"ticker": "A", "name": "A Co", "cik": "1"},
            {"ticker": "B", "name": "B Co", "cik": "2"},
        ]}]
    }))

    def flaky_submissions(cik):
        if cik == "0000000001":
            raise RuntimeError("timeout")
        return None

    monkeypatch.setattr(checker, "_fetch_bulk_tickers", lambda: {})
    monkeypatch.setattr(checker, "_fetch_submissions", flaky_submissions)
    monkeypatch.setattr(checker, "_corroborate_with_yahoo", lambda flagged: None)
    monkeypatch.setattr(checker, "THROTTLE_S", 0)

    result = checker.check_universe(str(config_path))
    assert result["flaggedCount"] == 2
    reasons = {c["ticker"]: c["reason"] for c in result["candidates"]}
    assert reasons["A"] == "check_failed"
    assert reasons["B"] == "cik_not_found"


def test_main_writes_review_file(tmp_path, monkeypatch):
    config_path = tmp_path / "industry-config.json"
    config_path.write_text(json.dumps({"industries": [{"tickers": [{"ticker": "OK", "name": "Ok Co", "cik": "1"}]}]}))
    output_path = tmp_path / "ticker_prune_review.json"

    monkeypatch.setattr(checker, "_fetch_bulk_tickers", lambda: {"OK": "0000000001"})
    monkeypatch.setattr("sys.argv", ["check_stale_tickers.py", "--config", str(config_path), "--output", str(output_path)])

    assert checker.main() == 0
    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written["checkedCount"] == 1
    assert written["flaggedCount"] == 0
