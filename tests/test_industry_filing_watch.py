"""Tests for the Industries filing watch (SEC-53): CIK loading from the
committed config, submissions-based detection, and the commit-state path.
atom parsing / diff-against-state are exercised in test_kpi_filing_watch.py
(reused as-is from kpi_filing_watch.py - not re-tested here). Network-free."""

from __future__ import annotations

import json

import industry_filing_watch as watch


def test_load_ciks_from_config_reads_ticker_cik_pairs(tmp_path):
    config_path = tmp_path / "industry-config.json"
    config_path.write_text(json.dumps({
        "industries": [
            {"tickers": [{"ticker": "AAPL", "cik": "320193"}, {"ticker": "MSFT", "cik": "789019"}]},
            {"tickers": [{"ticker": "NVDA", "cik": "1045810"}]},
        ]
    }))
    ciks = watch.load_ciks_from_config(str(config_path))
    assert ciks == {"AAPL": "320193", "MSFT": "789019", "NVDA": "1045810"}


def test_load_ciks_from_config_skips_entries_missing_ticker_or_cik(tmp_path):
    config_path = tmp_path / "industry-config.json"
    config_path.write_text(json.dumps({
        "industries": [{"tickers": [{"ticker": "AAPL", "cik": "320193"}, {"ticker": "NOCIK"}]}]
    }))
    ciks = watch.load_ciks_from_config(str(config_path))
    assert ciks == {"AAPL": "320193"}


def test_detect_via_submissions_flags_new_10q(monkeypatch):
    ciks = {"AAPL": "320193"}
    latest = {"AAPL": {"form": "10-Q", "accession": "old-accn"}}

    def fake_fetch(url):
        return json.dumps({
            "filings": {"recent": {
                "form": ["10-Q", "8-K"],
                "accessionNumber": ["new-accn", "other-accn"],
                "filingDate": ["2026-08-01", "2026-07-15"],
            }}
        })

    monkeypatch.setattr(watch, "_fetch", fake_fetch)
    monkeypatch.setattr(watch, "THROTTLE_S", 0)
    changed = watch.detect_via_submissions(ciks, latest)
    assert changed == {"AAPL": {"form": "10-Q", "accession": "new-accn", "filed": "2026-08-01"}}


def test_detect_via_submissions_no_change_when_accession_matches(monkeypatch):
    ciks = {"AAPL": "320193"}
    latest = {"AAPL": {"form": "10-Q", "accession": "same-accn"}}

    def fake_fetch(url):
        return json.dumps({
            "filings": {"recent": {
                "form": ["10-Q"], "accessionNumber": ["same-accn"], "filingDate": ["2026-08-01"],
            }}
        })

    monkeypatch.setattr(watch, "_fetch", fake_fetch)
    monkeypatch.setattr(watch, "THROTTLE_S", 0)
    assert watch.detect_via_submissions(ciks, latest) == {}


def test_detect_via_submissions_one_ticker_failure_does_not_abort(monkeypatch):
    ciks = {"A": "1", "B": "2"}
    latest = {}

    def fake_fetch(url):
        if "0000000001" in url:
            raise RuntimeError("timeout")
        return json.dumps({"filings": {"recent": {"form": ["10-K"], "accessionNumber": ["x"], "filingDate": ["2026-01-01"]}}})

    monkeypatch.setattr(watch, "_fetch", fake_fetch)
    monkeypatch.setattr(watch, "THROTTLE_S", 0)
    changed = watch.detect_via_submissions(ciks, latest)
    assert list(changed) == ["B"]


def test_main_commit_state_writes_only_on_change(tmp_path, monkeypatch):
    config_path = tmp_path / "industry-config.json"
    config_path.write_text(json.dumps({"industries": [{"tickers": [{"ticker": "AAPL", "cik": "1"}]}]}))
    state_path = tmp_path / "industry_state.json"
    state_path.write_text(json.dumps({"latest": {}}))

    monkeypatch.setattr(watch, "detect_via_atom", lambda ciks, latest: {"AAPL": {"form": "10-Q", "accession": "a1", "filed": "2026-08-01"}})
    monkeypatch.setattr(
        "sys.argv",
        ["industry_filing_watch.py", "--detect", "--commit-state", "--state", str(state_path), "--config", str(config_path)],
    )

    assert watch.main() == 0
    written = json.loads(state_path.read_text())
    assert written["latest"]["AAPL"]["accession"] == "a1"


def test_main_without_commit_state_leaves_state_file_untouched(tmp_path, monkeypatch):
    config_path = tmp_path / "industry-config.json"
    config_path.write_text(json.dumps({"industries": [{"tickers": [{"ticker": "AAPL", "cik": "1"}]}]}))
    state_path = tmp_path / "industry_state.json"
    state_path.write_text(json.dumps({"latest": {}}))

    monkeypatch.setattr(watch, "detect_via_atom", lambda ciks, latest: {"AAPL": {"form": "10-Q", "accession": "a1", "filed": "2026-08-01"}})
    monkeypatch.setattr("sys.argv", ["industry_filing_watch.py", "--detect", "--state", str(state_path), "--config", str(config_path)])

    assert watch.main() == 0
    assert json.loads(state_path.read_text()) == {"latest": {}}


def test_committed_state_file_has_sane_coverage_and_shape():
    """Bootstrapped via --full --commit-state against live EDGAR. Not every
    tracked ticker necessarily has a resolvable 10-Q/10-K (e.g. 20-F/40-F-only
    foreign filers), so this checks broad coverage rather than 1:1 parity
    with the ticker universe - unlike kpi_state.json's tighter 22/22 check,
    which tracks a small hand-picked set where every member does file one."""
    with open("industry_state.json", encoding="utf-8") as handle:
        state = json.load(handle)
    with open("apps/web/lib/server/industry-config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    tracked = {t["ticker"] for ind in config["industries"] for t in ind["tickers"]}
    latest = state["latest"]
    assert set(latest).issubset(tracked), "state has entries for tickers no longer in the universe"
    assert len(latest) / len(tracked) > 0.9, "state coverage dropped well below the ~96% bootstrap baseline"
    # Two legitimate entry shapes, by design: the --full submissions path
    # records a "filed" date, the hourly --detect atom path does not (the
    # shared differ in kpi_filing_watch only captures form + accession from
    # the feed). Change detection keys entirely on "accession", so "filed" is
    # informational - assert the load-bearing fields always, and "filed" only
    # when the submissions path was what wrote the entry.
    for ticker, info in latest.items():
        assert info["form"] in ("10-Q", "10-K"), ticker
        assert info["accession"], ticker
        if "filed" in info:
            assert info["filed"], ticker
