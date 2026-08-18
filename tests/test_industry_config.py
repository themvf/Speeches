"""Tests for the Industries tab config (SEC-53): the pure SIC grouping and
the integrity of the committed industry-config.json the web route imports."""

from __future__ import annotations

import json
import os

import build_industry_config as builder

CONFIG_PATH = os.path.join("apps", "web", "lib", "server", "industry-config.json")


def test_group_by_industry_groups_and_sorts():
    records = [
        {"ticker": "NVDA", "name": "NVIDIA", "cik": "1", "sic": "3674", "sic_description": "Semiconductors"},
        {"ticker": "AMD", "name": "AMD", "cik": "2", "sic": "3674", "sic_description": "Semiconductors"},
        {"ticker": "INTC", "name": "Intel", "cik": "3", "sic": "3674", "sic_description": "Semiconductors"},
        {"ticker": "JPM", "name": "JPMorgan", "cik": "4", "sic": "6021", "sic_description": "National Commercial Banks"},
        {"ticker": "BAC", "name": "BofA", "cik": "5", "sic": "6021", "sic_description": "National Commercial Banks"},
        {"ticker": "ODD", "name": "Oddball", "cik": "6", "sic": "", "sic_description": ""},
    ]
    industries = builder.group_by_industry(records)
    assert [i["label"] for i in industries] == ["Semiconductors", "National Commercial Banks", "Unclassified"]
    semis = industries[0]
    assert [t["ticker"] for t in semis["tickers"]] == ["AMD", "INTC", "NVDA"]  # ticker-sorted
    assert semis["sic"] == "3674"


def test_resolve_filed_date_matches_accession():
    submissions = {
        "filings": {
            "recent": {
                "accessionNumber": ["0001-26-000111", "0001-26-000222"],
                "filingDate": ["2026-08-05", "2026-05-01"],
            }
        }
    }
    assert builder.resolve_filed_date(submissions, "0001-26-000111") == "2026-08-05"
    assert builder.resolve_filed_date(submissions, "0001-26-000222") == "2026-05-01"


def test_resolve_filed_date_missing_or_aged_out_accession_returns_none():
    submissions = {
        "filings": {
            "recent": {
                "accessionNumber": ["0001-26-000111"],
                "filingDate": ["2026-08-05"],
            }
        }
    }
    assert builder.resolve_filed_date(submissions, None) is None
    assert builder.resolve_filed_date(submissions, "") is None
    assert builder.resolve_filed_date(submissions, "0009-99-999999") is None  # not in "recent" window
    assert builder.resolve_filed_date({}, "0001-26-000111") is None  # malformed/empty payload


def test_fetch_frame_metric_keeps_accn_for_filed_date_lookup(monkeypatch):
    def fake_fetch_json(url):
        return {"data": [{"cik": 1, "val": 100.0, "end": "2026-06-30", "accn": "0001-26-000111"}]}

    monkeypatch.setattr(builder, "_fetch_json", fake_fetch_json)
    monkeypatch.setattr(builder, "QUARTERS", ["CY2026Q2"])
    monkeypatch.setattr(builder, "THROTTLE_S", 0)
    out = builder.fetch_frame_metric(["Revenues"], "USD")
    assert out["1"]["accn"] == "0001-26-000111"


def test_build_financials_carries_accn_through_to_output(monkeypatch):
    def fake_fetch_frame_metric(concepts, unit, taxonomy="us-gaap", instant=False):
        if "Revenue" in concepts[0] or concepts[0] == "Revenues":
            return {"1": {"val": 100.0, "end": "2026-06-30", "accn": "0001-26-000111"}}
        if concepts[0] == "NetIncomeLoss":
            return {"1": {"val": 20.0, "end": "2026-06-30", "accn": "0001-26-000111"}}
        return {}

    monkeypatch.setattr(builder, "fetch_frame_metric", fake_fetch_frame_metric)
    financials = builder.build_financials()
    assert financials["1"]["accn"] == "0001-26-000111"
    assert financials["1"]["periodEnd"] == "2026-06-30"


def test_sub_industry_groups_have_no_duplicate_tickers():
    seen = {}
    for label, tickers in builder.SUB_INDUSTRY_GROUPS.items():
        for ticker in tickers:
            assert ticker not in seen, f"{ticker} in both {seen.get(ticker)!r} and {label!r}"
            seen[ticker] = label
    assert len(builder.SUB_INDUSTRY_BY_TICKER) == sum(len(t) for t in builder.SUB_INDUSTRY_GROUPS.values())


def test_committed_config_sub_industry_coverage_is_complete_for_large_sics():
    """Every ticker in the 13 large SIC buckets the sub-industry taxonomy
    targets must be tagged - a ticker silently missing its tag would fall
    into an unlabeled "Other" group in the UI instead of erroring, so this
    is the guard that actually catches it. If the universe grows a new
    member of one of these SICs, extend SUB_INDUSTRY_GROUPS in
    build_industry_config.py rather than loosening this test."""
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        config = json.load(handle)
    target_sics = {"7372", "6798", "3674", "2834", "7389", "6021", "4911", "6199", "1311", "7370", "6282", "6331", "5812"}
    missing = []
    for industry in config["industries"]:
        if industry["sic"] not in target_sics:
            continue
        for entry in industry["tickers"]:
            if entry["ticker"] not in builder.SUB_INDUSTRY_BY_TICKER:
                missing.append((industry["label"], entry["ticker"]))
    assert not missing, f"tickers missing a subIndustry tag: {missing}"


def test_committed_config_integrity():
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        config = json.load(handle)
    assert config["industries"], "config must not be empty"
    seen = set()
    labels = set()
    for industry in config["industries"]:
        assert industry["label"]
        assert industry["label"] not in labels, f"duplicate industry label {industry['label']}"
        labels.add(industry["label"])
        assert industry["tickers"], f"empty industry {industry['label']}"
        for entry in industry["tickers"]:
            assert entry["ticker"] and entry["name"] and entry["cik"], entry
            assert entry["ticker"] not in seen, f"ticker {entry['ticker']} in two industries"
            seen.add(entry["ticker"])
    # SEC-56: the universe should stay broad enough for real peer groups.
    assert len(seen) >= 400, f"universe shrank to {len(seen)}"
    # The CBOE KPI companies must all be classified.
    import kpi_config
    missing = set(kpi_config.COMPANY_KPIS) - seen
    assert not missing, f"KPI companies missing from industry config: {missing}"


def test_committed_config_has_meaningful_peer_depth():
    """SEC-56: the point of the expansion - most members should sit in a
    group with actual peers, not alone."""
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        config = json.load(handle)
    industries = config["industries"]
    assert sum(1 for i in industries if len(i["tickers"]) >= 3) >= 40
    total = sum(len(i["tickers"]) for i in industries)
    in_real_groups = sum(len(i["tickers"]) for i in industries if len(i["tickers"]) >= 3)
    assert in_real_groups / total >= 0.7, "most tickers should have >=2 peers"


def test_committed_config_financials_reconcile():
    """Revenue - profit must equal the derived expenses exactly, and share
    counts must be positive - the peer table's three columns have to tie."""
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        config = json.load(handle)
    checked = 0
    for industry in config["industries"]:
        for entry in industry["tickers"]:
            rev, profit, exp = entry.get("revenue"), entry.get("profit"), entry.get("expenses")
            if exp is not None:
                assert rev is not None and profit is not None, entry["ticker"]
                assert abs((rev - profit) - exp) < 1.0, f"{entry['ticker']} expenses don't reconcile"
                assert entry.get("periodEnd"), f"{entry['ticker']} has financials but no periodEnd"
                checked += 1
            shares = entry.get("sharesOutstanding")
            if shares is not None:
                assert shares > 0, f"{entry['ticker']} non-positive shares outstanding"
    assert checked >= 300, f"only {checked} companies carry reconciled financials"
