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
