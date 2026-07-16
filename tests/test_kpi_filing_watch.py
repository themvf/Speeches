"""Tests for the KPI filing watch (SEC-34): atom parsing, state diffing,
and the snapshot merge path. Network-free."""

from __future__ import annotations

import json

import build_kpi_snapshot
import kpi_filing_watch as watch

ATOM_SAMPLE = """<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
<title>10-Q - GENERAL ELECTRIC CO (0000040545) (Filer)</title>
<summary type="html">&lt;b&gt;Filed:&lt;/b&gt; 2026-07-16 &lt;b&gt;AccNo:&lt;/b&gt; 0000040545-26-000049</summary>
<id>urn:tag:sec.gov,2008:accession-number=0000040545-26-000049</id>
</entry>
<entry>
<title>10-Q/A - SOMECO AMENDED (0000999999) (Filer)</title>
<id>urn:tag:sec.gov,2008:accession-number=0000999999-26-000001</id>
</entry>
<entry>
<title>10-K - Apple Inc. (0000320193) (Filer)</title>
<summary type="html">&lt;b&gt;Filed:&lt;/b&gt; 2026-07-16 &lt;b&gt;AccNo:&lt;/b&gt; 0000320193-26-000099</summary>
<id>urn:tag:sec.gov,2008:accession-number=0000320193-26-000099</id>
</entry>
</feed>"""


def test_parse_atom_entries_extracts_exact_forms_and_skips_amendments():
    entries = watch.parse_atom_entries(ATOM_SAMPLE)
    assert ("10-Q", "40545", "0000040545-26-000049") in entries
    assert ("10-K", "320193", "0000320193-26-000099") in entries
    assert not any(cik == "999999" for _, cik, _ in entries)  # 10-Q/A ignored


def test_diff_against_state_flags_only_tracked_and_changed():
    ciks = {"AAPL": "320193", "GE_NOT_TRACKED_ELSEWHERE": "111111"}
    latest = {"AAPL": {"form": "10-Q", "accession": "0000320193-26-000013"}}
    seen = watch.parse_atom_entries(ATOM_SAMPLE)
    changed = watch.diff_against_state(seen, ciks, latest)
    # GE isn't in our CIK map -> ignored; AAPL has a NEW accession -> flagged.
    assert list(changed) == ["AAPL"]
    assert changed["AAPL"]["accession"] == "0000320193-26-000099"


def test_diff_against_state_no_change_is_empty():
    ciks = {"AAPL": "320193"}
    latest = {"AAPL": {"form": "10-K", "accession": "0000320193-26-000099"}}
    changed = watch.diff_against_state(watch.parse_atom_entries(ATOM_SAMPLE), ciks, latest)
    assert changed == {}


def test_merge_into_existing_preserves_untouched_companies_and_order(tmp_path):
    existing = {
        "generatedAt": "old", "source": "s",
        "companies": {
            "AAPL": {"name": "Apple", "kpis": {"eps": {"label": "EPS", "unit": "usd_per_share", "series": [1]}}},
            "MSFT": {"name": "Microsoft", "kpis": {"eps": {"label": "EPS", "unit": "usd_per_share", "series": [2]}}},
            "NVDA": {"name": "NVIDIA", "kpis": {"eps": {"label": "EPS", "unit": "usd_per_share", "series": [3]}}},
        },
    }
    out = tmp_path / "snap.json"
    out.write_text(json.dumps(existing), encoding="utf-8")

    fresh = {"generatedAt": "new", "source": "s",
             "companies": {"MSFT": {"name": "Microsoft", "kpis": {"eps": {"label": "EPS", "unit": "usd_per_share", "series": [99]}}}}}
    merged = build_kpi_snapshot.merge_into_existing(fresh, str(out))

    assert list(merged["companies"]) == ["AAPL", "MSFT", "NVDA"]  # order kept
    assert merged["companies"]["MSFT"]["kpis"]["eps"]["series"] == [99]  # updated
    assert merged["companies"]["AAPL"]["kpis"]["eps"]["series"] == [1]   # untouched
    assert merged["generatedAt"] == "new"


def test_state_file_ships_all_tracked_companies():
    import kpi_config
    with open("kpi_state.json", encoding="utf-8") as handle:
        state = json.load(handle)
    assert set(state["ciks"]) == set(kpi_config.COMPANY_KPIS)
    assert set(state["latest"]) == set(kpi_config.COMPANY_KPIS)
    for info in state["latest"].values():
        assert info["form"] in ("10-Q", "10-K") and info["accession"]
