"""Tests for filing_catalyst_sync (SEC-50): atom/idx parsing, exact-form
filtering, issuer-vs-owner handling, and the Form 4 chip summary."""

from __future__ import annotations

import filing_catalyst_sync as sync

TRACKED = {"1045810": "NVDA", "320193": "AAPL"}

ATOM_SAMPLE = """<feed>
<entry><title>8-K - NVIDIA CORP (0001045810) (Filer)</title>
<id>urn:tag:sec.gov,2008:accession-number=0001045810-26-000123</id></entry>
<entry><title>424B2 - BofA Finance LLC (0001682472) (Filer)</title>
<id>urn:tag:sec.gov,2008:accession-number=0001682472-26-000999</id></entry>
<entry><title>4 - Apple Inc. (0000320193) (Issuer)</title>
<id>urn:tag:sec.gov,2008:accession-number=0000320193-26-000456</id></entry>
<entry><title>4 - COOK TIMOTHY D (0001214156) (Reporting)</title>
<id>urn:tag:sec.gov,2008:accession-number=0000320193-26-000456</id></entry>
<entry><title>8-K - UNTRACKED CO (0009999999) (Filer)</title>
<id>urn:tag:sec.gov,2008:accession-number=0009999999-26-000001</id></entry>
<entry><title>10-K - NVIDIA CORP (0001045810) (Filer)</title>
<id>urn:tag:sec.gov,2008:accession-number=0001045810-26-000777</id></entry>
</feed>"""


def test_parse_atom_exact_forms_tracked_issuers_only():
    rows = sync.parse_atom(ATOM_SAMPLE, TRACKED)
    by_accession = {r["accession"]: r for r in rows}
    assert set(by_accession) == {"0001045810-26-000123", "0000320193-26-000456"}
    assert by_accession["0001045810-26-000123"]["ticker"] == "NVDA"
    assert by_accession["0001045810-26-000123"]["form"] == "8-K"
    # The Form 4 row must come from the ISSUER entry (Apple), never the
    # Reporting-owner CIK, and 424B2 / 10-K / untracked are all excluded.
    assert by_accession["0000320193-26-000456"]["ticker"] == "AAPL"
    assert by_accession["0000320193-26-000456"]["form"] == "4"


IDX_SAMPLE = """Form Type   Company Name    CIK    Date Filed   File Name
---------------------------------------------------------------
4                APPLE INC                                                     320193      20260716    edgar/data/320193/0000320193-26-000456.txt
8-K              NVIDIA CORP                                                   1045810     20260716    edgar/data/1045810/0001045810-26-000123.txt
8-K/A            NVIDIA CORP                                                   1045810     20260716    edgar/data/1045810/0001045810-26-000124.txt
4                RANDOM PERSON                                                 7777777     20260716    edgar/data/7777777/0007777777-26-000001.txt
10-Q             APPLE INC                                                     320193      20260716    edgar/data/320193/0000320193-26-000457.txt
"""


def test_parse_form_idx_exact_forms_and_tracked_only():
    rows = sync.parse_form_idx(IDX_SAMPLE, TRACKED)
    forms = {(r["ticker"], r["form"]) for r in rows}
    assert forms == {("AAPL", "4"), ("NVDA", "8-K")}  # 8-K/A, 10-Q, untracked dropped
    assert all(r["filed_at"].year == 2026 for r in rows)


def test_filing_index_url_shape():
    url = sync.filing_index_url("1045810", "0001045810-26-000123")
    assert url == ("https://www.sec.gov/Archives/edgar/data/1045810/"
                   "000104581026000123/0001045810-26-000123-index.htm")


def test_summarize_form4_nets_buys_and_sells():
    assert sync.summarize_form4([("S", 10000, 250.0)]) == "Insider sold $2.5M"
    assert sync.summarize_form4([("P", 2000, 50.0)]) == "Insider bought $100K"
    # Mixed: 5k awarded at 0 + 3k sold at 100 -> net bought 2k, value -300k -> sold label? No:
    # net shares positive => "bought", dollars = |0 - 300k| = $300K.
    assert sync.summarize_form4([("A", 5000, None), ("S", 3000, 100.0)]) == "Insider bought $300K"
    assert sync.summarize_form4([("X", 1, 1.0)]) == "Insider transaction"
    assert sync.summarize_form4([]) == "Insider transaction"
