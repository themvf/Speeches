#!/usr/bin/env python3
"""SEC-34: detect new 10-Q/10-K filings for the CBOE KPI companies.

Python-stdlib ONLY (urllib, no pip install) so the hourly no-change path on
GitHub Actions runs in seconds. Two detection modes:

  --detect  EDGAR market-wide "current filings" Atom feed
            (action=getcurrent), one request per form type = 2 requests,
            filtered to our CIKs. Near-real-time (entries appear within
            minutes of EDGAR acceptance) but window-limited to the most
            recent ~100 filings per form.
  --full    Per-CIK data.sec.gov submissions JSON (22 requests). Complete
            but heavier - the once-daily reconciliation that catches
            anything the hourly feed window missed.

State lives in the committed kpi_state.json: {"ciks": {ticker: cik},
"latest": {ticker: {form, accession, filed}}}. The script prints changed
tickers (space-separated) on stdout; diagnostics go to stderr. It never
writes state unless --commit-state is passed - the workflow rebuilds the
snapshot first and only then commits state + snapshot together, so a failed
rebuild can't strand a filing as "seen but not ingested".

Amendments (10-Q/A, 10-K/A) are ignored by exact form match: they rarely
carry full XBRL (see the Tesla 10-K/A warnings in the pilot) and the
original filing's numbers are what CBOE KPIs track.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from typing import Dict, List, Tuple

STATE_PATH = "kpi_state.json"
FORMS = ("10-Q", "10-K")
USER_AGENT = "PolicyResearchHub KPI watch (joshbandes@gmail.com)"
ATOM_URL = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent"
            "&type={form}&company=&dateb=&owner=include&count=100&output=atom")
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"

_ENTRY_RE = re.compile(
    r"<title>(?P<form>10-[QK])\s+-\s+.*?\((?P<cik>\d{10})\)\s+\(Filer\)</title>.*?"
    r"accession-number=(?P<accession>[\d-]+)",
    re.DOTALL,
)


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_atom_entries(xml: str) -> List[Tuple[str, str, str]]:
    """(form, cik-no-leading-zeros, accession) for exact 10-Q/10-K entries.
    The regex anchors on '(Filer)' and the accession urn, both stable feed
    features; amendments have form '10-Q/A' in the title and don't match."""
    out = []
    for m in _ENTRY_RE.finditer(xml):
        out.append((m.group("form"), m.group("cik").lstrip("0"), m.group("accession")))
    return out


def diff_against_state(
    seen: List[Tuple[str, str, str]],
    ciks: Dict[str, str],
    latest: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    """Tickers whose newest observed filing isn't the recorded one."""
    by_cik = {str(cik).lstrip("0"): ticker for ticker, cik in ciks.items()}
    changed: Dict[str, Dict[str, str]] = {}
    for form, cik, accession in seen:
        ticker = by_cik.get(cik)
        if ticker is None:
            continue
        if latest.get(ticker, {}).get("accession") == accession:
            continue
        # First match wins per ticker within one run (feed is newest-first).
        changed.setdefault(ticker, {"form": form, "accession": accession})
    return changed


def detect_via_atom(ciks: Dict[str, str], latest: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    seen: List[Tuple[str, str, str]] = []
    for form in FORMS:
        try:
            seen.extend(parse_atom_entries(_fetch(ATOM_URL.format(form=form))))
        except Exception as exc:
            print(f"[watch] atom fetch failed for {form}: {exc}", file=sys.stderr)
    return diff_against_state(seen, ciks, latest)


def detect_via_submissions(ciks: Dict[str, str], latest: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    changed: Dict[str, Dict[str, str]] = {}
    for ticker, cik in ciks.items():
        try:
            data = json.loads(_fetch(SUBMISSIONS_URL.format(cik=cik)))
        except Exception as exc:
            print(f"[watch] submissions fetch failed for {ticker}: {exc}", file=sys.stderr)
            continue
        recent = data.get("filings", {}).get("recent", {})
        for form, accession, filed in zip(recent.get("form", []), recent.get("accessionNumber", []), recent.get("filingDate", [])):
            if form not in FORMS:
                continue
            if latest.get(ticker, {}).get("accession") != accession:
                changed[ticker] = {"form": form, "accession": accession, "filed": filed}
            break  # newest 10-Q/10-K only
    return changed


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--detect", action="store_true", help="Atom feed (2 requests, near-real-time)")
    mode.add_argument("--full", action="store_true", help="Per-CIK submissions (complete reconciliation)")
    parser.add_argument("--commit-state", action="store_true",
                        help="Write detected accessions into kpi_state.json (run AFTER a successful rebuild)")
    parser.add_argument("--state", default=STATE_PATH)
    args = parser.parse_args(argv)

    with open(args.state, encoding="utf-8") as handle:
        state = json.load(handle)
    ciks: Dict[str, str] = state.get("ciks", {})
    latest: Dict[str, Dict[str, str]] = state.get("latest", {})

    changed = detect_via_atom(ciks, latest) if args.detect else detect_via_submissions(ciks, latest)

    for ticker, info in sorted(changed.items()):
        print(f"[watch] {ticker}: new {info['form']} {info['accession']}", file=sys.stderr)
    if args.commit_state and changed:
        for ticker, info in changed.items():
            latest[ticker] = info
        state["latest"] = latest
        with open(args.state, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=1, sort_keys=True)
        print(f"[watch] state updated for {len(changed)} ticker(s)", file=sys.stderr)

    print(" ".join(sorted(changed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
